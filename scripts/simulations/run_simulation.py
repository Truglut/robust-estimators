from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from copy import deepcopy

import numpy as np
import mrcfile

from .cli import build_simulation_parser, parse_simulation_cfg
from .configs import (
    DatasetConfig,
    EstimatorConfig,
    PreprocessingConfig,
    SimulationConfig,
    InitializationStrategy,
)
from .dataset_builder import create_evaluation_dataset, get_dataset_labels
from .output import initialize_output, save_estimator_result, save_json, snr_directory

from simple_mestimators.masks import create_circular_mask
from simple_mestimators.domain import ImageBatch, EstimatorResult
from simple_mestimators.estimators.construction import build_estimator


@dataclass
class PreparedSimulationDataset:
    preprocessed_images: np.ndarray
    labels: np.ndarray
    ground_truth: np.ndarray


def preprocess_images(
    images: np.ndarray, cfg: PreprocessingConfig, mask: np.ndarray | None
) -> np.ndarray:
    """
    Mask, then optionally standardize over all pixels.

    A copy is returned. Constant images become zero when standardized. Centering
    after masking can make pixels outside the mask nonzero.
    """
    images = np.array(images, dtype=np.result_type(images.dtype, np.float32), copy=True)
    if mask is not None:
        images *= mask
    if cfg.standardize in ("after", "both"):
        images -= images.mean(axis=(-2, -1), keepdims=True)
        std = images.std(axis=(-2, -1), keepdims=True)
        images /= np.where(std > 0, std, 1.0)
    return images


def prepare_simulation_dataset(
    *,
    ground_truth: np.ndarray,
    misclassified_path: Path | None,
    dataset_cfg: DatasetConfig,
    preprocessing_cfg: PreprocessingConfig,
    mask: np.ndarray | None,
    snr: float,
    rng: np.random.Generator,
) -> PreparedSimulationDataset:
    images, dataset_ground_truth, labels = create_evaluation_dataset(
        reference=ground_truth,
        misclassified_path=misclassified_path,
        dataset_cfg=dataset_cfg,
        rng=rng,
        snr=snr,
        standardize_before_noise=preprocessing_cfg.standardize in ["before", "both"],
    )

    return PreparedSimulationDataset(
        preprocessed_images=preprocess_images(images, preprocessing_cfg, mask),
        labels=labels,
        ground_truth=preprocess_images(dataset_ground_truth, preprocessing_cfg, mask),
    )


def prepare_initial_reference(
    dataset: PreparedSimulationDataset,
    initialization_strategy: InitializationStrategy,
) -> np.ndarray:
    images = dataset.preprocessed_images
    if initialization_strategy == "mean":
        return np.mean(images, axis=0)

    if initialization_strategy == "median":
        return np.median(images, axis=0)

    if initialization_strategy == "mean-good":
        good_idx = dataset.labels == 0
        return np.mean(images[good_idx], axis=0)

    if initialization_strategy == "ground-truth":
        return dataset.ground_truth

    raise ValueError(
        f"Unsupported initialization strategy: '{initialization_strategy}'."
    )


# TODO:
# 1. define what should be the inputs and outputs of this function
# 2. copy structure from mestimators repo
# 3. add new code for saving the basic results: images, weights, labels.
def run_experiment(
    dataset: PreparedSimulationDataset,
    estimator_cfgs: list[EstimatorConfig],
    initialization_strategy: InitializationStrategy,
) -> Iterator[tuple[str, EstimatorResult]]:
    image_batch = ImageBatch(real=dataset.preprocessed_images)
    initial_reference = prepare_initial_reference(dataset, initialization_strategy)

    for cfg in estimator_cfgs:
        estimator = build_estimator(cfg.to_method_cfg(), image_batch=image_batch)
        yield cfg.name, estimator.fit(image_batch, reference=initial_reference.copy())


def effective_config(cfg: SimulationConfig) -> SimulationConfig:
    """Resolve missing GMM random states without modifying the supplied config."""
    methods = deepcopy(cfg.estimators)
    for method in methods:
        if method.estimator_type in ("recursive_gmm", "fourier_gmm"):
            if method.params.get("random_state") is None:
                method.params["random_state"] = cfg.seed
    return replace(cfg, estimators=methods)


def run_simulation(cfg: SimulationConfig) -> Path:
    cfg = effective_config(cfg)

    ground_truth = np.asarray(mrcfile.read(cfg.io.reference_path))
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected a 2D reference image, got {ground_truth.shape}")

    reference = ground_truth.copy()
    if cfg.dataset.standardize_reference:
        reference /= reference.std()

    mask_radius = cfg.preprocessing.mask_radius
    if mask_radius is None:
        h, w = ground_truth.shape
        mask_radius = max(h // 2, w // 2)
    mask = create_circular_mask(image_shape=ground_truth.shape, radius=mask_radius)

    labels = None
    root = None
    for snr_index, snr in enumerate(cfg.snrs):
        for run_index in range(cfg.n_runs):
            print(f"SNR={snr:g}, run {run_index + 1}/{cfg.n_runs}")
            seed = int(
                np.random.SeedSequence([cfg.seed, snr_index, run_index]).generate_state(
                    1, dtype=np.uint64
                )[0]
            )
            dataset = prepare_simulation_dataset(
                ground_truth=ground_truth,
                misclassified_path=cfg.io.misclassified_path,
                dataset_cfg=cfg.dataset,
                preprocessing_cfg=cfg.preprocessing,
                mask=mask,
                snr=snr,
                rng=np.random.default_rng(seed),
            )

            if root is None or labels is None:
                labels = dataset.labels
                root = initialize_output(
                    cfg, ground_truth=dataset.ground_truth, mask=mask, labels=labels
                )
            elif not np.array_equal(dataset.labels, labels):
                raise RuntimeError("Dataset label order changed between runs")

            # Save the sample mean for this run
            unitary_weights = np.ones(
                dataset.preprocessed_images.shape[0],
                dtype=dataset.preprocessed_images.dtype,
            )
            mean_result = EstimatorResult(
                real_estimate=dataset.preprocessed_images.mean(axis=0),
                weights={"real": unitary_weights},
            )
            directory = (
                root
                / "estimators"
                / "mean"
                / snr_directory(snr)
                / f"run_{run_index:03d}"
            )
            save_estimator_result(directory, mean_result, cfg.io, dataset_seed=seed)

            for name, result in run_experiment(
                dataset, cfg.estimators, cfg.initialization_strategy
            ):
                directory = (
                    root
                    / "estimators"
                    / name
                    / snr_directory(snr)
                    / f"run_{run_index:03d}"
                )
                save_estimator_result(directory, result, cfg.io, dataset_seed=seed)
                del result
            del dataset

    assert root is not None

    return root


def main() -> None:
    parser = build_simulation_parser()
    cfg = parse_simulation_cfg(parser.parse_args())
    root_results_dir = run_simulation(cfg)
    print(f"Saved results to {root_results_dir}.")


if __name__ == "__main__":
    main()
