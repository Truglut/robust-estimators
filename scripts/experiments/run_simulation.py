from pathlib import Path
from dataclasses import replace

import numpy as np
import mrcfile

from .cli import build_simulation_parser, parse_simulation_cfg
from .configs import (
    DatasetConfig,
    PreprocessingConfig,
    SimulationConfig,
)
from .dataset_builder import LABELS, create_evaluation_dataset
from .output import initialize_output, save_estimator_result, snr_directory
from .pipeline import (
    PreparedDataset,
    preprocess_images,
    prepare_initial_reference,
    run_experiment,
    effective_config,
    sample_mean_result,
)

from simple_mestimators.masks import create_circular_mask

# Compatibility for callers that imported this class from run_simulation.
PreparedSimulationDataset = PreparedDataset


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
        mask=mask,
    )


def run_simulation(cfg: SimulationConfig) -> Path:
    cfg = effective_config(cfg)

    ground_truth = np.asarray(mrcfile.read(cfg.io.reference_path))
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected a 2D reference image, got {ground_truth.shape}")

    mask_radius = cfg.preprocessing.mask_radius
    if mask_radius is None:
        h, w = ground_truth.shape
        mask_radius = min(h // 2, w // 2)
        cfg = replace(
            cfg, preprocessing=replace(cfg.preprocessing, mask_radius=mask_radius)
        )
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
                    cfg,
                    ground_truth=dataset.ground_truth,
                    mask=mask,
                    labels=labels,
                    label_names=LABELS,
                )
            elif not np.array_equal(dataset.labels, labels):
                raise RuntimeError("Dataset label order changed between runs")

            mean_result = sample_mean_result(dataset)
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
