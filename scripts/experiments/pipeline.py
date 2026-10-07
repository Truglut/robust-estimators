"""Preprocessing and estimator execution shared by generated and stored images."""

from typing import overload
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .configs import (
    EstimationConfig,
    EstimatorConfig,
    InitializationStrategy,
    PreprocessingConfig,
    SimulationConfig,
)
from simple_mestimators.domain import ImageBatch, EstimatorResult
from simple_mestimators.estimators.construction import build_estimator
from simple_mestimators.estimators.noise_corrected_cosine import (
    NoiseCorrectedCosineEstimator,
)


@dataclass
class PreparedDataset:
    preprocessed_images: np.ndarray
    labels: np.ndarray | None = None
    ground_truth: np.ndarray | None = None
    mask: np.ndarray | None = None


def preprocess_images(
    images: np.ndarray, cfg: PreprocessingConfig, mask: np.ndarray | None
) -> np.ndarray:
    """Mask, then optionally standardize over all pixels, returning a copy.

    Constant images become zero when standardized. Centering after masking can
    make pixels outside the mask nonzero. Simulation-only pre-noise
    standardization remains the responsibility of dataset_builder.
    """
    images = np.array(images, dtype=np.result_type(images.dtype, np.float32), copy=True)
    if mask is not None:
        images *= mask
    if cfg.standardize in ("after", "both"):
        images -= images.mean(axis=(-2, -1), keepdims=True)
        std = images.std(axis=(-2, -1), keepdims=True)
        images /= np.where(std > 0, std, 1.0)
    return images


def prepare_initial_reference(
    dataset: PreparedDataset, initialization_strategy: InitializationStrategy
) -> np.ndarray:
    images = dataset.preprocessed_images
    if initialization_strategy == "mean":
        return images.mean(axis=0)
    if initialization_strategy == "median":
        return np.median(images, axis=0)
    if initialization_strategy == "mean-good":
        if dataset.labels is None or not np.any(dataset.labels == 0):
            raise ValueError("mean-good initialization requires labels and good images")
        return images[dataset.labels == 0].mean(axis=0)
    if initialization_strategy == "ground-truth":
        if dataset.ground_truth is None:
            raise ValueError(
                "ground-truth initialization requires a ground-truth image"
            )
        return dataset.ground_truth
    raise ValueError(
        f"Unsupported initialization strategy: {initialization_strategy!r}"
    )


def run_experiment(
    dataset: PreparedDataset,
    estimator_cfgs: list[EstimatorConfig],
    initialization_strategy: InitializationStrategy,
) -> Iterator[tuple[str, EstimatorResult]]:
    image_batch = ImageBatch(real=dataset.preprocessed_images, mask=dataset.mask)
    initial_reference = prepare_initial_reference(dataset, initialization_strategy)

    for cfg in estimator_cfgs:
        estimator = build_estimator(cfg.to_method_cfg(), image_batch=image_batch)
        if isinstance(estimator, NoiseCorrectedCosineEstimator):
            if initialization_strategy == "mean":
                initial_weights = np.full(
                    image_batch.n_images, 1.0 / image_batch.n_images
                )
            elif initialization_strategy == "mean-good":
                good = dataset.labels == 0
                initial_weights = good.astype(float) / good.sum()
            elif initialization_strategy == "ground-truth":
                initial_weights = None
            else:
                raise ValueError(
                    "Noise-corrected cosine requires mean, mean-good, or ground-truth "
                    "initialization; a median's noise covariance is not supported"
                )
            result = estimator.fit(
                image_batch,
                reference=initial_reference.copy(),
                initial_reference_weights=initial_weights,
            )
        else:
            result = estimator.fit(image_batch, reference=initial_reference.copy())
        yield cfg.name, result


def sample_mean_result(dataset: PreparedDataset) -> EstimatorResult:
    images = dataset.preprocessed_images
    return EstimatorResult(
        real_estimate=images.mean(axis=0),
        weights={"real": np.ones(images.shape[0], dtype=images.dtype)},
    )


@overload
def effective_config(cfg: SimulationConfig) -> SimulationConfig: ...


@overload
def effective_config(cfg: EstimationConfig) -> EstimationConfig: ...


def effective_config(
    cfg: SimulationConfig | EstimationConfig,
) -> SimulationConfig | EstimationConfig:
    """Resolve missing GMM random states without modifying the supplied config."""
    methods = deepcopy(cfg.estimators)
    for method in methods:
        if (
            method.estimator_type == "noise_corrected_cosine"
            and method.params.get("weighting", "gmm") == "gmm"
        ):
            gmm_params = method.params.setdefault("gmm_params", {})
            if gmm_params.get("random_state") is None:
                gmm_params["random_state"] = cfg.seed
        if method.estimator_type in ("recursive_gmm", "fourier_gmm"):
            if method.params.get("random_state") is None:
                method.params["random_state"] = cfg.seed
    return replace(cfg, estimators=methods)
