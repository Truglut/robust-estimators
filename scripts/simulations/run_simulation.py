from dataclasses import dataclass

import numpy as np

from .cli import build_simulation_parser, parse_simulation_args
from .configs import (
    IOConfig,
    DatasetConfig,
    EstimatorConfig,
    PreprocessingConfig,
    SimulationConfig,
    StandardizeOption,
    InitializationStrategy,
)
from .dataset_builder import create_evaluation_dataset

from simple_mestimators.masks import create_circular_mask


@dataclass
class PreparedSimulationDataset:
    raw_images: np.ndarray
    preprocessed_images: np.ndarray
    ground_truth: np.ndarray
    labels: np.ndarray
    mask: np.ndarray


def prepare_simulation_dataset(
    *,
    dataset_cfg: DatasetConfig,
    io_cfg: IOConfig,
    preprocessing_cfg: PreprocessingConfig,
    snr: float,
    rng: np.random.Generator,
) -> PreparedSimulationDataset:
    images, ground_truth, labels = create_evaluation_dataset(
        dataset_cfg=dataset_cfg,
        io_cfg=io_cfg,
        rng=rng,
        snr=snr,
        standardize_before_noise=preprocessing_cfg.standardize in ["before", "both"],
    )

    mask_radius = preprocessing_cfg.mask_radius
    if mask_radius is None:
        h, w = images.shape[1:]
        mask_radius = max(h // 2, w // 2)
    mask = create_circular_mask(image_shape=ground_truth.shape, radius=mask_radius)
    preprocessed = images * mask

    if preprocessing_cfg.standardize in ["after", "both"]:
        preprocessed -= preprocessed.mean(axis=(-2, -1), keepdims=True)
        preprocessed /= preprocessed.std(axis=(-2, -1), keepdims=True)

    return PreparedSimulationDataset(
        raw_images=images,
        preprocessed_images=preprocessed,
        ground_truth=ground_truth,
        labels=labels,
        mask=mask,
    )


def prepare_initial_reference(
    dataset: PreparedSimulationDataset, initialization_strategy: InitializationStrategy
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
def run_experiment() -> None:
    pass
