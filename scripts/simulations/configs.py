from typing import Literal

from pathlib import Path
from dataclasses import dataclass

StandardizeOption = Literal["before", "after", "both", "none"] | None
InitializationStrategy = Literal["mean", "median", "mean-good", "ground-truth"]


@dataclass
class IOConfig:
    reference_path: Path
    misclassified_path: Path | None = None
    base_results_path: Path | None = None
    save_weights: bool = True
    save_images: bool = True
    save_labels: bool = True


@dataclass
class DatasetConfig:
    n_copies: int
    n_rotated: int = 0
    n_shifted: int = 0
    n_misclassified: int = 0
    n_noise: int = 0
    max_rotation_inliers: float | None = None

    min_rotation_outliers: float | None = None
    max_rotation_outliers: float | None = None

    # If either of these is None, no rotation will be applied to misclassified images
    min_rotation_misclassified: float | None = None
    max_rotation_misclassified: float | None = None

    min_shift_outliers: float | None = None
    max_shift_outliers: float | None = None

    standardize_reference: bool = True

    def __post_init__(self):
        missing_rotation_parameters = (
            self.min_rotation_outliers is None or self.max_rotation_outliers is None
        )
        if self.n_rotated > 0 and missing_rotation_parameters:
            raise ValueError(
                "Requested rotated outliers but at least one of min_rotation_outliers "
                "or max_rotation_outlirs was not provided."
            )

        missing_shift_parameters = (
            self.min_shift_outliers is None or self.max_shift_outliers is None
        )
        if self.n_shifted > 0 and missing_shift_parameters:
            raise ValueError(
                "Requested shifted outliers but at least one of min_shift_outliers "
                "or max_shift outliers was not provided."
            )

        if (self.min_rotation_outliers is None) != (self.max_rotation_outliers is None):
            raise ValueError(
                "min_rotation_outliers and max_rotation_outliers must be provided together"
            )

        if (self.min_rotation_misclassified is None) != (self.max_rotation_misclassified is None):
            raise ValueError(
                "min_rotation_misclassified and max_rotation_misclassified "
                "must be provided together"
            )

        if (self.min_shift_outliers is None) != (self.max_shift_outliers is None):
            raise ValueError(
                "min_shift_outliers and max_shift_outliers "
                "must be provided together"
            )

    @property
    def total_copies(self):
        return (
            self.n_copies
            + self.n_misclassified
            + self.n_shifted
            + self.n_rotated
            + self.n_noise
        )


@dataclass
class EstimatorConfig:
    name: str
    estimator_type: str
    params: dict


@dataclass
class PreprocessingConfig:
    mask_radius: float | None = None
    standardize: StandardizeOption = "none"


@dataclass
class SimulationConfig:
    snrs: list[float]
    n_runs: int
    initialization_strategy: InitializationStrategy
    dataset: DatasetConfig
    io: IOConfig
    preprocessing: PreprocessingConfig
    estimators: list[EstimatorConfig]

    def __post_init__(self):
        if self.dataset.n_misclassified > 0 and self.io.misclassified_path is None:
            raise ValueError(
                "Requested misclassified images but no misclassified path was provided"
            )
