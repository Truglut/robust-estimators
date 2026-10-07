from typing import Literal, Any, get_args

from pathlib import Path
from dataclasses import dataclass

StandardizeOption = Literal["before", "after", "both", "none"]
InitializationStrategy = Literal["mean", "median", "mean-good", "ground-truth"]
DataInitializationStrategy = Literal["mean", "median"]


@dataclass
class IOConfig:
    reference_path: Path
    misclassified_path: Path | None = None
    base_results_path: Path | None = None
    save_weights: bool = True
    save_estimates: bool = True
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
        counts = (
            self.n_copies,
            self.n_rotated,
            self.n_shifted,
            self.n_misclassified,
            self.n_noise,
        )
        if any((not isinstance(n, int)) or n < 0 for n in counts):
            raise ValueError("Image counts must be non-negative integers.")
        if self.total_copies == 0:
            raise ValueError("The total image count must be greater than zero.")

        for name, count, lower, upper in (
            (
                "rotation_outliers",
                self.n_rotated,
                self.min_rotation_outliers,
                self.max_rotation_outliers,
            ),
            (
                "rotation_misclassified",
                0,
                self.min_rotation_misclassified,
                self.max_rotation_misclassified,
            ),
            (
                "shift_outliers",
                self.n_shifted,
                self.min_shift_outliers,
                self.max_shift_outliers,
            ),
        ):
            if (lower is None) != (upper is None) or (count > 0 and lower is None):
                raise ValueError(f"Provide both min_{name} and max_{name}")
            if lower is not None and upper is not None and upper < lower:
                raise ValueError(f"Bounds for {name} must have min <= max")

        if self.min_shift_outliers is not None and self.min_shift_outliers < 0:
            raise ValueError("Shift magnitudes must be nonnegative")

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

    def __post_init__(self):
        # Names are used directly as directory components, avoiding lossy sanitizing.
        if (
            not self.name
            or self.name in (".", "..")
            or any(c in self.name for c in '/\\<>:"|?*')
            or any(ord(c) < 32 for c in self.name)
            or self.name.endswith((" ", "."))
        ):
            raise ValueError("Estimator names must be safe, nonempty directory names")

    def to_method_cfg(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.estimator_type, "params": self.params}


@dataclass
class PreprocessingConfig:
    mask_radius: float | None = None
    standardize: StandardizeOption = "none"


@dataclass
class EstimationIOConfig:
    dataset_path: Path
    base_results_path: Path
    save_weights: bool = True
    save_estimates: bool = True


@dataclass
class EstimationConfig:
    """One fit of each estimator on an existing stack of aligned 2D images."""

    io: EstimationIOConfig
    estimators: list[EstimatorConfig]
    preprocessing: PreprocessingConfig
    initialization_strategy: DataInitializationStrategy = "mean"
    seed: int = 42

    def __post_init__(self):
        names = [method.name for method in self.estimators]
        if not names or len(names) != len(set(names)) or "mean" in names:
            raise ValueError("Provide unique estimator names; 'mean' is reserved")
        if self.initialization_strategy not in get_args(DataInitializationStrategy):
            raise ValueError(
                "Stored datasets support only mean or median initialization"
            )
        if self.preprocessing.standardize not in ("none", "after"):
            raise ValueError(
                "Stored datasets support only 'none' or 'after' standardization"
            )
        radius = self.preprocessing.mask_radius
        if radius is not None and (not 0 < radius < float("inf")):
            raise ValueError("mask_radius must be finite and positive")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if self.initialization_strategy == "median" and any(
            method.estimator_type == "noise_corrected_cosine"
            for method in self.estimators
        ):
            raise ValueError(
                "Noise-corrected cosine requires mean initialization for stored datasets"
            )


@dataclass
class SimulationConfig:
    snrs: list[float]
    n_runs: int
    initialization_strategy: InitializationStrategy
    dataset: DatasetConfig
    io: IOConfig
    preprocessing: PreprocessingConfig
    estimators: list[EstimatorConfig]
    seed: int = 42

    def __post_init__(self):
        if any(snr <= 0 for snr in self.snrs):
            raise ValueError("SNR values must be positive")
        if len(set(self.snrs)) != len(self.snrs):
            raise ValueError("SNRs must be unique")

        if not isinstance(self.n_runs, int) or self.n_runs < 1:
            raise ValueError("n_runs must be a positive integer")

        names = [method.name for method in self.estimators]
        if not names or len(names) != len(set(names)) or "mean" in names:
            raise ValueError("Provide unique estimator names; 'mean' is reserved")

        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")

        if self.initialization_strategy not in get_args(InitializationStrategy):
            raise ValueError(
                f"Unsupported initialization: {self.initialization_strategy!r}"
            )

        if self.initialization_strategy == "mean-good" and self.dataset.n_copies == 0:
            raise ValueError("mean-good initialization requires good images")

        if self.dataset.n_misclassified > 0 and self.io.misclassified_path is None:
            raise ValueError("Requested misclassified images but no path was provided")
