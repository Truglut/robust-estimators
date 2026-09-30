from pathlib import Path
from dataclasses import dataclass


@dataclass
class IOConfig:
    reference_image_path: Path
    misclassified_path: Path | None = None
    base_results_path: Path | None = None 

@dataclass
class GenerationConfig:
    n_copies: int
    n_rotated: int
    n_shifted: int
    n_misclassified: int
    n_noise: int
    snrs: list[float]

@dataclass
class ExperimentConfig:
    n_runs: int
    # TODO
