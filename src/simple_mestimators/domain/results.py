from dataclasses import dataclass

import numpy as np

from .data import ImageSpace


@dataclass
class EstimatorResult:
    """Standard output returned by estimators."""

    real_estimate: np.ndarray
    weights: dict[ImageSpace, np.ndarray | None]
