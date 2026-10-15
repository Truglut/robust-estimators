from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import ImageSpace


@dataclass
class GMMDiagnostics:
    distances: np.ndarray
    weights: np.ndarray
    raw_responsibilities: np.ndarray
    means: tuple[float, float]
    variances: tuple[float, float]
    component_weights: tuple[float, float]


@dataclass
class EstimatorResult:
    """Standard output returned by estimators."""

    real_estimate: np.ndarray
    weights: dict[ImageSpace, np.ndarray | None]
    gmm_diagnostics: GMMDiagnostics | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
