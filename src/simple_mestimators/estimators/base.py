import numpy as np

from simple_mestimators.domain import ImageBatch, EstimatorResult


class Estimator:
    def fit(
        self, images: ImageBatch, *, reference: np.ndarray | None = None
    ) -> EstimatorResult:
        raise NotImplementedError
