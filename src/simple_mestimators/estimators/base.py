from simple_mestimators.domain import ImageBatch, EstimatorResult


class Estimator:
    def fit(self, images: ImageBatch) -> EstimatorResult:
        raise NotImplementedError
