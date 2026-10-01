from typing import Literal

import numpy as np

from simple_mestimators.domain import ImageBatch, EstimatorResult
from .base import Estimator
from .irls import IRLSSolver

WeightApproach = Literal["per-image", "per-coefficient"]


class JointIRLSFourier(Estimator):
    def __init__(
        self,
        irls_solver: IRLSSolver,
        eps: float = 1.0e-8,
        weight_approach: WeightApproach = "per-coefficient",
        mask: np.ndarray | None = None,
    ) -> None:
        self.solver = irls_solver
        self.eps = eps
        self.weight_approach = weight_approach
        self.mask = mask

    @property
    def max_iter(self):
        return self.solver.max_iter

    def solve(
        self,
        fourier_images: np.ndarray,
        image_variance: np.ndarray,
        image_std: np.ndarray | None = None,
        ctf: np.ndarray | None = None,
        reference: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_variance: np.ndarray | float | None = None,
        max_iter_override: int | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """All inputs are assumed to be Fourier-space appropriate"""
        mask = self.mask if mask is None else mask

        full_variance = image_variance
        full_ctf = ctf
        full_prior_mean = prior_mean
        full_prior_variance = prior_variance
        images_masked = fourier_images
        if mask is not None:
            images_masked = fourier_images[:, mask]

            if image_variance.size > 1:
                image_variance = image_variance[mask]
            image_std = np.sqrt(image_variance)

            if reference is not None:
                reference = reference[mask]

            if prior_mean is not None:
                prior_mean = prior_mean[mask]
            if isinstance(prior_variance, np.ndarray) and prior_variance.size > 1:
                prior_variance = prior_variance[mask]

            if ctf is not None:
                ctf = ctf[:, mask]

        estimate, weights = self.solver.solve(
            images=images_masked,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )

        if mask is not None:
            # Aggregate and reshape weights to make averaging possible
            # NOTE: there are other possibilities that could be considered here for
            # local weights, although masking will mostly be used with global weights
            weight_spatial_dims = tuple(range(1, weights.ndim))
            target_weight_shape = (fourier_images.shape[0],) + (1,) * (
                fourier_images.ndim - 1
            )
            weights_update = weights.mean(axis=weight_spatial_dims).reshape(
                target_weight_shape
            )

            # Re-calculate estimate with unmasked images
            estimate = IRLSSolver.calculate_update(
                images=fourier_images,
                weights=weights_update,
                ctf=full_ctf,
                prior_mean=full_prior_mean,
                prior_variance=full_prior_variance,
                image_variance=full_variance,
                eps=self.eps,
            )

        return estimate, weights

    def fit(
        self,
        images: ImageBatch,
        *,
        reference: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_variance: np.ndarray | float | None = None,
        max_iter_override: int | None = None,
        mask: np.ndarray | None = None,
    ) -> EstimatorResult:
        fourier_images = images.ensure_fourier()
        image_variance = images.fourier_modulus_variance()
        mask = self.mask if mask is None else mask
        if reference is not None and not np.iscomplexobj(reference):
            reference = np.fft.rfft2(reference, norm=images.norm)
        if prior_mean is not None and not np.iscomplexobj(prior_mean):
            prior_mean = np.fft.rfft2(prior_mean, norm=images.norm)

        fourier_estimate, weights = self.solve(
            fourier_images=fourier_images,
            image_variance=image_variance,
            ctf=images.ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
            mask=mask,
        )

        if mask is not None:
            if weights.shape[1:] == (1,):
                # Masking flattens the Fourier grid for the solve. Restore the
                # broadcast dimensions for per-image and zero-iteration weights.
                weights = weights.reshape((fourier_images.shape[0], 1, 1))
            else:
                # Keep selected coefficient weights in their original positions.
                # NaN denotes coefficients that were not used to obtain weights.
                full_weights = np.full(
                    fourier_images.shape, np.nan, dtype=weights.dtype
                )
                full_weights[:, mask] = weights
                weights = full_weights

        return EstimatorResult(
            real_estimate=np.fft.irfft2(
                fourier_estimate, s=images.real_shape, norm=images.norm
            ),
            weights={"real": None, "fourier": weights},
        )
