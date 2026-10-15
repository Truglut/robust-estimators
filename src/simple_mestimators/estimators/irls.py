from __future__ import annotations

import numpy as np

from .base import Estimator
from .weights import WeightFunction
from simple_mestimators.domain import ImageBatch, EstimatorResult


class IRLSSolver(Estimator):
    """Iteratively reweighted least-squares solver."""

    def __init__(
        self,
        weight_function: WeightFunction,
        max_iter: int,
        tol: float = 1.0e-5,
        damping_coef: float = 0.0,
        min_weight: float | None = None,
        max_weight: float | None = None,
        eps: float = 1e-8,
    ):
        # Estimator configuration
        self.weight_function = weight_function
        self.max_iter = max_iter
        self.tol = tol  # tolerance for stopping iterations
        self.eps = eps  # small numerical value to avoid division by zero

        # Regularization strategies
        self.damping_coef = damping_coef
        self.min_weight = min_weight
        self.max_weight = max_weight

    def _get_safe_variance(
        self,
        images: np.ndarray,
        image_variance: np.ndarray | None = None,
        image_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates per-pixel image variance and standard deviation, clamping them
        to protect against division by zero
        """
        if image_variance is None:
            image_variance = images.var(axis=0)
        if image_std is None:
            image_std = np.sqrt(image_variance)

        image_variance = np.clip(image_variance, min=self.eps)
        image_std = np.clip(image_std, min=self.eps)

        return image_variance, image_std

    @classmethod
    def calculate_update(
        cls,
        images: np.ndarray,
        weights: np.ndarray,
        *,
        ctf: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_variance: np.ndarray | float | None = None,
        image_variance: np.ndarray | None = None,
        eps: float = 1.0e-8,
    ) -> np.ndarray:
        # New estimate calculation:
        # x_new = (s_1 / image_variance + prior_mean / prior_variance) /
        #         (s_2 / image_variance + 1 / prior_variance)
        if ctf is None:
            s_1 = np.sum(weights * images, axis=0)
            s_2 = np.sum(weights, axis=0)
        else:
            s_1 = np.sum(weights * ctf * images, axis=0)
            s_2 = np.sum(weights * np.square(ctf), axis=0)

        if prior_mean is None or prior_variance is None:
            return s_1 / (s_2.clip(min=eps))

        # Assume prior_mean and prior_variance are safe to divide by
        reciprocal_prior_variance = 1.0 / prior_variance
        numerator = s_1 / image_variance + prior_mean * reciprocal_prior_variance
        denominator = s_2 / image_variance + reciprocal_prior_variance

        return numerator / denominator

    def _fit_one_iteration(
        self,
        images: np.ndarray,
        image_variance: np.ndarray,
        image_std: np.ndarray,
        reference: np.ndarray,
        ctf: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_variance: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs a single iteration of the Reweighted Least Squares update"""
        weights = self.weight_function(images, reference, image_std)

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = np.clip(weights, min=self.min_weight, max=self.max_weight)

        if weights.ndim == 1:
            # Reshape weights to shape (batch, 1, ..., 1) to broadcast over image batch
            weights = weights.reshape(weights.shape[0], *((1,) * (images.ndim - 1)))

        update = IRLSSolver.calculate_update(
            images=images,
            weights=weights,
            ctf=ctf,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            image_variance=image_variance,
            eps=self.eps,
        )

        # Use update damping for calculating the new estimate
        eta = self.damping_coef
        new_estimate = eta * reference + (1.0 - eta) * update

        return new_estimate, weights

    def solve(
        self,
        images: np.ndarray,
        *,
        image_variance: np.ndarray | None = None,
        image_std: np.ndarray | None = None,
        ctf: np.ndarray | None = None,
        reference: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_variance: np.ndarray | float | None = None,
        max_iter_override: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes the low-level Iteratively Reweighted Least Squares (IRLS) optimization.

        This method operates domain-agnostically on raw arrays (real or complex).

        Parameters
        ----------
        images : np.ndarray
            Array of shape ``(n_images, *image_shape)`` containing the images to
            be averaged using the robust IRLS procedure, batched along the first
            dimension. These can be real-space or Fourier-space images, provided
            ``self.weight_function`` handles complex values.
        image_variance : np.ndarray, optional
            Variance of the input images. Can be a array matching ``image_shape``
            (per-pixel variance) or a scalar (global image variance).
            Defaults to variance of ``images`` along dimension 0.
        image_std : np.ndarray, optional
            Standard deviation of the input images. Pre-calculated square root
            of ``image_variance`` to avoid redundant computation.
        ctf : np.ndarray, optional
            Contrast Transfer Function of the input images matching or broadcastable
            to ``images.shape``. If None, images are assumed to be CTF-corrected.
        reference : np.ndarray, optional
            Initial reference for robust averaging. Should match ``image_shape``.
            Defaults to the average of all images, i.e. ``images.mean(dim=0)``.
        prior_mean : np.ndarray, optional
            Prior mean for the estimator. This will bias the produced estimation
            towards the prior mean, serving as a type of regularization (e.g. the
            prior mean might be a array of zeros, keeping the values of the
            reconstructed averages closer to zero).
            Its shape should match the shape of one image.
            Cannot be provided without also providing a value for ``prior_variance``.
            If not provided, no regularization will be applied.
        prior_variance : np.ndarray, optional
            Prior variance for the estimator. This effectively controls the strength
            of the regularization imposed by the prior mean. A higher value of the
            prior variance means a *weaker* regularization.
            Cannot be provided without also providing a value for ``prior_mean``.
            If not provided, no regularization will be applied.
        max_iter_override : int, optional
            Maximum number of IRLS iterations to be performed by the estimator.
            Overrides the estimator's default ``max_iter`` for this run.

        Returns
        -------
        reference : np.ndarray
            The estimated robust average in the same domain and shape as an input image.
        weights : np.ndarray
            Element-wise particle weights from the final iteration, matching
            ``images.shape``. Returns a array of ones if ``max_iter == 0``.

        Raises
        ------
        ValueError
            If ``max_iter`` is negative.
        """
        max_iter = self.max_iter if max_iter_override is None else max_iter_override
        self.converged = False
        self.n_its = 0
        self.diagnostics = {
            "n_iter": 0, "converged": False, "stop_reason": "max_iter",
            "relative_change": None, "max_iter_effective": max_iter,
        }

        if max_iter < 0:
            raise ValueError(f"`max_iter` must be non-negative, got {max_iter}.")

        if (prior_mean is None) != (prior_variance is None):
            raise ValueError(
                "prior_mean and prior_variance have to be provided together"
            )

        # Calculate default initial reference if not provided
        if reference is None:
            reference = images.mean(axis=0)

        weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)
        weights = np.ones(shape=weight_shape, dtype=images.dtype)

        # Handle 0-iteration shortcut (unweighted baseline)
        if max_iter == 0:
            self.converged = True
            self.diagnostics.update(converged=True, stop_reason="zero_iterations")
            return reference, weights

        # Get safe variances and std
        image_variance, image_std = self._get_safe_variance(
            images, image_variance, image_std
        )
        if prior_variance is not None:
            if isinstance(prior_variance, np.ndarray):
                prior_variance = np.clip(prior_variance, min=self.eps)
            else:
                prior_variance = max(prior_variance, self.eps)

        # Main iterations loop
        for i in range(max_iter):
            next_reference, weights = self._fit_one_iteration(
                images,
                image_variance=image_variance,
                image_std=image_std,
                reference=reference,
                ctf=ctf,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )

            # Convergence check with norm-zero protection
            ref_norm = np.linalg.norm(reference)
            diff_norm = np.linalg.norm(next_reference - reference)
            relative_difference = diff_norm / (ref_norm + self.eps)
            self.n_its = i + 1
            self.diagnostics.update(n_iter=self.n_its, relative_change=float(relative_difference))

            # Update reference before possibly breaking out of the loop
            reference = next_reference

            if relative_difference < self.tol:
                self.converged = True
                self.diagnostics.update(converged=True, stop_reason="tolerance")
                break

        return reference, weights

    def fit(
        self,
        images: ImageBatch,
        *,
        reference: np.ndarray | None = None,
        prior_mean: np.ndarray | None = None,
        prior_variance: np.ndarray | float | None = None,
        max_iter_override: int | None = None,
    ):
        """
        Fit IRLS to an ImageBatch in the selected space.

        Parameters
        ----------
        batch:
            Canonical image batch.
        reference:
            Optional initial reference. For Fourier spaces, this may be either a
            complex Fourier tensor or an already-selected real/imaginary component.
        prior_mean:
            Optional prior mean. Same selection rules as ``reference``.
        prior_variance:
            Optional prior variance.
        max_iter_override:
            Optional temporary iteration limit.
        """
        imgs, image_variance, ctf = images.select_space_data(space="real")
        image_std = np.sqrt(image_variance)

        estimate, weights = self.solve(
            imgs,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )

        return EstimatorResult(
            real_estimate=estimate,
            weights={
                "real": weights,
                "fourier": None,
            },
            diagnostics=self.diagnostics.copy(),
        )
