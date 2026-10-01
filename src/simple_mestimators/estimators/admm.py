from __future__ import annotations

from typing import NamedTuple
from dataclasses import dataclass

import numpy as np

from .irls import IRLSSolver
from .fourier_irls import JointIRLSFourier
from .base import Estimator

from simple_mestimators.domain import EstimatorResult, ImageBatch


class ImageData(NamedTuple):
    images: np.ndarray
    variance: np.ndarray
    std: np.ndarray


class ADMMData(NamedTuple):
    real: ImageData
    fourier: ImageData
    ctf: np.ndarray | None = None


@dataclass
class ADMMState:
    reference_real: np.ndarray
    reference_fourier: np.ndarray
    dual_vars: np.ndarray
    mu: float


class ADMMIterationResult(NamedTuple):
    next_real: np.ndarray
    next_fourier: np.ndarray
    weights_real: np.ndarray
    weights_fourier: np.ndarray
    next_real_transformed: np.ndarray


class ADMMEstimator(Estimator):
    """ADMM estimator coupling real-space and Fourier-space IRLS updates."""

    def __init__(
        self,
        irls_real: IRLSSolver,
        irls_fourier: JointIRLSFourier,
        max_iter: int,
        initial_mu: float,
        fourier_multiplier: float,
        atol: float = 1.0e-4,
        rtol: float = 1.0e-4,
        eps: float = 1.0e-8,
    ) -> None:
        self.irls_real = irls_real
        self.irls_fourier = irls_fourier
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.initial_mu = initial_mu
        self.fourier_multiplier = fourier_multiplier
        self.eps = eps

        self.converged = False
        self.n_its = None

    def _real_update(
        self,
        *,
        data: ADMMData,
        state: ADMMState,
        real_irls_max_iter: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the real-space subproblem with IRLS"""
        # Pass signal shape from real reference to preserve odd dimensions
        real_shape = state.reference_real.shape[-2:]
        prior_mean = np.fft.irfft2(
            state.reference_fourier + state.dual_vars / state.mu,
            s=real_shape,
            norm="ortho",
        )
        return self.irls_real.solve(
            images=data.real.images,
            image_variance=data.real.variance,
            image_std=data.real.std,
            reference=state.reference_real,
            prior_mean=prior_mean,
            prior_variance=1.0 / state.mu,
            max_iter_override=real_irls_max_iter,
        )

    def _fourier_update(
        self,
        *,
        data: ADMMData,
        state: ADMMState,
        next_real_transformed: np.ndarray,
        fourier_irls_max_iter: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the Fourier-space subproblem with the joint IRLS approach"""
        prior_mean = next_real_transformed - state.dual_vars / state.mu
        prior_variance = self.fourier_multiplier * (1.0 / state.mu)

        return self.irls_fourier.solve(
            fourier_images=data.fourier.images,
            image_variance=data.fourier.variance,
            image_std=data.fourier.std,
            ctf=data.ctf,
            reference=state.reference_fourier,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=fourier_irls_max_iter,
        )

    def _fit_one_iteration(
        self,
        *,
        data: ADMMData,
        state: ADMMState,
        real_irls_max_iter: int | None = None,
        fourier_irls_max_iter: int | None = None,
    ) -> ADMMIterationResult:
        # Real update
        next_real, real_update_weights = self._real_update(
            data=data,
            state=state,
            real_irls_max_iter=real_irls_max_iter,
        )

        # Fourier space update
        next_real_transformed = np.fft.rfft2(next_real, norm="ortho")
        next_fourier, fourier_update_weights = self._fourier_update(
            data=data,
            state=state,
            next_real_transformed=next_real_transformed,
            fourier_irls_max_iter=fourier_irls_max_iter,
        )

        return ADMMIterationResult(
            next_real=next_real,
            next_fourier=next_fourier,
            weights_real=real_update_weights,
            weights_fourier=fourier_update_weights,
            next_real_transformed=next_real_transformed,
        )

    def fit(
        self,
        images: ImageBatch,
        *,
        reference: np.ndarray | None = None,
        initial_reference_real: np.ndarray | None = None,
        initial_reference_fourier: np.ndarray | None = None,
    ) -> EstimatorResult:
        if images.norm != "ortho":
            raise ValueError("The ADMM method requires a unitary Fourier transform")

        data = self._prepare_data(images)

        state = self._initialize_state(
            data=data,
            reference=reference,
            initial_reference_real=initial_reference_real,
            initial_reference_fourier=initial_reference_fourier,
        )

        results = None
        self.converged = False
        for i in range(self.max_iter):
            results = self._fit_one_iteration(
                data=data,
                state=state,
                real_irls_max_iter=min(5 + i, self.irls_real.max_iter),
                fourier_irls_max_iter=min(5 + i, self.irls_fourier.max_iter),
            )

            # Update dual variables
            primal_residual = results.next_fourier - results.next_real_transformed
            state.dual_vars += state.mu * primal_residual

            # Convergence check only every five iterations to save time
            if i % 5 != 4:
                state.reference_real = results.next_real
                state.reference_fourier = results.next_fourier
                continue

            # Calculate residual norms for convergence check before updating references
            primal_norm, dual_norm, eps_primal, eps_dual = self._residuals(
                state=state,
                results=results,
                primal_residual=primal_residual,
            )

            state.reference_real = results.next_real
            state.reference_fourier = results.next_fourier

            if primal_norm < eps_primal and dual_norm < eps_dual:
                self.converged = True
                break

            state.mu = self._mu_update(state.mu, primal_norm, dual_norm)

        real_shape = state.reference_real.shape[-2:]
        fourier_ref_to_real = np.fft.irfft2(
            state.reference_fourier, s=real_shape, norm="ortho"
        )
        estimate = 0.5 * (state.reference_real + fourier_ref_to_real)

        if results is None:
            weight_shape = (images.n_images,) + (1,) * (images.real.ndim - 1)
            weights_real = np.ones(weight_shape, dtype=images.real.dtype)
            weights_fourier = weights_real
        else:
            weights_real = results.weights_real
            weights_fourier = results.weights_fourier

        return EstimatorResult(
            real_estimate=estimate,
            weights={"real": weights_real, "fourier": weights_fourier},
        )

    def _mu_update(self, mu: float, primal_norm: float, dual_norm: float) -> float:
        if primal_norm > 10 * dual_norm:
            return 2.0 * mu
        if dual_norm > 10 * primal_norm:
            return 0.5 * mu
        return mu

    def _residuals(
        self,
        *,
        state: ADMMState,
        results: ADMMIterationResult,
        primal_residual: np.ndarray,
    ):
        next_real = results.next_real
        next_real_transformed = results.next_real_transformed
        next_fourier = results.next_fourier

        p = next_fourier.size  # number of restrictions
        n = next_real.size  # number of primal variables
        primal_norm = np.linalg.norm(primal_residual).item()
        dual_norm = (
            state.mu * np.linalg.norm(next_fourier - state.reference_fourier).item()
        )

        eps_primal = np.sqrt(p) * self.atol + self.rtol * max(
            np.linalg.norm(next_real_transformed).item(),
            np.linalg.norm(next_fourier).item(),
        )
        eps_dual = (
            np.sqrt(n) * self.atol + self.rtol * np.linalg.norm(state.dual_vars).item()
        )

        return primal_norm, dual_norm, eps_primal, eps_dual

    def _prepare_data(self, images: ImageBatch) -> ADMMData:
        """Prepare real- and Fourier-space image data for ADMM estimation."""
        # We have already validated images.norm == "ortho"
        fourier_images = images.ensure_fourier()
        image_variance_fourier = images.fourier_modulus_variance().clip(min=1.0e-8)
        image_std_fourier = np.sqrt(image_variance_fourier)

        image_variance_real = images.real_variance().clip(min=1.0e-8)
        image_std_real = np.sqrt(image_variance_real)

        return ADMMData(
            real=ImageData(
                images=images.real,
                variance=image_variance_real,
                std=image_std_real,
            ),
            fourier=ImageData(
                images=fourier_images,
                variance=image_variance_fourier,
                std=image_std_fourier,
            ),
            ctf=images.ctf,
        )

    def _initialize_state(
        self,
        data: ADMMData,
        reference: np.ndarray | None = None,
        initial_reference_real: np.ndarray | None = None,
        initial_reference_fourier: np.ndarray | None = None,
    ) -> ADMMState:
        """Initialize references, dual variables, and penalty parameter."""
        given_general_reference = reference is not None
        given_specific_references = (
            initial_reference_real is not None or initial_reference_fourier is not None
        )
        if given_general_reference and given_specific_references:
            raise ValueError(
                "Cannot provide both a general ``reference`` and space-specific references for ADMM"
            )

        if reference is not None:
            initial_reference_real = reference
            initial_reference_fourier = np.fft.rfft2(reference, norm="ortho")

        if initial_reference_real is None:
            initial_reference_real = data.real.images.mean(axis=0)

        if initial_reference_fourier is None:
            initial_reference_fourier = data.fourier.images.mean(axis=0)

        return ADMMState(
            reference_real=initial_reference_real,
            reference_fourier=initial_reference_fourier,
            dual_vars=np.zeros_like(initial_reference_fourier),
            mu=self.initial_mu,
        )
