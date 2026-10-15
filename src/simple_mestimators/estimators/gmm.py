"""
GMM-based robust estimator.

This module implements a recursive robust averaging method that fits a two-component
Gaussian mixture model to distances/dissimilarities to the current reference and uses
the responsibilities of the closest component as image weights.
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from .base import Estimator
from .weights import weighted_average
from .distances import DistanceFunction

from simple_mestimators.domain import (
    ImageSpace,
    ImageBatch,
    EstimatorResult,
    GMMDiagnostics,
)

MIN_ELEMENTS_FOR_GMM = 50

GMM_MIN_SEPARATION = 0.20
GMM_MIN_GOOD_COMPONENT_WEIGHT = 0.50

GMM_DEFAULT_BAD_WEIGHT = 0.05
GMM_DEFAULT_BAD_QUANTILE = 0.95
GMM_DEFAULT_GOOD_QUANTILE = 0.5


class RecursiveGMMEstimator(Estimator):
    """
    Recursive robust averaging with peak-corrected GMM responsibilities.

    Unusable iterations stop the recursion and return the ordinary image mean
    with unit weights. For rejected models the array diagnostics describe the
    attempted fit, while scalar diagnostics flag the fallback explicitly.

    The good component has the lower mean distance. By default, reject models
    whose mean separation divided by ``sqrt(var_1 + var_2)`` is below a certain
    threshold (``GMM_MIN_SEPARATION``), or whose good component has mixture
    weight below ``GMM_MIN_GOOD_COMPONENT_WEIGHT``.
    """

    def __init__(
        self,
        distance_function: DistanceFunction,
        max_iter: int = 1,
        tol: float = 1.0e-4,
        standardize_distances: bool = True,
        random_state: int | None = None,
        gmm_max_iter: int = 20,
        gmm_tol: float = 1.0e-4,
        initialize_params: bool = True,
        bad_quantile: float = GMM_DEFAULT_BAD_QUANTILE,
        bad_weight: float = GMM_DEFAULT_BAD_WEIGHT,
        good_quantile: float = GMM_DEFAULT_GOOD_QUANTILE,
        check_degenerate_model: bool = True,
        min_component_separation: float = GMM_MIN_SEPARATION,
        min_good_component_weight: float = GMM_MIN_GOOD_COMPONENT_WEIGHT,
        reg_covar: float = 1.0e-6,
        n_init: int = 1,
        init_params: str = "kmeans",
    ):
        self.model = GaussianMixture(
            n_components=2,
            max_iter=gmm_max_iter,
            tol=gmm_tol,
            random_state=random_state,
            warm_start=True,
            reg_covar=reg_covar,
            n_init=n_init,
            init_params=init_params,  # type: ignore
        )

        self.distance_function = distance_function
        self.max_iter = max_iter
        self.tol = tol
        self.standardize_distances = standardize_distances

        self.initialize_params = initialize_params
        self.bad_quantile = bad_quantile
        self.bad_weight = bad_weight
        self.good_quantile = good_quantile

        self.check_degenerate_model = check_degenerate_model
        self.min_component_separation = min_component_separation
        self.min_good_component_weight = min_good_component_weight

        self.gmm_max_iter = gmm_max_iter
        self.gmm_tol = gmm_tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.init_params = init_params

        self.n_its = None
        self.converged = False
        self.diagnostics = {}

    def _new_model(self) -> GaussianMixture:
        """Create a fresh GMM, resetting any state from previous ``fit`` calls."""
        return GaussianMixture(
            n_components=2,
            max_iter=self.gmm_max_iter,
            tol=self.gmm_tol,
            random_state=self.model.random_state,  # type: ignore
            warm_start=True,
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            init_params=self.init_params,  # type: ignore
        )

    def _initialize_model_params(self, distances: np.ndarray) -> None:
        """
        Initialize component weights and means from the observed distances.
        """
        initial_weights = [1.0 - self.bad_weight, self.bad_weight]
        component_weights = np.array(initial_weights, dtype=distances.dtype)

        mean_quantiles = [self.good_quantile, self.bad_quantile]
        component_means = np.quantile(
            distances.reshape(-1),
            np.array(mean_quantiles, dtype=distances.dtype),
        )

        self.model.means_init = component_means.reshape(2, 1)  # type: ignore
        self.model.weights_init = component_weights  # type: ignore

    def _standardize(self, distances: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Optionally standardize distances to zero mean and unit variance."""
        if not self.standardize_distances:
            return distances, 0.0, 1.0

        std = distances.std().clip(min=1.0e-8)
        mean = distances.mean()

        return (distances - mean) / std, float(mean), float(std)

    def _responsibility_weights(
        self, model: GaussianMixture, distances_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior probabilities of the lower-distance GMM component."""
        good_component = self.get_good_component_idx()
        responsibilities = model.predict_proba(distances_np)[:, good_component]

        order = np.argsort(distances_np.reshape(-1), kind="stable")
        sorted_raw = responsibilities[order]
        peak_idx = int(sorted_raw.argmax())
        sorted_peak = sorted_raw.copy()
        sorted_peak[:peak_idx] = sorted_raw[peak_idx]
        sorted_peak[peak_idx:] = np.minimum.accumulate(sorted_raw[peak_idx:])
        peak_weights = np.empty_like(responsibilities)
        peak_weights[order] = sorted_peak

        return responsibilities, peak_weights

    def get_model_means(self) -> tuple[float, float]:
        return tuple(self.model.means_[:, 0])  # type: ignore

    def get_model_variances(self) -> tuple[float, float]:
        return tuple(self.model.covariances_[:, 0, 0])  # type: ignore

    def get_model_weights(self) -> tuple[float, float]:
        return tuple(self.model.weights_)  # type: ignore

    def get_good_component_idx(self) -> int:
        means = self.get_model_means()
        return int(np.argmin(means))

    def _degeneracy_checks(self) -> tuple[bool, bool]:
        """Check normalized mean separation and the good component's mass."""
        means = self.get_model_means()
        variances = self.get_model_variances()
        separation = abs(means[1] - means[0]) / np.sqrt(variances[0] + variances[1])
        good_component = np.argmin(means)
        good_weight = self.get_model_weights()[good_component]

        return (
            bool(separation < self.min_component_separation),
            bool(good_weight < self.min_good_component_weight),  # type: ignore
        )

    def _fit_one_iteration(
        self,
        images: np.ndarray,
        reference: np.ndarray,
        initialize_params: bool = False,
    ) -> tuple[bool, np.ndarray, np.ndarray, GMMDiagnostics]:
        """
        Perform one iteration of the recursive GMM estimation procedure.

        Returns
        -------
        degenerate : bool
        estimate : np.ndarray
        weights : np.ndarray
        diagnostics : GMMDiagnostics
        """
        distances = self.distance_function(images, reference)
        if np.iscomplexobj(distances) or distances.size != images.shape[0]:
            raise ValueError(
                "The distance function must return one real distance per image"
            )
        weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)

        std_distances, _, _ = self._standardize(distances.reshape(-1))
        if not np.isfinite(std_distances).all():
            raise ValueError("Some distances were not finite")

        # Prepare distances for sklearn's GaussianMixture
        if std_distances.ndim == 1:
            std_distances = std_distances[:, None]

        if initialize_params:
            self._initialize_model_params(std_distances)

        # Fit GMM to the distance distribution. With warm_start=True, later
        # recursive iterations start from the previous iteration's fitted model.
        self.model.fit(std_distances)

        # Get weights and update reference
        raw_responsibilities, weights = self._responsibility_weights(
            self.model, std_distances
        )
        weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)
        diagnostics = GMMDiagnostics(
            distances=distances,
            raw_responsibilities=raw_responsibilities,
            weights=weights,
            means=self.get_model_means(),
            variances=self.get_model_variances(),
            component_weights=self.get_model_weights(),
        )

        degenerate = False
        separation_failed, mass_failed = self._degeneracy_checks()
        self.diagnostics.update(
            gmm_converged=bool(self.model.converged_),
            gmm_n_iter=int(self.model.n_iter_),
            component_means=list(self.get_model_means()),
            component_variances=list(self.get_model_variances()),
            component_weights=list(self.get_model_weights()),
            component_separation=float(
                abs(self.model.means_[0, 0] - self.model.means_[1, 0])  # type: ignore
                / np.sqrt(sum(self.get_model_variances()))
            ),
            good_component_weight=float(
                self.get_model_weights()[self.get_good_component_idx()]
            ),
        )
        if self.check_degenerate_model and (separation_failed or mass_failed):
            reasons = []
            if separation_failed:
                reasons.append("insufficient_separation")
            if mass_failed:
                reasons.append("insufficient_good_component_weight")
            self.diagnostics.update(
                fallback=True, fallback_reasons=reasons, stop_reason="fallback"
            )
            degenerate = True
            estimate = images.mean(axis=0)
            weights = np.ones(shape=weight_shape, dtype=images.dtype)
            return degenerate, estimate, weights, diagnostics

        weights = weights.reshape(weight_shape)
        estimate = weighted_average(images, weights, eps=0)

        return degenerate, estimate, weights, diagnostics

    def solve(
        self, images: np.ndarray, reference: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, GMMDiagnostics | None]:
        # Reset the GMM to avoid carrying over state from previous solve() calls
        self.model = self._new_model()

        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative")

        # Get initial reference
        if reference is None:
            reference = images.mean(axis=0)

        self.converged = False
        self.n_its = 0
        self.diagnostics = {
            "n_iter": 0,
            "converged": False,
            "stop_reason": "max_iter",
            "relative_change": None,
            "fallback": False,
            "fallback_reasons": [],
            "gmm_converged": None,
            "gmm_n_iter": None,
        }

        weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)
        weights = np.ones(shape=weight_shape, dtype=np.float64)

        diagnostics = None
        for i in range(self.max_iter):
            self.n_its = i + 1
            self.diagnostics["n_iter"] = self.n_its
            degenerate, estimate, weights, diagnostics = self._fit_one_iteration(
                images, reference, initialize_params=self.initialize_params and i == 0
            )

            if degenerate:
                reference = estimate
                break

            rel_change = np.linalg.norm(estimate - reference) / np.linalg.norm(
                reference
            ).clip(min=1.0e-8)
            reference = estimate
            self.diagnostics["relative_change"] = float(rel_change)

            if rel_change < self.tol:
                self.converged = True
                self.diagnostics.update(converged=True, stop_reason="tolerance")
                break

        return reference, weights, diagnostics

    def fit(
        self, images: ImageBatch, *, reference: np.ndarray | None = None
    ) -> EstimatorResult:
        """Fit the recursive estimator and return its result."""
        estimate, weights, diagnostics = self.solve(images.real, reference)

        return EstimatorResult(
            real_estimate=estimate,
            weights={"real": weights, "fourier": None},
            gmm_diagnostics=diagnostics,
            diagnostics=self.diagnostics.copy(),
        )


class FourierGMM(Estimator):
    # TODO: allow for masks

    def __init__(self, solver: RecursiveGMMEstimator):
        self.solver = solver

    def fit(
        self,
        images: ImageBatch,
        *,
        reference: np.ndarray | None = None,
    ) -> EstimatorResult:
        fourier_images = images.ensure_fourier()
        if reference is None:
            reference = fourier_images.mean(axis=0)
        elif not np.iscomplexobj(reference):
            reference = np.fft.rfft2(reference, norm=images.norm)

        fourier_estimate, weights, diagnostics = self.solver.solve(
            fourier_images, reference
        )
        estimate = np.fft.irfft2(
            fourier_estimate, s=images.real_shape, norm=images.norm
        )

        return EstimatorResult(
            real_estimate=estimate,
            weights={"real": None, "fourier": weights},
            gmm_diagnostics=diagnostics,
            diagnostics=self.solver.diagnostics.copy(),
        )
