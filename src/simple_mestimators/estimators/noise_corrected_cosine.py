"""Experimental noise-corrected cosine with cached image-side quantities.

Assumes additive white noise in the supplied image units. Reference covariance
propagation is exact for fixed linear weights, and a plug-in approximation for
weights selected from the noisy data. No labels or simulation SNR are used here.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.fft import rfft2, fftfreq, rfftfreq

from .base import Estimator
from .gmm import RecursiveGMMEstimator
from simple_mestimators.domain import ImageBatch, EstimatorResult


def estimate_noise_variance(images, support, batch_size=64):
    """Estimate per-image white-noise variance from checkerboard MAD.

    Only 2x2 blocks inside the binary support are used. Their noise variance is
    four times the input pixel variance. Fine-scale signal can bias estimates
    upward; estimates from image-wise standardized data are plug-in estimates.
    """
    valid = support[:-1, :-1] & support[1:, :-1] & support[:-1, 1:] & support[1:, 1:]
    if not valid.any():
        raise ValueError("Automatic noise estimation needs a valid 2x2 mask block")
    result = np.empty(len(images), dtype=np.float64)
    for start in range(0, len(images), batch_size):
        block = np.asarray(images[start : start + batch_size], dtype=np.float64)
        differences = (
            block[:, :-1, :-1]
            - block[:, 1:, :-1]
            - block[:, :-1, 1:]
            + block[:, 1:, 1:]
        )[:, valid]
        median = np.median(differences, axis=1, keepdims=True)
        mad = np.median(np.abs(differences - median), axis=1)
        result[start : start + len(block)] = (mad / (2 * 0.6744897501960817)) ** 2
    return result


class NoiseCorrectedCosineDistance:
    """Cache centered images, noise estimates, and corrected image energies.

    Parameters
    ----------
    images : np.ndarray
        Real spatial images of shape (n, h, w), unchanged during iterations.
    mask : np.ndarray, optional
        Binary support. Defaults to the whole image.
    noise_variance : 'auto', float or np.ndarray
        Per-pixel variance before this metric's preprocessing, in the supplied
        image units. Auto uses only observed images, not clean signals or SNR.
    pool_noise : bool
        Pool automatic estimates only if images share the same noise variance.
    min_signal_fraction : float
        Resolve energies only above this fraction of expected noise energy.
        Unresolved inputs receive score zero (distance one), with a warning.
    clip : bool
        Clip cosine to [-1, 1]. Defaults False to avoid a point mass in GMM input.
        Unclipped distances can be negative or greater than two.
    batch_size : int
        Maximum batch size for initial preprocessing and noise estimation.
    filter_sigma : float
        Gaussian smoothing width in pixels, applied only to the metric.
        Zero disables smoothing. This is NOT a noise standard deviation.
    min_frequency, max_frequency : float, optional
        Radial band limits in cycles/pixel. Default: no band restriction.
        Filtering uses periodic Fourier transfer functions, not reflective edges.
    """

    def __init__(
        self,
        images,
        *,
        mask=None,
        noise_variance="auto",
        pool_noise=False,
        min_signal_fraction=0.05,
        clip=False,
        batch_size=64,
        filter_sigma=0.0,
        min_frequency=0.0,
        max_frequency=None,
    ):
        images = np.asarray(images)
        if (
            images.ndim != 3
            or not len(images)
            or np.iscomplexobj(images)
            or not np.isfinite(images).all()
        ):
            raise ValueError("Expected a nonempty finite real batch (n, h, w)")

        self.shape = images.shape

        if mask is None:
            self.support = np.ones(images.shape[1:], dtype=bool)
        else:
            mask = np.asarray(mask)
            if mask.shape != images.shape[1:] or not np.all((mask == 0) | (mask == 1)):
                raise ValueError("mask must be binary and match the image shape")
            self.support = mask.astype(bool)

        count = int(self.support.sum())
        if count < 2 or batch_size < 1:
            raise ValueError("Need at least two support pixels and batch_size >= 1")

        if not np.isfinite(min_signal_fraction) or min_signal_fraction < 0:
            raise ValueError("min_signal_fraction must be finite and nonnegative")
        self.min_signal_fraction = min_signal_fraction

        self.clip = clip
        self._warned_unresolved = False
        self.trace = float(count - 1)
        self._feature_scale = None
        self.noise_profile = None
        self.spectral_weights = None

        if (
            not np.isfinite(filter_sigma)
            or filter_sigma < 0
            or not np.isfinite(min_frequency)
            or min_frequency < 0
            or (
                max_frequency is not None
                and (not np.isfinite(max_frequency) or max_frequency <= min_frequency)
            )
        ):
            raise ValueError("Invalid filter width or radial frequency limits")

        if filter_sigma > 0 or min_frequency > 0 or max_frequency is not None:
            h, w = self.shape[1:]
            radius = np.hypot(fftfreq(h)[:, None], rfftfreq(w)[None, :])
            keep = (radius > 0) & (radius >= min_frequency)
            if max_frequency is not None:
                keep &= radius <= max_frequency
            omega = keep * np.exp(-4 * np.pi**2 * filter_sigma**2 * radius**2)
            # rfft omits conjugate partners: double interior columns, but not
            # DC or the Nyquist column (the latter exists only for even widths).
            multiplicity = np.full(w // 2 + 1, 2.0)
            multiplicity[0] = 1.0
            if w % 2 == 0:
                multiplicity[-1] = 1.0
            self.spectral_weights = omega * multiplicity[None, :]
            mask_fft = rfft2(self.support.astype(float), norm="ortho")
            self.noise_profile = np.maximum(
                count / (h * w) - np.abs(mask_fft) ** 2 / count, 0
            )
            self.trace = float(np.sum(self.spectral_weights * self.noise_profile))
            self._frequency_keep = self.spectral_weights > 0
            if not self._frequency_keep.any():
                raise ValueError("The selected filter retains no Fourier coefficients")
            self._feature_scale = np.sqrt(self.spectral_weights[self._frequency_keep])

        if isinstance(noise_variance, str):
            if noise_variance != "auto":
                raise ValueError(
                    "noise_variance must be 'auto', a scalar, or shape (n,)"
                )
            noise = estimate_noise_variance(images, self.support, batch_size)
            if pool_noise:
                noise[:] = np.median(noise)
        else:
            noise = np.broadcast_to(
                np.asarray(noise_variance, dtype=float), (len(images),)
            ).copy()
        if not np.isfinite(noise).all() or np.any(noise < 0):
            raise ValueError("noise_variance must be finite and nonnegative")
        self.noise_variance = noise

        first = self._features(images[:batch_size])
        self.features = np.empty((len(images), first.shape[1]), dtype=first.dtype)
        self.features[: len(first)] = first
        for start in range(len(first), len(images), batch_size):
            self.features[start : start + batch_size] = self._features(
                images[start : start + batch_size]
            )
        self.image_energy = np.einsum(
            "ij,ij->i", self.features.real, self.features.real
        )
        if np.iscomplexobj(self.features):
            self.image_energy += np.einsum(
                "ij,ij->i", self.features.imag, self.features.imag
            )
        self.noise_energy = self.trace * noise
        self.signal_energy = self.image_energy - self.noise_energy

    def _features(self, images):
        values = np.asarray(images, dtype=np.float64)[:, self.support]
        values -= values.mean(axis=1, keepdims=True)
        if self._feature_scale is None:
            return values
        centered = np.zeros((len(images), *self.shape[1:]), dtype=np.float64)
        centered[:, self.support] = values
        spectrum = rfft2(centered, axes=(-2, -1), norm="ortho")
        return spectrum[:, self._frequency_keep] * self._feature_scale

    def distances(
        self, reference, *, reference_weights=None, reference_noise_variance=0.0
    ):
        """Return one corrected distance per cached image.

        ``reference_weights`` are normalized coefficients of a reference built
        from this exact batch. If omitted, the reference must be independent;
        supply its pixel noise variance explicitly (zero for a clean reference).
        """
        reference = np.asarray(reference)
        if (
            reference.shape != self.shape[1:]
            or np.iscomplexobj(reference)
            or not np.isfinite(reference).all()
        ):
            raise ValueError("reference must be finite, real, and match an image")
        if reference_weights is not None:
            a = np.asarray(reference_weights, dtype=float)
            if (
                a.shape != (self.shape[0],)
                or not np.isfinite(a).all()
                or np.any(a < 0)
                or not np.isclose(a.sum(), 1.0)
            ):
                raise ValueError(
                    "reference_weights must be normalized nonnegative shape (n,)"
                )
            ref_variance = np.dot(a * a, self.noise_variance)
            cross_noise = self.trace * a * self.noise_variance
        else:
            ref_variance = reference_noise_variance
            cross_noise = 0.0
        if not np.isfinite(ref_variance) or ref_variance < 0:
            raise ValueError("reference noise variance must be finite and nonnegative")
        ref = self._features(reference[None])[0]
        cross = (self.features @ ref.conj()).real - cross_noise
        ref_noise_energy = self.trace * ref_variance
        ref_signal = np.vdot(ref, ref).real - ref_noise_energy
        usable = (
            (self.signal_energy > 0)
            & (ref_signal > 0)
            & (self.signal_energy > self.min_signal_fraction * self.noise_energy)
            & (ref_signal > self.min_signal_fraction * ref_noise_energy)
        )
        denominator = np.sqrt(np.maximum(self.signal_energy, 0) * max(ref_signal, 0))
        scores = np.divide(cross, denominator, out=np.zeros_like(cross), where=usable)
        self.last_unresolved = ~usable
        self.last_clipped = np.abs(scores) > 1
        self.last_scores = np.clip(scores, -1, 1) if self.clip else scores
        if np.any(~usable) and not self._warned_unresolved:
            warnings.warn(
                "Noise correction left signal energies unresolved; those distances are 1. "
                "Inspect noise estimates or try filtering.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_unresolved = True
        return 1.0 - self.last_scores


class NoiseCorrectedCosineEstimator(Estimator):
    """Iterative image weighting with a cached noise-corrected cosine metric.

    ``weighting='gmm'`` reuses the existing recursive GMM's fitting, peak
    correction, and degeneracy checks; ``'cosine'`` uses max(0, min(1, cosine))
    raised to ``weight_power``. Both update the original, unfiltered images.
    An all-zero weight update falls back to the ordinary mean and stops.
    """

    def __init__(
        self,
        *,
        weighting="gmm",
        max_iter=10,
        tol=1e-3,
        metric_params=None,
        gmm_params=None,
        weight_power=1.0,
    ):
        if weighting not in ("gmm", "cosine"):
            raise ValueError("weighting must be 'gmm' or 'cosine'")
        if (
            not isinstance(max_iter, int)
            or max_iter < 0
            or not np.isfinite(tol)
            or tol < 0
            or not np.isfinite(weight_power)
            or weight_power <= 0
        ):
            raise ValueError("Invalid iteration limit, tolerance, or weight_power")
        self.weighting = weighting
        self.max_iter = max_iter
        self.tol = tol
        self.metric_params = dict(metric_params or {})
        self.gmm_params = dict(gmm_params or {})
        self.weight_power = weight_power

    def fit(
        self,
        images: ImageBatch,
        *,
        reference=None,
        initial_reference_weights=None,
        reference_noise_variance=0.0,
    ):
        """Fit using a known linear initial reference, or an independent one.

        For a supplied dataset mean, pass uniform ``initial_reference_weights``.
        A supplied reference without coefficients is treated as independent.
        Nonlinear references (e.g. medians) need a different noise model.
        """
        if images.ctf is not None:
            raise ValueError("Noise-corrected cosine does not support CTF models yet")
        imgs = images.real
        n = len(imgs)
        a = initial_reference_weights
        if reference is None:
            reference = imgs.mean(axis=0)
            a = np.full(n, 1.0 / n)
        reference = np.array(reference, dtype=np.float64, copy=True)
        weights = np.ones((n, 1, 1), dtype=np.float64)
        self.converged = self.max_iter == 0
        self.n_its = 0
        self.last_distances = None
        diagnostics = None
        params = dict(self.metric_params)
        params.setdefault("mask", images.mask)
        self.metric = NoiseCorrectedCosineDistance(imgs, **params)

        def distance_function(_images, ref):
            return self.metric.distances(
                ref,
                reference_weights=a,
                reference_noise_variance=reference_noise_variance,
            )

        gmm = (
            RecursiveGMMEstimator(
                distance_function=distance_function, **self.gmm_params
            )
            if self.weighting == "gmm"
            else None
        )
        for iteration in range(self.max_iter):
            self.n_its = iteration + 1
            degenerate = False
            if gmm is not None:
                degenerate, update, weights, diagnostics = gmm._fit_one_iteration(
                    imgs,
                    reference,
                    initialize_params=gmm.initialize_params and iteration == 0,
                )
                self.last_distances = diagnostics.distances
            else:
                self.last_distances = distance_function(imgs, reference)
                w = np.clip(1.0 - self.last_distances, 0, 1) ** self.weight_power
                weights = w[:, None, None]
                if w.sum() <= 1e-8:
                    warnings.warn(
                        "All cosine weights vanished; returning the ordinary mean.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    weights = np.ones_like(weights)
                    degenerate = True
                a_next = weights.reshape(n) / weights.sum()
                update = np.tensordot(a_next, imgs, axes=(0, 0))
            a = weights.reshape(n) / weights.sum()
            relative_change = np.linalg.norm(update - reference) / max(
                np.linalg.norm(reference), 1e-8
            )
            reference = update
            if degenerate:
                break
            if relative_change < self.tol:
                self.converged = True
                break
        self.reference_weights = a
        return EstimatorResult(
            real_estimate=reference,
            weights={"real": weights, "fourier": None},
            gmm_diagnostics=diagnostics,
        )
