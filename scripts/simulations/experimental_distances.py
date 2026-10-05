"""Experimental, signed structural distances for real cryo-EM image batches.

All metrics return shape (n,) with smaller values meaning greater similarity.
They accept real spatial images, perform their own full, orthonormal FFTs, and
do not search over rotations/translations: misalignment should remain visible.
No metric or noise estimate uses labels, clean images, or the simulation SNR.

Noise debiasing assumes independent additive white noise inside a binary mask.
The automatic estimator additionally assumes locally smooth signal. Its model
does not cover arbitrary CTF-filtered/colored noise. Estimated signal energies
and a balanced reference are experimental plug-in estimates, not guarantees.
"""

from __future__ import annotations

from functools import partial
from typing import Callable
import warnings

import numpy as np
from scipy.fft import fft2, fftfreq


def _support(images: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if images.ndim != 3 or np.iscomplexobj(images) or len(images) == 0:
        raise ValueError("Expected a nonempty real spatial batch of shape (n, h, w)")
    if not np.isfinite(images).all():
        raise ValueError("images contain non-finite values")
    if mask is None:
        support = np.ones(images.shape[-2:], dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.shape != images.shape[-2:] or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary and match the spatial image shape")
        support = mask.astype(bool)
    if np.count_nonzero(support) < 2:
        raise ValueError("At least two pixels must lie inside the mask")
    return support


def _centered_fft(images: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float64)
    means = values[..., support].mean(axis=-1)
    centered = np.where(support, values - means[..., None, None], 0.0)
    return fft2(centered, axes=(-2, -1), norm="ortho")


def _frequency_radius(shape: tuple[int, int]) -> np.ndarray:
    return np.hypot(fftfreq(shape[0])[:, None], fftfreq(shape[1])[None, :])


def _spectral_weights(
    shape: tuple[int, int], sigma: float, min_frequency: float,
    max_frequency: float | None,
) -> np.ndarray:
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(min_frequency) or min_frequency < 0:
        raise ValueError("min_frequency must be finite and nonnegative")
    if max_frequency is not None and (
        not np.isfinite(max_frequency) or max_frequency <= min_frequency
    ):
        raise ValueError("max_frequency must be finite and greater than min_frequency")
    radius = _frequency_radius(shape)
    keep = (radius > 0) & (radius >= min_frequency)
    if max_frequency is not None:
        keep &= radius <= max_frequency
    if not np.any(keep):
        raise ValueError("The selected frequency range contains no Fourier coefficients")
    # Squared transfer function of a periodic Gaussian with spatial std sigma.
    return keep * np.exp(-4 * np.pi**2 * sigma**2 * radius**2)


def _noise_power_profile(support: np.ndarray) -> np.ndarray:
    """Expected Fourier power for unit white noise, masked and centered.

    For K support pixels in a P-pixel box, the covariance in real space is
    diag(mask) - mask @ mask.T / K. Its Fourier diagonal is
    K/P - abs(FFT_ortho(mask))**2/K. This accounts for the mask and centering;
    using just the number of retained frequencies would be incorrect.
    """
    count = float(np.count_nonzero(support))
    spectrum = fft2(support.astype(float), norm="ortho")
    return np.maximum(count / support.size - np.abs(spectrum)**2 / count, 0.0)


def estimate_noise_variance(
    images: np.ndarray, *, mask: np.ndarray | None = None,
    pool_noise: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Estimate white-noise variance from the noisy input, before filtering.

    Parameters
    ----------
    images : np.ndarray
        Real spatial batch of shape (n, h, w).
    mask : np.ndarray, optional
        Binary signal/noise support. Only 2x2 blocks fully inside it are used.
    pool_noise : bool, optional
        Replace per-image estimates by their median. Use only when all images
        have the same noise variance in the current units; image-wise
        post-noise standardization can invalidate that assumption.
    batch_size : int, optional
        Maximum number of images processed together.

    Returns
    -------
    np.ndarray
        Estimated per-image noise variances. The 2x2 checkerboard difference
        has variance 4*sigma_noise**2 under independent Gaussian noise, so its
        MAD divided by 2*0.67448975 estimates sigma_noise. Locally smooth
        signal cancels approximately; high-frequency signal can bias it up.
    """
    images = np.asarray(images)
    support = _support(images, mask)
    valid = (support[:-1, :-1] & support[1:, :-1]
             & support[:-1, 1:] & support[1:, 1:])
    if not np.any(valid):
        raise ValueError("Automatic noise estimation requires at least one valid 2x2 block")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    variances = np.empty(len(images))
    for start in range(0, len(images), batch_size):
        block = np.asarray(images[start:start + batch_size], dtype=np.float64)
        difference = (block[:, :-1, :-1] - block[:, 1:, :-1]
                      - block[:, :-1, 1:] + block[:, 1:, 1:])[:, valid]
        median = np.median(difference, axis=1, keepdims=True)
        mad = np.median(np.abs(difference - median), axis=1)
        variances[start:start + len(block)] = (mad / (2 * 0.6744897501960817))**2
    if pool_noise:
        variances[:] = np.median(variances)
    if not np.isfinite(variances).all():
        raise ValueError("Noise estimates contain non-finite values")
    return variances


def _resolve_noise(images, mask, noise_variance, pool_noise, batch_size):
    if isinstance(noise_variance, str):
        if noise_variance != "auto":
            raise ValueError("noise_variance must be 'auto', a scalar, or shape (n,)")
        return estimate_noise_variance(
            images, mask=mask, pool_noise=pool_noise, batch_size=batch_size
        )
    values = np.asarray(noise_variance, dtype=float)
    if values.ndim == 0:
        values = np.full(len(images), float(values))
    if values.shape != (len(images),) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("noise_variance must be finite, nonnegative, and scalar or shape (n,)")
    return values


def _reference_weights(reference_weights, n_images):
    if reference_weights is None:
        return None
    weights = np.asarray(reference_weights, dtype=float)
    if (weights.shape != (n_images,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("reference_weights must be nonnegative, shape (n,), and sum to one")
    return weights


def _spectral_pairs(images, reference, support, reference_weights, leave_one_out, batch_size):
    reference = np.asarray(reference)
    if reference.shape != images.shape[1:] or np.iscomplexobj(reference):
        raise ValueError("reference must be a real spatial image of shape (h, w)")
    if not np.isfinite(reference).all():
        raise ValueError("reference contains non-finite values")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if leave_one_out and (reference_weights is None or np.any(reference_weights >= 1.0)):
        raise ValueError("leave_one_out requires linear reference weights strictly below one")
    reference_fft = _centered_fft(reference, support)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        image_fft = _centered_fft(images[start:stop], support)
        if leave_one_out:
            w = reference_weights[start:stop, None, None]
            ref_fft = (reference_fft - w * image_fft) / (1.0 - w)
        else:
            ref_fft = reference_fft
        yield slice(start, stop), image_fft, ref_fft


def _moments(image_fft, reference_fft, spectral_weights):
    axes = (-2, -1)
    cross = np.sum((image_fft * reference_fft.conj()).real * spectral_weights, axis=axes)
    image_energy = np.sum(np.abs(image_fft)**2 * spectral_weights, axis=axes)
    reference_energy = np.sum(np.abs(reference_fft)**2 * spectral_weights, axis=axes)
    return cross, image_energy, reference_energy


def _cosine(cross, image_energy, reference_energy):
    denominator = np.sqrt(np.maximum(image_energy, 0) * np.maximum(reference_energy, 0))
    score = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
    return np.clip(score, -1.0, 1.0)


def lowpass_cosine_distance(
    images: np.ndarray, reference: np.ndarray, *, sigma: float = 2.0,
    min_frequency: float = 0.0, max_frequency: float | None = 0.25,
    mask: np.ndarray | None = None, reference_weights: np.ndarray | None = None,
    leave_one_out: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Return 1 minus signed cosine after centering and Gaussian/band filtering.

    ``sigma`` is in pixels; frequencies are in cycles/pixel. Positive scaling
    and constant offsets are invariant without noise. Filtering improves the
    effective SNR but does not fully remove unequal-SNR attenuation. Set
    sigma=0 and max_frequency=None for the unfiltered centered-cosine baseline.
    ``reference_weights`` describe a linear reference formed from this batch;
    supplying them enables exact removal of each image's own contribution.
    """
    images = np.asarray(images)
    support = _support(images, mask)
    weights = _reference_weights(reference_weights, len(images))
    spectral_weights = _spectral_weights(images.shape[1:], sigma, min_frequency, max_frequency)
    distances = np.empty(len(images))
    for idx, image_fft, ref_fft in _spectral_pairs(
        images, reference, support, weights, leave_one_out, batch_size
    ):
        distances[idx] = 1.0 - _cosine(*_moments(image_fft, ref_fft, spectral_weights))
    return distances


def noise_debiased_cosine_distance(
    images: np.ndarray, reference: np.ndarray, *, sigma: float = 2.0,
    min_frequency: float = 0.0, max_frequency: float | None = 0.25,
    noise_variance: str | float | np.ndarray = "auto", pool_noise: bool = False,
    reference_noise_variance: float = 0.0, min_signal_fraction: float = 0.05,
    mask: np.ndarray | None = None, reference_weights: np.ndarray | None = None,
    leave_one_out: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Estimate latent signed cosine by subtracting expected noise energy.

    Parameters
    ----------
    images, reference : np.ndarray
        Real spatial image batch and reference. Smaller output is more similar.
    sigma, min_frequency, max_frequency : float
        Gaussian std in pixels and radial frequency limits in cycles/pixel.
    noise_variance : 'auto', float or np.ndarray, optional
        Noise variance BEFORE this metric's filtering, in current image units.
        'auto' uses only noisy inputs. A scalar/per-image array allows calibration.
    pool_noise : bool, optional
        Pool automatic noise estimates only for a common noise variance.
    reference_noise_variance : float, optional
        White-noise variance of an independent reference, default 0. Ignored
        when reference_weights supplies its linear construction from the batch.
    min_signal_fraction : float, optional
        Treat signal energy estimates below this fraction of expected noise
        energy as unresolved. These receive neutral distance 1, not perfect
        similarity. This stabilizer is a tuning parameter, not a confidence test.
    mask : np.ndarray, optional
        Binary mask. Masking and centering enter the exact noise-energy trace.
    reference_weights : np.ndarray, optional
        Normalized linear reference weights. For fixed weights the reference
        noise variance and self-noise covariance are accounted for exactly.
        For data-dependent weights these are plug-in approximations.
    leave_one_out : bool, optional
        Remove the image from its linear reference before calculating the score.
    batch_size : int, optional
        Maximum FFT batch size.

    Returns
    -------
    np.ndarray
        Distances in [0, 2]. Estimated cosines are clipped to [-1, 1]; clipping
        and unresolved counts should be examined when tuning this estimator.
    """
    images = np.asarray(images)
    support = _support(images, mask)
    weights = _reference_weights(reference_weights, len(images))
    if not np.isfinite(min_signal_fraction) or min_signal_fraction < 0:
        raise ValueError("min_signal_fraction must be finite and nonnegative")
    if not np.isfinite(reference_noise_variance) or reference_noise_variance < 0:
        raise ValueError("reference_noise_variance must be finite and nonnegative")
    noise = _resolve_noise(images, support, noise_variance, pool_noise, batch_size)
    spectral_weights = _spectral_weights(images.shape[1:], sigma, min_frequency, max_frequency)
    trace = float(np.sum(spectral_weights * _noise_power_profile(support)))
    ref_variance = (float(np.sum(weights**2 * noise)) if weights is not None
                    else reference_noise_variance)
    distances = np.empty(len(images))
    unresolved = 0
    for idx, image_fft, ref_fft in _spectral_pairs(
        images, reference, support, weights, leave_one_out, batch_size
    ):
        cross, image_energy, ref_energy = _moments(image_fft, ref_fft, spectral_weights)
        if leave_one_out:
            w = weights[idx]
            ref_noise = (ref_variance - w**2 * noise[idx]) / (1.0 - w)**2
            cross_noise = 0.0
        else:
            ref_noise = ref_variance
            cross_noise = weights[idx] * noise[idx] if weights is not None else 0.0
        image_signal = image_energy - trace * noise[idx]
        ref_signal = ref_energy - trace * np.maximum(ref_noise, 0.0)
        usable = ((image_signal > min_signal_fraction * trace * noise[idx])
                  & (ref_signal > min_signal_fraction * trace * ref_noise)
                  & (image_signal > 0) & (ref_signal > 0))
        score = _cosine(cross - trace * cross_noise, image_signal, ref_signal)
        distances[idx] = 1.0 - np.where(usable, score, 0.0)
        unresolved += np.count_nonzero(~usable)
    if unresolved:
        warnings.warn(
            f"Noise debiasing left {unresolved}/{len(images)} signal energies unresolved; "
            "those distances are neutral (1). Check filtering and noise estimates.",
            RuntimeWarning, stacklevel=2,
        )
    return distances


def bandwise_cosine_distance(
    images: np.ndarray, reference: np.ndarray, *,
    band_edges: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    mask: np.ndarray | None = None, reference_weights: np.ndarray | None = None,
    leave_one_out: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Return 1 minus the equal-band average of signed Fourier cosines.

    Each radial band is normalized separately and receives equal weight,
    preventing a high-energy low-frequency band from dominating the sum.
    This preserves directional phase information but does not correct for
    noise attenuation. Avoid extending the bands far into pure-noise regions.
    Edges are cycles/pixel; DC is always excluded. Empty/zero-power bands
    contribute neutral similarity, without changing other bands' weights.
    """
    images = np.asarray(images)
    support = _support(images, mask)
    weights = _reference_weights(reference_weights, len(images))
    edges = np.asarray(band_edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("band_edges must be finite, nonnegative, and strictly increasing")
    radius = _frequency_radius(images.shape[1:])
    bands = [(radius > 0) & (radius >= lower) & (radius < upper)
             for lower, upper in zip(edges[:-1], edges[1:])]
    if not any(np.any(band) for band in bands):
        raise ValueError("The selected bands contain no Fourier coefficients")
    distances = np.empty(len(images))
    for idx, image_fft, ref_fft in _spectral_pairs(
        images, reference, support, weights, leave_one_out, batch_size
    ):
        score = sum(_cosine(*_moments(image_fft, ref_fft, band)) for band in bands)
        distances[idx] = 1.0 - score / len(bands)
    return distances


def phase_cosine_distance(
    images: np.ndarray, reference: np.ndarray, *,
    min_frequency: float = 0.02, max_frequency: float | None = 0.20,
    relative_floor: float = 1.0e-6,
    mask: np.ndarray | None = None, reference_weights: np.ndarray | None = None,
    leave_one_out: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Return 1 minus equal-frequency phase agreement at the current alignment.

    The score is mean(cos(phase_image - phase_reference)) over selected nonzero
    coefficients. Amplitudes are removed rather than used as weights. A
    relative coefficient floor excludes numerical zeros and near-zero reference
    coefficients. Phase-only scores can fail when phases are noise-dominated;
    restrict the frequency range instead of assuming whitening removes noise.
    """
    images = np.asarray(images)
    support = _support(images, mask)
    weights = _reference_weights(reference_weights, len(images))
    if not np.isfinite(relative_floor) or not 0 <= relative_floor < 1:
        raise ValueError("relative_floor must lie in [0, 1)")
    band = _spectral_weights(images.shape[1:], 0.0, min_frequency, max_frequency).astype(bool)
    distances = np.empty(len(images))
    for idx, image_fft, ref_fft in _spectral_pairs(
        images, reference, support, weights, leave_one_out, batch_size
    ):
        image = image_fft[:, band]
        ref = ref_fft[..., band]
        image_abs, ref_abs = np.abs(image), np.abs(ref)
        valid = ((image_abs > relative_floor * image_abs.max(axis=-1, keepdims=True))
                 & (ref_abs > relative_floor * ref_abs.max(axis=-1, keepdims=True)))
        unit_product = np.divide(
            (image * ref.conj()).real, image_abs * ref_abs,
            out=np.zeros_like(image_abs), where=valid,
        )
        count = valid.sum(axis=-1)
        score = np.divide(unit_product.sum(axis=-1), count,
                          out=np.zeros(len(image)), where=count > 0)
        distances[idx] = 1.0 - np.clip(score, -1.0, 1.0)
    return distances


def contrast_balanced_reference(
    images: np.ndarray, *, mask: np.ndarray | None = None, sigma: float = 2.0,
    min_frequency: float = 0.0, max_frequency: float | None = 0.25,
    noise_variance: str | float | np.ndarray = "auto", pool_noise: bool = False,
    min_signal_fraction: float = 0.05, max_weight_ratio: float = 5.0,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Form a mean with bounded inverse estimated signal-norm contributions.

    Signal norms are estimated from centered, filtered noisy images after
    subtracting expected noise energy. The weights are then applied to the
    ORIGINAL preprocessed images, preserving their spatial representation.
    Unresolved images receive median contribution (weight 1 before normalizing),
    avoiding their amplification through a tiny denominator. Inverse norms are
    clipped within [1/max_weight_ratio, max_weight_ratio] of the median weight.

    Returns
    -------
    reference : np.ndarray
        Real weighted-mean image of shape (h, w).
    reference_weights : np.ndarray
        Linear weights of shape (n,), summing to one. Their data dependence
        means subsequent noise corrections are plug-in approximations.
    """
    images = np.asarray(images)
    support = _support(images, mask)
    if (not np.isfinite(max_weight_ratio) or max_weight_ratio < 1
            or not np.isfinite(min_signal_fraction) or min_signal_fraction < 0):
        raise ValueError("max_weight_ratio must be >= 1 and min_signal_fraction >= 0")
    noise = _resolve_noise(images, support, noise_variance, pool_noise, batch_size)
    spectral_weights = _spectral_weights(images.shape[1:], sigma, min_frequency, max_frequency)
    trace = float(np.sum(spectral_weights * _noise_power_profile(support)))
    signal_energy = np.empty(len(images))
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        spectrum = _centered_fft(images[start:stop], support)
        signal_energy[start:stop] = np.sum(
            np.abs(spectrum)**2 * spectral_weights, axis=(-2, -1)
        ) - trace * noise[start:stop]
    usable = (signal_energy > min_signal_fraction * trace * noise) & (signal_energy > 0)
    contributions = np.ones(len(images))
    if np.any(usable):
        norms = np.sqrt(signal_energy[usable])
        contributions[usable] = np.clip(
            np.median(norms) / norms, 1.0 / max_weight_ratio, max_weight_ratio
        )
    if not np.all(usable):
        warnings.warn(
            f"Balanced reference: {np.count_nonzero(~usable)}/{len(images)} norm estimates "
            "were unresolved; those contributions were left at the median weight.",
            RuntimeWarning, stacklevel=2,
        )
    weights = contributions / contributions.sum()
    reference = np.einsum("n,nhw->hw", weights, images, optimize=True)
    return reference, weights


EXPERIMENTAL_DISTANCE_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "lowpass_cosine": lowpass_cosine_distance,
    "noise_debiased_cosine": noise_debiased_cosine_distance,
    "bandwise_cosine": bandwise_cosine_distance,
    "phase_cosine": phase_cosine_distance,
}


def get_experimental_distance_function(
    name: str, params: dict | None = None, *, mask: np.ndarray | None = None,
    reference_weights: np.ndarray | None = None, leave_one_out: bool = False,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Configure a local experimental metric without changing project registries."""
    configured = dict(params or {})
    reserved = {"mask", "reference_weights", "leave_one_out"} & configured.keys()
    if reserved:
        raise ValueError(f"Set {sorted(reserved)} through the script/reference controls")
    return partial(
        EXPERIMENTAL_DISTANCE_FUNCTIONS[name], **configured,
        mask=mask, reference_weights=reference_weights, leave_one_out=leave_one_out,
    )
