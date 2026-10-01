"""
Robust weighting functions for M-estimators.

The functions in this module compute either pixelwise weights or global
image-level weights used by the IRLS estimators.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

TAGARE_CONSTANT: float = 1.0e-5
WeightFunction = Callable[[np.ndarray, np.ndarray, np.ndarray | float], np.ndarray]


def weighted_average(
    images: np.ndarray, weights: np.ndarray, axis: int = 0, eps: float = 1.0e-6
) -> np.ndarray:
    """Computes the weighted average of an input image array along a specified axis.

    Parameters
    ----------
    images : np.ndarray
        Input data array containing image components or batches.
    weights : np.ndarray
        Weight coefficients matching or broadcastable to the shape of `images`.
    axis : int, optional
        The axis along which the average is computed, by default 0.
    eps : float, optional
        Small constant for numerical stability to avoid zero-division, by default 1.0e-6.

    Returns
    -------
    np.ndarray
        The resulting weighted average array with `axis` reduced.

    Raises
    ------
    ValueError
        If the maximum weight sum is effectively zero, indicating degenerate metrics.
    """
    weight_sum = np.sum(weights, axis=axis, keepdims=True)

    if np.max(weight_sum) < 1e-8:
        raise ValueError("All weights are effectively zero")

    weighted_sum = np.sum(images * weights, axis=axis, keepdims=True)
    result = weighted_sum / np.maximum(weight_sum, eps)

    return np.squeeze(result, axis=axis)


def smooth_redescending_weights(
    images: np.ndarray,
    reference: np.ndarray,
    std: float | np.ndarray,
    delta: float,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """
    Computes smooth redescending M-estimator weights using a Gaussian-like influence metric.

    Parameters
    ----------
    images : np.ndarray
        Images tensor of shape (n, h, w).
    reference : np.ndarray
        Reference template tensor of shape (h, w).
    std : np.ndarray | float
        Standard deviation scale factor.
    delta : float
        Tuning parameter governing the rejection scale threshold of outlier features.
    eps : float, optional
        Stabilizing denominator offset, by default 1.0e-8.

    Returns
    -------
    np.ndarray
        Tensor of shape (n, h, w) containing the smooth redescending weights.
    """
    abs_residuals = np.abs((images - reference) / np.maximum(std, eps))
    variance_scale = delta**2

    sq_residuals = np.square(abs_residuals, out=abs_residuals)

    return np.exp(-sq_residuals / variance_scale)


def norm_smooth_redescending_weights(
    images: np.ndarray,
    reference: np.ndarray,
    std: float | np.ndarray,
    delta: float,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """
    Computes smooth redescending M-estimator weights using a Gaussian-like influence metric.

    Parameters
    ----------
    images : np.ndarray
        Images tensor of shape (n, h, w).
    reference : np.ndarray
        Reference template tensor of shape (h, w).
    std : np.ndarray | float
        Standard deviation scale factor.
    delta : float
        Tuning parameter governing the rejection scale threshold of outlier features.
    eps : float, optional
        Stabilizing denominator offset, by default 1.0e-8.

    Returns
    -------
    np.ndarray
        Tensor of shape (n, 1, 1) containing the smooth redescending weights.
    """
    abs_residuals = np.abs((images - reference) / np.maximum(std, eps))

    axes = tuple(range(1, abs_residuals.ndim))
    mean_squared_residual = np.square(abs_residuals, out=abs_residuals).mean(
        axis=axes, keepdims=True
    )

    return np.exp(-mean_squared_residual / delta**2)


def tagare_weights(
    images: np.ndarray,
    reference: np.ndarray,
    std: np.ndarray | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """
    Calculate non-local structural similarity weights following the
    Huang-Tagare model.

    Parameters
    ----------
    images : np.ndarray
        Images tensor of shape ``(n, h, w)``.
    reference : np.ndarray
        Reference template tensor of shape ``(h, w)``.
    std : np.ndarray or float, optional
        Unused parameter retained for compatibility with the weight-function
        interface, by default 1.0.
    beta : float, optional
        Scale coefficient for the exponential penalty applied to the
        component orthogonal to the reference, by default 1.0e-6.
    eps : float, optional
        Small value used to avoid division by zero, by default 1.0e-8.

    Returns
    -------
    np.ndarray
        Image-level weights with shape ``(n, 1, 1)``.
    """
    images_flat = images.reshape(images.shape[0], -1)
    reference_flat = reference.flatten()

    image_norms = np.linalg.vector_norm(images_flat, axis=1)
    reference_norm = np.linalg.vector_norm(reference_flat)

    dot_products = images_flat @ reference_flat.conj()

    cos = dot_products / (image_norms * reference_norm).clip(min=eps)
    cos_abs = np.abs(cos)

    image_norm_sq = np.square(image_norms)
    orth_norm_sq = np.maximum(image_norm_sq * (1.0 - np.square(cos_abs)), 0.0)

    weights = cos_abs * np.exp(-beta * orth_norm_sq)

    return weights.reshape(-1, *([1] * reference.ndim))


# Global weight configuration mappings
FUNCTION_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "smooth": smooth_redescending_weights,
    "norm_smooth": norm_smooth_redescending_weights,
    "global": tagare_weights,
    "tagare_weights": tagare_weights,
}

# Set of functions that need the beta parameter, including distance functions
NEED_BETA_PARAMETER = ("global", "tagare_weights")


def configured_function(
    registry: dict[str, Callable[..., np.ndarray]],
    name: str,
    params: dict | tuple | None,
    images: np.ndarray | None = None,
):
    """
    Retrieves function from registry and configures it with the given parameters
    through partial evaluation

    Parameters
    ----------
    registry : dict[str, Callable[..., np.ndarray]]
        Function registry mapping names to functions
    name : str
        The name of the requested function. Should match a key in `registry`
    params : dict | tuple | None
        Specified parameters for the function.
    images : np.ndarray | None, optional
        Images tensor, used to calculate the "auto" beta parameter, by default None

    Returns
    -------
    Callable
        The configured function

    Raises
    ------
    ValueError
        If `name` does not match any of the keys in `registry`
    ValueError
        If `beta="auto"` is requested but the images are not provided
    """
    try:
        fn = registry[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown function name: {name!r}. Available: {sorted(registry)}"
        ) from exc
    params_dict = dict(params or {})
    if name in NEED_BETA_PARAMETER and params_dict.get("beta", "auto") == "auto":
        if images is None:
            raise ValueError(
                f"Cannot calculate automatic beta for {name!r} without images."
            )
        multiplier = params_dict.pop("auto_multiplier", 1.0)
        params_dict["beta"] = calculate_beta_auto(images, multiplier)
    return partial(fn, **params_dict)


def get_weight_function(
    name: str, params: dict | tuple | None = None, imgs: np.ndarray | None = None
) -> WeightFunction:
    """
    Factory configuration utility that builds specialized partial instances of weighting
    functions loaded with target parameter configurations.

    Parameters
    ----------
    name : str
        The registration key mapping string token corresponding to the robust weight function.
    params : dict | tuple
        Hyperparameters passed to define runtime properties. If a tuple of pairs is supplied,
        it is transformed into a standard dictionary.
    images : np.ndarray | None, optional
        Contextual batch images dataset context tensor required to compute automated beta scale
        parameters if config indicates "beta": "auto", by default None.

    Returns
    -------
    Callable[..., np.ndarray]
        A pre-configured robust weight calculation partial function instance.

    Raises
    ------
    ValueError
        If the name is missing from the registry, or if automated beta calculations are
        requested without passing an accompanying data dataset context tensor.
    """
    return configured_function(FUNCTION_REGISTRY, name, params, imgs)


def calculate_beta_auto(imgs: np.ndarray, mult: float = 1.0) -> float:
    """
    Automatically scales the Tagare exponential scaling factor based on the average
    variance across the provided input images dataset.

    Parameters
    ----------
    images : np.ndarray
        Input batch dataset images tensor of shape (n, h, w).
    mult : float, optional
        Scalar multiplier adjustment value modifying the baseline parameter scaling, by default 1.0.

    Returns
    -------
    float
        The calculated floating-point automatic beta scaling parameter.
    """
    axes = tuple(range(1, imgs.ndim))
    variance = float(imgs.var(axis=axes).mean())
    return mult * TAGARE_CONSTANT / max(variance, 1.0e-8)
