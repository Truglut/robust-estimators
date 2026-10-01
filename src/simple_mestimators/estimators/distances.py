from __future__ import annotations

from typing import Callable

import numpy as np

from .weights import tagare_weights, configured_function

DistanceFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def mean_l1_norm(
    images: np.ndarray,
    reference: np.ndarray,
    std: float | np.ndarray = 1.0,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """
    Calculates the mean L1 distance from each image to the reference.
    
    Returns
    -------
    np.ndarray
        Array of shape ``(n,)`` containing the mean absolute distance from
        each image to the reference.
    """
    residuals = (images - reference) / np.maximum(std, eps)

    spatial_axes = tuple(range(1, images.ndim))

    return np.abs(residuals).mean(axis=spatial_axes)


def mean_l2_norm(
    images: np.ndarray,
    reference: np.ndarray,
    std: float | np.ndarray = 1.0,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """
    Calculates the mean L2 distance (i.e. the square root of the mean squared error) 
    from each image to the reference.
    
    Returns
    -------
    np.ndarray
        Array of shape ``(n,)`` containing the mean L2 distance from
        each image to the reference.
    """
    residuals = (images - reference) / np.maximum(std, eps)

    spatial_axes = tuple(range(1, images.ndim))

    return np.sqrt(np.square(np.abs(residuals)).mean(axis=spatial_axes))


def tagare_distance(
    images: np.ndarray,
    reference: np.ndarray,
    std: np.ndarray | float = 1.0,
    beta: float = 1.0e-6,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """
    Calculates the distance based on Tagare weights (similarity) between the
    reference and each of the input images.

    Returns
    -------
    np.ndarray
        Array of shape (n,) containing the inverted Tagare similarity distances.
    """
    return -tagare_weights(images, reference, std, beta, eps).reshape(-1)


DISTANCE_FUNCTION_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "l2": mean_l2_norm,
    "l2_norm": mean_l2_norm,
    "l1": mean_l1_norm,
    "l1_norm": mean_l1_norm,
    "tagare_weights": tagare_distance,
}


def get_distance_function(
    name: str, params: dict | tuple, imgs: np.ndarray | None = None
) -> DistanceFunction:
    """
    Factory function to retrieve and pre-configure a specific distance function via partial evaluation.

    Parameters
    ----------
    name : str
        The registration key name of the distance metric (e.g., 'l2', 'tagare_weights').
    params : dict | tuple
        Hyperparameters passed down to the base function. If a tuple of pairs is given,
        it is transformed into a dictionary seamlessly.
    imgs : np.ndarray | None, optional
        A collection of images tensor used exclusively for calculating automated beta scaling
        parameter if params configures it to "auto", by default None.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        A standard configured partial function expecting `(images, reference)` arguments.

    Raises
    -------
    ValueError
        If the name is not found within `FUNCTION_REGISTRY` or if 'auto' beta scaling
        is requested without passing the backing `imgs` tensor.
    """
    return configured_function(DISTANCE_FUNCTION_REGISTRY, name, params, imgs)
