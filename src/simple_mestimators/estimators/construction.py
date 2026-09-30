from typing import Literal, Callable, get_args
from copy import deepcopy

import numpy as np

from simple_mestimators.domain import ImageSpace, ImageBatch

from simple_mestimators.estimators.base import Estimator
from simple_mestimators.estimators.admm import ADMMEstimator
from simple_mestimators.estimators.irls import IRLSSolver
from simple_mestimators.estimators.fourier_irls import JointIRLSFourier
from simple_mestimators.estimators.gmm import RecursiveGMMEstimator, FourierGMM

from simple_mestimators.estimators.distances import get_distance_function
from simple_mestimators.estimators.weights import (
    get_weight_function,
    norm_smooth_redescending_weights,
)

from simple_mestimators.masks import create_lowpass_rfft_mask

LOWPASS_NORMALIZED_CUTOFF = 0.3
DEFAULT_FOURIER_GMM_DELTA = 1.5

EstimatorTypeName = Literal[
    "m_estimator",
    "joint_fourier",
    "recursive_gmm",
    "admm",
    "masked_fourier",
    "fourier_gmm",
]


def _build_m_estimator(
    params: dict, image_batch: ImageBatch, space: ImageSpace = "real"
) -> IRLSSolver:
    weight_func = get_weight_function(
        name=params["weight_function"],
        params=params.get("weight_params", {}),
        imgs=image_batch.select_space(space),
    )
    return IRLSSolver(weight_function=weight_func, **params.get("solver_params", {}))


def _build_joint_fourier(params: dict, image_batch: ImageBatch) -> JointIRLSFourier:
    if params["weight_function"].startswith("norm"):
        weight_approach = "per-image"
    else:
        weight_approach = "per-coefficient"
    solver = _build_m_estimator(params, image_batch, space="fourier")
    return JointIRLSFourier(solver, weight_approach=weight_approach, mask=None)


def _build_masked_fourier(params: dict, image_batch: ImageBatch) -> JointIRLSFourier:
    if params["weight_function"].startswith("norm"):
        weight_approach = "per-image"
    else:
        weight_approach = "per-coefficient"

    mask_cutoff = params.pop("lowpass_cutoff", LOWPASS_NORMALIZED_CUTOFF)
    mask = create_lowpass_rfft_mask(
        image_shape=image_batch.real_shape, cutoff=mask_cutoff, unit="normalized"
    )

    solver = _build_m_estimator(params, image_batch, space="fourier")

    return JointIRLSFourier(
        irls_solver=solver, weight_approach=weight_approach, mask=mask
    )


def _build_recursive_gmm(
    params: dict, image_batch: ImageBatch, space: ImageSpace = "real"
) -> RecursiveGMMEstimator:
    distance_func = get_distance_function(
        name=params["distance_function"],
        params=params.get("distance_params", {}),
        imgs=image_batch.select_space(space),
    )
    return RecursiveGMMEstimator(
        distance_function=distance_func,
        random_state=params.get("random_state", None),
        max_iter=params.get("max_iter", 10),
        tol=params.get("tol", 1e-3),
    )


def _build_admm(params: dict, image_batch: ImageBatch) -> ADMMEstimator:
    irls_real = _build_m_estimator(params["real_estimator"], image_batch, space="real")
    irls_fourier = _build_joint_fourier(params["real_estimator"], image_batch)

    return ADMMEstimator(
        irls_real=irls_real,
        irls_fourier=irls_fourier,
        **params.get("solver_params", {}),
    )


def _build_fourier_gmm(params: dict, image_batch: ImageBatch):
    lowpass_mask = create_lowpass_rfft_mask(
        image_shape=image_batch.real_shape,
        cutoff=LOWPASS_NORMALIZED_CUTOFF,
        unit="normalized",
    )
    fourier_variance = image_batch.fourier_modulus_variance()[lowpass_mask]
    std = np.sqrt(fourier_variance)
    delta = params.pop("delta", DEFAULT_FOURIER_GMM_DELTA)

    def distance_function(images, reference):
        weights = norm_smooth_redescending_weights(
            images[:, lowpass_mask], reference[lowpass_mask], std=std, delta=delta
        )
        return -weights.reshape(images.shape[0], 1)
        # Uncomment for directly using L1 norm as distance
        # import torch
        # residuals = (images[:, lowpass_mask] - reference[lowpass_mask]) / std
        # dims = tuple(range(1, residuals.ndim))
        # return torch.linalg.vector_norm(residuals[:, lowpass_mask], ord=1, dim=dims)

    solver = RecursiveGMMEstimator(
        distance_function=distance_function,
        random_state=params.get("random_state", None),
        max_iter=params.get("max_iter", 10),
        tol=params.get("tol", 1e-3),
    )

    return FourierGMM(solver=solver)


BUILDER_REGISTRY: dict[EstimatorTypeName, Callable[[dict, ImageBatch], Estimator]] = {
    "m_estimator": _build_m_estimator,
    "joint_fourier": _build_joint_fourier,
    "masked_fourier": _build_masked_fourier,
    "recursive_gmm": _build_recursive_gmm,
    "admm": _build_admm,
    "fourier_gmm": _build_fourier_gmm,
}


def build_estimator(method_cfg: dict, image_batch: ImageBatch) -> Estimator:
    """
    Factory function that reads the YAML config block and returns
    the instantiated Estimator object on the specified device.
    """
    estimator_type = method_cfg["type"]
    params = deepcopy(method_cfg.get("params", {}))

    if estimator_type not in BUILDER_REGISTRY:
        raise ValueError(
            f"Unknown estimator type: {estimator_type}. "
            f"Valid types are {get_args(EstimatorTypeName)}"
        )

    return BUILDER_REGISTRY[estimator_type](params, image_batch)
