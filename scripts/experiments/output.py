"""Write minimal experiment results in estimator-first directories."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simple_mestimators.domain import EstimatorResult
    from .configs import (
        IOConfig,
        SimulationConfig,
        EstimationConfig,
        EstimationIOConfig,
    )


def snr_directory(snr: float) -> str:
    """Avoid collisions caused by rounding SNR values in directory names."""
    return f"snr_{float(snr)}"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def save_json(path: Path, data: dict) -> None:
    """Write JSON without overwriting an existing file."""
    content = json.dumps(data, indent=2, default=_json_default, allow_nan=False)
    with path.open("x", encoding="utf-8") as file:
        file.write(content + "\n")


def initialize_output(
    cfg: SimulationConfig | EstimationConfig,
    *,
    ground_truth: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    label_names: dict[str, int] | None = None,
    dataset_metadata: dict | None = None,
) -> Path:
    """
    Save shared arrays and configuration once, in a new or empty directory.

    ``ground_truth`` is the processed clean target returned by dataset preparation,
    in the same scale and preprocessing space as the estimator inputs.
    """
    if cfg.io.base_results_path is None:
        raise ValueError("A results directory is required")
    root = Path(cfg.io.base_results_path)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"Results directory must be empty: {root}")

    config = asdict(cfg)
    config.pop("estimators")
    is_simulation = hasattr(cfg, "snrs")
    config.update(
        format_version=2 if is_simulation else 3,
        experiment_type="simulation" if is_simulation else "dataset",
        estimator_names=["mean", *[method.name for method in cfg.estimators]],
    )
    if label_names is not None:
        config["label_names"] = label_names
    if dataset_metadata is not None:
        config["dataset_metadata"] = dataset_metadata
    save_json(root / "config.json", config)
    if ground_truth is not None:
        np.save(root / "ground_truth.npy", ground_truth, allow_pickle=False)
    if mask is not None:
        np.save(root / "mask.npy", mask, allow_pickle=False)
    if labels is not None and getattr(cfg.io, "save_labels", True):
        np.save(root / "labels.npy", labels, allow_pickle=False)
    mean_directory = root / "estimators" / "mean"
    mean_directory.mkdir(parents=True)
    save_json(
        mean_directory / "config.json", {"name": "mean", "type": "mean", "params": {}}
    )
    for method in cfg.estimators:
        directory = root / "estimators" / method.name
        directory.mkdir(parents=True)
        save_json(directory / "config.json", method.to_method_cfg())
    return root


def save_estimator_result(
    directory: Path,
    result: EstimatorResult,
    io_cfg: IOConfig | EstimationIOConfig,
    *,
    dataset_seed: int | None = None,
) -> None:
    """
    Save one result directly; write ``run.json`` last to mark completion.

    Missing weight spaces are omitted. Array shapes and dtypes are preserved.
    On failure the directory may contain partial results, without a valid
    completion file. Existing directories are never overwritten.
    """
    directory.mkdir(parents=True, exist_ok=False)
    if io_cfg.save_estimates:
        if result.real_estimate is not None:
            np.save(
                directory / "estimate.npy", result.real_estimate, allow_pickle=False
            )
        fourier_estimate = getattr(result, "fourier_estimate", None)
        if fourier_estimate is not None:
            np.save(
                directory / "fourier_estimate.npy", fourier_estimate, allow_pickle=False
            )
    if io_cfg.save_weights:
        for space, weights in result.weights.items():
            if weights is not None:
                np.save(directory / f"weights_{space}.npy", weights, allow_pickle=False)
    save_json(directory / "run.json", {"dataset_seed": dataset_seed})
