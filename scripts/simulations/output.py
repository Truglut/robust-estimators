"""Write minimal simulation results in estimator-first directories."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .dataset_builder import LABELS

if TYPE_CHECKING:
    from simple_mestimators.domain import EstimatorResult
    from .configs import IOConfig, SimulationConfig


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
    cfg: SimulationConfig,
    *,
    ground_truth: np.ndarray,
    mask: np.ndarray | None,
    labels: np.ndarray,
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
    config.update(format_version=2, label_names=LABELS,
                  estimator_names=[method.name for method in cfg.estimators])
    save_json(root / "config.json", config)
    np.save(root / "ground_truth.npy", ground_truth, allow_pickle=False)
    if mask is not None:
        np.save(root / "mask.npy", mask, allow_pickle=False)
    if cfg.io.save_labels:
        np.save(root / "labels.npy", labels, allow_pickle=False)
    for method in cfg.estimators:
        directory = root / "estimators" / method.name
        directory.mkdir(parents=True)
        save_json(directory / "config.json", method.to_method_cfg())
    return root


def save_estimator_result(
    directory: Path,
    result: EstimatorResult,
    io_cfg: IOConfig,
    *,
    dataset_seed: int,
) -> None:
    """
    Save one result directly; write ``run.json`` last to mark completion.

    Missing weight spaces are omitted. Array shapes and dtypes are preserved.
    On failure the directory may contain partial results, without a valid
    completion file. Existing directories are never overwritten.
    """
    directory.mkdir(parents=True, exist_ok=False)
    if io_cfg.save_estimates:
        np.save(directory / "estimate.npy", result.real_estimate, allow_pickle=False)
    if io_cfg.save_weights:
        for space, weights in result.weights.items():
            if weights is not None:
                np.save(directory / f"weights_{space}.npy", weights, allow_pickle=False)
    save_json(directory / "run.json", {"dataset_seed": dataset_seed})
