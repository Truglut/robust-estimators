"""Read simulation and stored-dataset results written by output.py.

This module is intentionally standalone: it depends only on the Python standard
library and NumPy, and does not import any project modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import TypeAlias

import numpy as np

_SNR_DIR_RE = re.compile(r"^snr_(.+)$")
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")
_WEIGHT_FILE_RE = re.compile(r"^weights_(.+)\.npy$")


@dataclass(frozen=True)
class SavedEstimatorResult:
    """Result saved for one estimator, SNR and simulation run."""

    estimate: np.ndarray | None
    weights: dict[str, np.ndarray]
    dataset_seed: int | None
    fourier_estimate: np.ndarray | None = None


@dataclass(frozen=True)
class SimulationParameters:
    """Shared simulation arrays saved at the experiment root."""

    ground_truth: np.ndarray
    labels: np.ndarray | None
    mask: np.ndarray | None


@dataclass(frozen=True)
class ExperimentParameters:
    """Optional shared arrays, also valid for experiments on unlabeled data."""

    ground_truth: np.ndarray | None
    labels: np.ndarray | None
    mask: np.ndarray | None


EstimatorResults: TypeAlias = dict[float, dict[int, SavedEstimatorResult]]
AllEstimatorResults: TypeAlias = dict[str, EstimatorResults]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _parse_snr_directory(path: Path) -> float:
    match = _SNR_DIR_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid SNR directory name: {path.name!r}")

    try:
        return float(match.group(1))
    except ValueError as exc:
        raise ValueError(f"Could not parse SNR from directory: {path.name!r}") from exc


def _parse_run_directory(path: Path) -> int:
    match = _RUN_DIR_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid run directory name: {path.name!r}")
    return int(match.group(1))


def _read_saved_run(run_directory: Path) -> SavedEstimatorResult:
    """Read one completed run directory."""
    manifest_path = run_directory / "run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Run directory has no completion manifest: {manifest_path}"
        )

    manifest = _read_json(manifest_path)

    estimate_path = run_directory / "estimate.npy"
    estimate = (
        np.load(estimate_path, allow_pickle=False) if estimate_path.is_file() else None
    )

    weights: dict[str, np.ndarray] = {}
    for path in sorted(run_directory.glob("weights_*.npy")):
        match = _WEIGHT_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        space = match.group(1)
        weights[space] = np.load(path, allow_pickle=False)

    dataset_seed = manifest.get("dataset_seed")
    if dataset_seed is not None:
        dataset_seed = int(dataset_seed)

    return SavedEstimatorResult(
        estimate=estimate,
        weights=weights,
        dataset_seed=dataset_seed,
        fourier_estimate=(
            np.load(run_directory / "fourier_estimate.npy", allow_pickle=False)
            if (run_directory / "fourier_estimate.npy").is_file()
            else None
        ),
    )


def read_estimator_results(
    base_directory: Path,
    estimator_name: str | Path,
) -> EstimatorResults:
    """Read all completed results for one estimator.

    Parameters
    ----------
    base_directory : Path
        Root simulation-results directory containing the ``estimators`` folder.
    estimator_name : str | Path
        Estimator identifier, i.e. the directory name below ``estimators``.

    Returns
    -------
    dict[float, dict[int, SavedEstimatorResult]]
        Results indexed first by SNR and then by run index.

    Notes
    -----
    SNR and run index are recovered from the directory names (``snr_*`` and
    ``run_*``). ``run.json`` is used only as a completion marker and to recover
    metadata actually written there, such as ``dataset_seed``.

    Run directories without ``run.json`` are ignored because the writer creates
    that file last, after all requested arrays have been saved.
    """
    base_directory = Path(base_directory)
    estimator_name = Path(estimator_name)

    if estimator_name.is_absolute():
        estimator_directory = estimator_name
    else:
        estimator_directory = base_directory / "estimators" / estimator_name

    if not estimator_directory.is_dir():
        raise FileNotFoundError(
            f"Estimator results directory does not exist: {estimator_directory}"
        )

    results: EstimatorResults = {}

    if any(
        path.is_dir() and _RUN_DIR_RE.fullmatch(path.name)
        for path in estimator_directory.iterdir()
    ):
        raise ValueError(
            "Stored-dataset results require read_dataset_estimator_results()"
        )

    snr_directories = sorted(
        (
            path
            for path in estimator_directory.iterdir()
            if path.is_dir() and _SNR_DIR_RE.fullmatch(path.name)
        ),
        key=_parse_snr_directory,
    )

    for snr_directory in snr_directories:
        snr = _parse_snr_directory(snr_directory)
        runs: dict[int, SavedEstimatorResult] = {}

        run_directories = sorted(
            (
                path
                for path in snr_directory.iterdir()
                if path.is_dir() and _RUN_DIR_RE.fullmatch(path.name)
            ),
            key=_parse_run_directory,
        )

        for run_directory in run_directories:
            # run.json is written last by the output code and therefore acts as
            # the completion marker for a successfully saved run.
            if not (run_directory / "run.json").is_file():
                continue

            run_index = _parse_run_directory(run_directory)
            runs[run_index] = _read_saved_run(run_directory)

        if runs:
            results[snr] = runs

    return results


def read_all_estimator_results(base_directory: Path) -> AllEstimatorResults:
    """Read all estimators saved below ``base_directory / 'estimators'``.

    Returns
    -------
    dict[str, dict[float, dict[int, SavedEstimatorResult]]]
        Results indexed by estimator name, SNR and run index.
    """
    base_directory = Path(base_directory)
    estimators_directory = base_directory / "estimators"

    if not estimators_directory.is_dir():
        raise FileNotFoundError(
            f"Estimators directory does not exist: {estimators_directory}"
        )

    results: AllEstimatorResults = {}

    for estimator_directory in sorted(
        (path for path in estimators_directory.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        results[estimator_directory.name] = read_estimator_results(
            base_directory,
            estimator_directory.name,
        )

    return results


def read_simulation_parameters(base_directory: Path) -> SimulationParameters:
    """Read the shared ground truth, labels and mask for a simulation.

    ``ground_truth.npy`` is required because the output writer always saves it.
    ``labels.npy`` and ``mask.npy`` are optional because their creation depends
    on the simulation configuration.
    """
    base_directory = Path(base_directory)

    ground_truth_path = base_directory / "ground_truth.npy"
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file does not exist: {ground_truth_path}"
        )

    ground_truth = np.load(ground_truth_path, allow_pickle=False)

    labels_path = base_directory / "labels.npy"
    labels = np.load(labels_path, allow_pickle=False) if labels_path.is_file() else None

    mask_path = base_directory / "mask.npy"
    mask = np.load(mask_path, allow_pickle=False) if mask_path.is_file() else None

    return SimulationParameters(
        ground_truth=ground_truth,
        labels=labels,
        mask=mask,
    )


def read_simulation_config(base_directory: Path) -> dict:
    """Read the root ``config.json`` without interpreting project-specific fields."""
    config_path = Path(base_directory) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Simulation config does not exist: {config_path}")
    return _read_json(config_path)


def read_experiment_config(base_directory: Path) -> dict:
    """Read the configuration of either kind of experiment."""
    return _read_json(Path(base_directory) / "config.json")


def read_experiment_parameters(base_directory: Path) -> ExperimentParameters:
    """Read available shared arrays without requiring ground truth or labels."""
    root = Path(base_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {root}")

    def optional_array(name):
        path = root / name
        return np.load(path, allow_pickle=False) if path.is_file() else None

    return ExperimentParameters(
        ground_truth=optional_array("ground_truth.npy"),
        labels=optional_array("labels.npy"),
        mask=optional_array("mask.npy"),
    )


def read_dataset_estimator_results(
    base_directory: Path,
    estimator_name: str | Path,
) -> dict[int, SavedEstimatorResult]:
    """Read stored-dataset results indexed by run, without an SNR dimension."""
    name = Path(estimator_name)
    directory = (
        name if name.is_absolute() else Path(base_directory) / "estimators" / name
    )
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Estimator results directory does not exist: {directory}"
        )
    if any(
        path.is_dir() and _SNR_DIR_RE.fullmatch(path.name)
        for path in directory.iterdir()
    ):
        raise ValueError("Simulation results require read_estimator_results()")
    runs = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_dir() and _RUN_DIR_RE.fullmatch(path.name)
        ),
        key=_parse_run_directory,
    )
    return {
        _parse_run_directory(path): _read_saved_run(path)
        for path in runs
        if (path / "run.json").is_file()
    }


def read_all_dataset_estimator_results(
    base_directory: Path,
) -> dict[str, dict[int, SavedEstimatorResult]]:
    """Read all estimators for an experiment on a stored dataset."""
    directory = Path(base_directory) / "estimators"
    if not directory.is_dir():
        raise FileNotFoundError(f"Estimators directory does not exist: {directory}")
    return {
        path.name: read_dataset_estimator_results(base_directory, path.name)
        for path in sorted(directory.iterdir())
        if path.is_dir()
    }


def read_estimator_config(base_directory: Path, estimator_name: str | Path) -> dict:
    """Read one estimator's saved ``config.json`` as a plain dictionary."""
    estimator_name = Path(estimator_name)
    if estimator_name.is_absolute():
        config_path = estimator_name / "config.json"
    else:
        config_path = (
            Path(base_directory) / "estimators" / estimator_name / "config.json"
        )

    if not config_path.is_file():
        raise FileNotFoundError(f"Estimator config does not exist: {config_path}")

    return _read_json(config_path)
