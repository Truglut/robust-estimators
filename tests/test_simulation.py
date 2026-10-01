from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
import json

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose, assert_array_equal

from simple_mestimators.domain import EstimatorResult
from scripts.simulations.cli import build_simulation_parser, parse_simulation_cfg
from scripts.simulations.configs import (
    DatasetConfig, EstimatorConfig, IOConfig, PreprocessingConfig, SimulationConfig,
)
from scripts.simulations.output import initialize_output, save_estimator_result, snr_directory
from scripts.simulations.run_simulation import (
    effective_config, prepare_simulation_dataset, run_experiment, run_simulation,
)


def methods():
    real = {"weight_function": "norm_smooth", "weight_params": {"delta": 2.}, "solver_params": {"max_iter": 2}}
    fourier = {"weight_function": "smooth", "weight_params": {"delta": 2.}, "solver_params": {"max_iter": 2}}
    return [
        EstimatorConfig("real", "m_estimator", deepcopy(real)),
        EstimatorConfig("fourier", "joint_fourier", deepcopy(fourier)),
        EstimatorConfig("masked", "masked_fourier", {**deepcopy(real), "lowpass_cutoff": .4}),
        EstimatorConfig("gmm", "recursive_gmm", {"distance_function": "l2", "max_iter": 2, "gmm_max_iter": 100}),
        EstimatorConfig("fgmm", "fourier_gmm", {"max_iter": 2, "gmm_max_iter": 100}),
        EstimatorConfig("admm", "admm", {
            "real_estimator": deepcopy(real), "fourier_estimator": deepcopy(fourier),
            "solver_params": {"max_iter": 2, "initial_mu": 1., "fourier_multiplier": 1.}}),
    ]


def make_config(path, results):
    return SimulationConfig(snrs=[.2, .3], n_runs=2, initialization_strategy="mean", seed=23,
                             dataset=DatasetConfig(n_copies=50, n_noise=10),
                             io=IOConfig(path, base_results_path=results),
                             preprocessing=PreprocessingConfig(mask_radius=2.), estimators=methods())


def test_effective_config_only_resolves_missing_gmm_random_states(tmp_path):
    cfg = make_config(tmp_path / "ref.mrc", tmp_path / "out")
    cfg.estimators[4].params["random_state"] = 91
    original = deepcopy(cfg)
    effective = effective_config(cfg)
    assert cfg == original
    assert effective.estimators[3].params["random_state"] == cfg.seed
    assert effective.estimators[4].params["random_state"] == 91
    effective.estimators[0].params["solver_params"]["max_iter"] = 99
    assert cfg == original


def test_all_methods_saved_outputs_replay_same_dataset_and_selection_is_invariant(reference, write_mrc, tmp_path):
    path = write_mrc("reference.mrc", reference)
    cfg = make_config(path, tmp_path / "all")
    original = deepcopy(cfg)
    root = run_simulation(cfg)
    assert cfg == original
    config = json.loads((root / "config.json").read_text())
    assert config["seed"] == cfg.seed
    assert config["snrs"] == cfg.snrs
    assert config["n_runs"] == cfg.n_runs
    assert config["dataset"] == asdict(cfg.dataset)
    assert config["io"]["reference_path"] == str(path)
    effective = effective_config(cfg)
    mask = np.load(root / "mask.npy", allow_pickle=False)
    source = reference.astype(np.float32)
    seeds = set()
    for snr in cfg.snrs:
        for run in range(cfg.n_runs):
            directories = [root / "estimators" / m.name / snr_directory(snr) / f"run_{run:03d}" for m in cfg.estimators]
            metadata = [json.loads((d / "run.json").read_text()) for d in directories]
            assert all(item == metadata[0] for item in metadata)
            seed = metadata[0]["dataset_seed"]
            expected_seed = int(np.random.SeedSequence([cfg.seed, cfg.snrs.index(snr), run]).generate_state(1, dtype=np.uint64)[0])
            assert seed == expected_seed
            seeds.add(seed)
            dataset = prepare_simulation_dataset(ground_truth=source, misclassified_path=None,
                dataset_cfg=cfg.dataset, preprocessing_cfg=cfg.preprocessing, mask=mask,
                snr=snr, rng=np.random.default_rng(seed))
            assert_allclose(np.load(root / "ground_truth.npy"), dataset.ground_truth)
            assert_array_equal(np.load(root / "labels.npy"), dataset.labels)
            original_images = dataset.preprocessed_images.copy()
            replay = dict(run_experiment(dataset, effective.estimators, cfg.initialization_strategy))
            assert_array_equal(dataset.preprocessed_images, original_images)
            for method, directory in zip(cfg.estimators, directories):
                expected = replay[method.name]
                assert_allclose(np.load(directory / "estimate.npy"), expected.real_estimate)
                for space, weights in expected.weights.items():
                    file = directory / f"weights_{space}.npy"
                    if weights is None:
                        assert not file.exists()
                    else:
                        assert_array_equal(np.load(file, allow_pickle=False), weights)
    assert len(seeds) == 4
    for method in effective.estimators:
        saved = json.loads((root / "estimators" / method.name / "config.json").read_text())
        assert saved == method.to_method_cfg()
    # Reverse and subset selection: every overlapping result must be identical.
    selected = replace(cfg, io=replace(cfg.io, base_results_path=tmp_path / "selected"),
                        estimators=list(reversed(cfg.estimators[1::2])))
    subset_root = run_simulation(selected)
    for method in selected.estimators:
        for file in (subset_root / "estimators" / method.name).rglob("*.npy"):
            assert_array_equal(np.load(file), np.load(root / file.relative_to(subset_root)))
    assert_array_equal(np.load(subset_root / "ground_truth.npy"), np.load(root / "ground_truth.npy"))
    assert_array_equal(np.load(subset_root / "labels.npy"), np.load(root / "labels.npy"))
    assert not any(file.suffix in (".mrc", ".mrcs", ".png", ".pdf") or "images" in file.name for file in root.rglob("*"))
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run_simulation(cfg)
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("save_weights,save_estimates,save_labels", list(product([False, True], repeat=3)))
def test_saving_flags_preserve_arrays_and_completion_metadata(tmp_path, save_weights, save_estimates, save_labels):
    cfg = make_config(tmp_path / "reference.mrc", tmp_path / "results")
    cfg.io = replace(cfg.io, save_weights=save_weights, save_estimates=save_estimates, save_labels=save_labels)
    estimate = np.arange(6, dtype=np.float32).reshape(2, 3)
    weights = np.array([1., .2, np.nan], dtype=np.float64)[:, None, None]
    root = initialize_output(cfg, ground_truth=estimate, mask=None, labels=np.array([0, 0, 4], dtype=np.uint8))
    directory = root / "estimators" / "real" / "snr_0.2" / "run_000"
    result = EstimatorResult(estimate, {"real": weights, "fourier": None})
    save_estimator_result(directory, result, cfg.io, dataset_seed=2**63 + 1)
    assert (root / "ground_truth.npy").exists()
    assert (root / "labels.npy").exists() == save_labels
    expected = {"run.json"}
    if save_estimates:
        expected.add("estimate.npy")
        loaded = np.load(directory / "estimate.npy", allow_pickle=False)
        assert loaded.dtype == estimate.dtype
        assert_array_equal(loaded, estimate)
    if save_weights:
        expected.add("weights_real.npy")
        loaded = np.load(directory / "weights_real.npy", allow_pickle=False)
        assert loaded.dtype == weights.dtype
        assert_array_equal(loaded, weights)
    assert {p.name for p in directory.iterdir()} == expected
    assert json.loads((directory / "run.json").read_text()) == {"dataset_seed": 2**63 + 1}
    contents = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        save_estimator_result(directory, result, cfg.io, dataset_seed=0)
    assert contents == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_failed_array_write_never_marks_run_complete_and_cannot_be_overwritten(tmp_path):
    io = IOConfig(tmp_path / "reference.mrc")
    result = EstimatorResult(np.zeros((2, 3)), {"real": np.array([object()], dtype=object)})
    directory = tmp_path / "partial"
    with pytest.raises(ValueError, match="Object arrays"):
        save_estimator_result(directory, result, io, dataset_seed=1)
    assert (directory / "estimate.npy").exists()
    assert not (directory / "run.json").exists()
    with pytest.raises(FileExistsError):
        save_estimator_result(directory, result, io, dataset_seed=2)


def test_results_saved_before_later_estimator_failure(reference, write_mrc, tmp_path):
    cfg = make_config(write_mrc("ref.mrc", reference), tmp_path / "results")
    cfg.estimators = [cfg.estimators[0], EstimatorConfig("broken", "unknown", {})]
    with pytest.raises(ValueError, match="Unknown estimator type"):
        run_simulation(cfg)
    directory = cfg.io.base_results_path / "estimators" / "real" / snr_directory(.2) / "run_000"
    assert (directory / "estimate.npy").exists()
    assert (directory / "weights_real.npy").exists()
    assert (directory / "run.json").exists()
    assert not (cfg.io.base_results_path / "estimators" / "broken" / snr_directory(.2)).exists()


def write_yaml(tmp_path, reference_path, misclassified_path):
    path = tmp_path / "config.yaml"
    data = {"data": {"reference_image_path": str(reference_path), "misclassified_path": misclassified_path},
            "generation": {"n_copies": 3},
            "experiment": {"methods": [methods()[0].to_method_cfg()]}}
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("override", [False, True])
def test_cli_defaults_overrides_and_saved_effective_config(tmp_path, write_mrc, reference, override):
    path = write_mrc("reference.mrc", reference)
    alternative = write_mrc("alternative.mrc", reference * 2)
    yaml_path = write_yaml(tmp_path, path, "unused.mrcs")
    argv = ["--config", str(yaml_path), "--results", str(tmp_path / "results")]
    if override:
        argv += ["--reference-image-path", str(alternative), "--misclassified-path", "overridden.mrcs",
                 "--snr", ".1", ".4", "--seed", "17", "--n-runs", "2",
                 "--standardize", "both", "--no-standardize-reference", "--initialization-strategy", "median",
                 "--no-save-weights", "--no-save-labels", "--no-save-estimates"]
    cfg = parse_simulation_cfg(build_simulation_parser().parse_args(argv))
    assert cfg.snrs == ([.1, .4] if override else [.01])
    assert cfg.seed == (17 if override else 42)
    assert cfg.n_runs == (2 if override else 1)
    assert cfg.io.reference_path == (alternative if override else path)
    assert str(cfg.io.misclassified_path) == ("overridden.mrcs" if override else "unused.mrcs")
    assert cfg.dataset.standardize_reference == (not override)
    assert cfg.preprocessing.standardize == ("both" if override else "none")
    assert cfg.initialization_strategy == ("median" if override else "mean")
    assert cfg.io.save_weights == cfg.io.save_labels == cfg.io.save_estimates == (not override)
    root = run_simulation(cfg)
    saved = json.loads((root / "config.json").read_text())
    expected = asdict(effective_config(cfg))
    expected.pop("estimators")
    # Convert Path fields using the same JSON representation promised to readers.
    expected = json.loads(json.dumps(expected, default=str))
    for key, value in expected.items():
        assert saved[key] == value
    for snr in cfg.snrs:
        for run in range(cfg.n_runs):
            directory = root / "estimators" / "real" / snr_directory(snr) / f"run_{run:03d}"
            assert (directory / "run.json").exists()
            assert (directory / "estimate.npy").exists() == (not override)


def test_close_snrs_do_not_collide():
    assert snr_directory(.01001) != snr_directory(.01002)
