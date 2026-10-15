"""Scientific settings survive YAML, model recreation, and result serialization."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from simple_mestimators.domain import ImageBatch
from simple_mestimators.estimators.construction import build_estimator
from simple_mestimators.estimators.irls import IRLSSolver
from simple_mestimators.estimators.weights import tagare_weights
from simple_mestimators.estimators.distances import tagare_distance
from scripts.experiments.cli import build_simulation_parser, parse_simulation_cfg
from scripts.experiments.run_simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]


def document():
    return yaml.safe_load((ROOT / "configs/paper_simulations.yaml").read_text())


def test_yaml_science_and_cli_paths_are_resolved_without_silent_overrides(tmp_path):
    cfg = document()
    cfg["data"] = {"reference_image_path": "images/ref.mrc", "misclassified_path": "images/bad.mrc"}
    cfg["simulation"] = {"snrs": [.03], "n_runs": 4}
    cfg["experiment"].update(seed=17, initialization_strategy="median")
    cfg["generation"].update(standardize_reference=False, max_rotation_inliers=None)
    cfg["preprocessing"] = {"standardize": "after", "mask_radius": 3.0}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    parser = build_simulation_parser()
    args = ["--config", str(path), "--results", str(tmp_path / "results")]
    parsed = parse_simulation_cfg(parser.parse_args(args))
    assert parsed.io.reference_path == tmp_path / "images/ref.mrc"
    assert parsed.io.misclassified_path == tmp_path / "images/bad.mrc"
    assert (parsed.snrs, parsed.n_runs, parsed.seed) == ([.03], 4, 17)
    assert parsed.initialization_strategy == "median"
    assert parsed.preprocessing.standardize == "after"
    assert not parsed.dataset.standardize_reference
    overridden = parse_simulation_cfg(parser.parse_args(args + ["--reference-image-path", "elsewhere.mrc"]))
    assert overridden.io.reference_path == Path.cwd() / "elsewhere.mrc"
    assert overridden.snrs == parsed.snrs
    for option in (["--snr", ".8"], ["--seed", "12"], ["--n-runs", "2"]):
        with pytest.raises(SystemExit):
            parser.parse_args(args + option)
    cfg["simulation"]["n_run"] = 3
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Unknown simulation"):
        parse_simulation_cfg(parser.parse_args(args))


def test_diagnostics_and_resolved_auto_beta_survive_a_complete_simulation(tmp_path, write_mrc, reference):
    cfg = document()
    cfg["data"]["reference_image_path"] = str(write_mrc("ref.mrc", reference))
    cfg["generation"].update(n_copies=60, n_misclassified=0)
    cfg["simulation"] = {"snrs": [.2], "n_runs": 1}
    # Force a deterministic rejection, rather than depending on a fitted distribution.
    cfg["experiment"]["methods"][-1]["params"]["min_component_separation"] = 1.e9
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "results"
    config = parse_simulation_cfg(build_simulation_parser().parse_args([
        "--config", str(path), "--results", str(out)]))
    run_simulation(config)
    saved = json.loads((out / "config.json").read_text())
    assert saved["preprocessing"]["mask_radius"] == 2
    assert len(saved["provenance"]["source_sha256"]) == 64
    assert saved["provenance"]["packages"]["numpy"]
    runs = {m["name"]: json.loads(next((out / "estimators" / m["name"]).rglob("run.json")).read_text())["diagnostics"]
            for m in cfg["experiment"]["methods"]}
    assert runs["GMM-reweighted CC"]["fallback"]
    assert runs["GMM-reweighted CC"]["stop_reason"] == "fallback"
    assert "insufficient_separation" in runs["GMM-reweighted CC"]["fallback_reasons"]
    cc = runs["CC weights"]["effective_estimator"]["weight_function"]["params"]
    gmm = runs["GMM-reweighted CC"]["effective_estimator"]["distance_function"]["params"]
    admm = runs["ADMM"]["effective_estimator"]["irls_real"]["weight_function"]["params"]
    assert cc == gmm == admm
    assert cc["beta"] > 0
    assert runs["ADMM"]["n_iter"] > 0
    assert runs["ADMM"]["primal_residual"] is not None
    assert runs["Fourier IRLS"]["stop_reason"] in {"tolerance", "max_iter"}


def test_irls_diagnostics_reset_when_iteration_budget_changes():
    images = np.arange(24.).reshape(4, 2, 3)
    solver = IRLSSolver(lambda x, r, std: np.ones((len(x), 1, 1)), max_iter=2)
    solver.solve(images)
    assert solver.converged
    solver.tol = 0
    solver.solve(images, max_iter_override=1)
    assert not solver.converged
    assert solver.n_its == 1
    assert solver.diagnostics["stop_reason"] == "max_iter"
    solver.solve(images, max_iter_override=0)
    assert solver.diagnostics["n_iter"] == 0
    assert solver.diagnostics["stop_reason"] == "zero_iterations"


def test_gmm_options_survive_model_reset_and_admm_types_are_validated():
    batch = ImageBatch(np.random.default_rng(2).normal(size=(60, 4, 5)))
    cfg = document()["experiment"]["methods"][-1]
    cfg["params"].update(reg_covar=.003, n_init=2, init_params="random")
    gmm = build_estimator(cfg, batch)
    gmm.fit(batch)
    assert gmm.model.reg_covar == .003
    assert gmm.model.n_init == 2
    assert gmm.model.init_params == "random"
    admm = document()["experiment"]["methods"][2]
    admm["params"]["real_estimator"]["type"] = "recursive_gmm"
    with pytest.raises(ValueError, match="real_estimator.type"):
        build_estimator(admm, batch)


def test_tagare_weights_and_distance_share_small_norm_stabilization():
    images = np.array([[[1.e-5, 2.e-5]], [[2.e-5, -1.e-5]]])
    reference = np.array([[2.e-5, 1.e-5]])
    assert_allclose(tagare_distance(images, reference), -tagare_weights(images, reference).ravel())


def test_bash_preserves_paths_with_spaces_and_does_not_override_science(tmp_path):
    data = tmp_path / "input images" / "ribosome"
    data.mkdir(parents=True)
    for angle in (0, 10, 50):
        (data / f"image_tilt_{angle}.mrc").touch()
    fake_python = tmp_path / "fake python"
    log = tmp_path / "commands.jsonl"
    fake_python.write_text("#!/usr/bin/env python3\nimport sys,json\n"
                           f"with open({str(log)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    fake_python.chmod(0o755)
    script = ROOT / "run_tilt_simulations.sh"
    subprocess.run(["bash", str(script), "--config", str(ROOT / "configs/paper_simulations.yaml"),
                    "--data-dir", str(data.parent), "--results-dir", str(tmp_path / "output images"),
                    "--structure", "ribosome", "--tilt", "10", "--tilt", "50",
                    "--python", str(fake_python)], check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    assert all("--snr" not in call and "--n-runs" not in call for call in calls)
    assert calls[0][calls[0].index("--reference-image-path") + 1] == str(data / "image_tilt_0.mrc")
