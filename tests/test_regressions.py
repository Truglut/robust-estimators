"""Contract regressions on the supplied production code; see TESTING.md.

These tests are intentionally not weakened to match faulty behavior. Strict
xfails will flag a fix as XPASS, requiring removal of the corresponding marker.
Run with --runxfail to see the original failures.
"""

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from simple_mestimators.domain import ImageBatch
from simple_mestimators.estimators.distances import mean_l2_norm
from simple_mestimators.estimators.fourier_irls import JointIRLSFourier
from simple_mestimators.estimators.gmm import RecursiveGMMEstimator
from simple_mestimators.estimators.irls import IRLSSolver
from scripts.experiments.cli import build_simulation_parser, parse_simulation_cfg
from conftest import unit_weights


@pytest.mark.parametrize("omit", [False, True], ids=["null-path", "missing-path"])
def test_cli_accepts_no_misclassified_file_when_none_requested(tmp_path, omit):
    data = {"reference_image_path": "ref.mrc", "misclassified_path": None}
    if omit:
        data.pop("misclassified_path")
    document = {
        "data": data,
        "generation": {"n_copies": 3},
        "experiment": {
            "methods": [
                {
                    "name": "real",
                    "type": "m_estimator",
                    "params": {
                        "weight_function": "norm_smooth",
                        "weight_params": {"delta": 2},
                        "solver_params": {"max_iter": 1},
                    },
                }
            ]
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    args = build_simulation_parser().parse_args(
        ["--config", str(path), "--results", str(tmp_path / "out")]
    )
    cfg = parse_simulation_cfg(args)
    assert cfg.io.misclassified_path is None


@pytest.mark.parametrize("ctf_shape", ["frequency-grid", "per-image"])
def test_masked_fourier_accepts_broadcastable_ctf(ctf_shape):
    images = np.random.default_rng(1).normal(size=(4, 3, 5))
    batch = ImageBatch(images)
    grid = batch.ensure_fourier()
    ctf = (
        np.full(batch.rfft_shape, 0.7)
        if ctf_shape == "frequency-grid"
        else np.array([0.5, -0.8, 1.2, 1.0])[:, None, None]
    )
    mask = np.zeros(batch.rfft_shape, bool)
    mask[0, 0] = mask[1, 1] = True
    batch.ctf = ctf
    estimator = JointIRLSFourier(IRLSSolver(unit_weights, 1), mask=mask)
    result = estimator.fit(batch)
    full_ctf = np.broadcast_to(ctf, grid.shape)
    expected_fourier = (full_ctf * grid).sum(0) / (full_ctf**2).sum(0)
    assert_allclose(
        result.real_estimate,
        np.fft.irfft2(expected_fourier, s=batch.real_shape, norm=batch.norm),
    )
