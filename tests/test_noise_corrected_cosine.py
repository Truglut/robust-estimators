"""Scientific and simulation integration checks for the experimental estimator."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from simple_mestimators.domain import ImageBatch
from simple_mestimators.estimators.construction import build_estimator
from simple_mestimators.estimators.noise_corrected_cosine import (
    NoiseCorrectedCosineDistance, NoiseCorrectedCosineEstimator,
    estimate_noise_variance,
)
from scripts.experiments.configs import (
    DatasetConfig, EstimatorConfig, IOConfig, PreprocessingConfig, SimulationConfig,
)
from scripts.experiments.run_simulation import (
    PreparedSimulationDataset, effective_config, run_experiment, run_simulation,
)


def test_real_metric_matches_dense_mask_center_covariance_and_reference_correction():
    rng = np.random.default_rng(17)
    images = rng.normal(size=(9, 5, 7))
    original = images.copy()
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 1:6] = True
    noise = np.linspace(.01, .09, 9)
    a = np.arange(1, 10, dtype=float)
    a /= a.sum()
    reference = np.tensordot(a, images, axes=(0, 0))
    metric = NoiseCorrectedCosineDistance(images, mask=mask, noise_variance=noise,
                                        min_signal_fraction=0, clip=False)
    m = mask.reshape(-1).astype(float)
    c = np.diag(m) - np.outer(m, m) / m.sum()
    trace = np.trace(c @ c.T)
    x = images.reshape(9, -1) @ c.T
    r = c @ reference.reshape(-1)
    expected = 1 - (x @ r - trace * a * noise) / np.sqrt(
        ((x*x).sum(axis=1) - trace * noise)
        * (r @ r - trace * np.dot(a*a, noise)))
    assert_allclose(metric.distances(reference, reference_weights=a), expected)
    assert metric.trace == mask.sum() - 1
    assert_array_equal(images, original)
    # An independent reference needs no numerator correction.
    expected = 1 - (x @ r) / np.sqrt(
        ((x*x).sum(axis=1) - trace * noise) * (r @ r - trace * .02))
    assert_allclose(metric.distances(reference, reference_noise_variance=.02), expected)


def test_mad_estimates_noise_in_smooth_signal_and_current_scaled_units():
    rng = np.random.default_rng(7)
    y, x = np.mgrid[:48, :53]
    plane = 2 + .1*x - .2*y
    images = plane + rng.normal(0, .5, (24, 48, 53))
    support = np.ones((48, 53), dtype=bool)
    variance = estimate_noise_variance(images, support)
    assert_allclose(variance.mean(), .25, rtol=.06)
    assert_allclose(estimate_noise_variance(3*images + 4, support), 9*variance)


def test_cosine_iterations_match_manual_reference_weight_updates_and_reset():
    rng = np.random.default_rng(29)
    images = rng.normal(size=(12, 9, 11)) + np.linspace(0, 2, 99).reshape(9, 11)
    batch = ImageBatch(images)
    params = {"noise_variance": .1, "min_signal_fraction": 0}
    estimator = NoiseCorrectedCosineEstimator(weighting="cosine", max_iter=2, tol=0,
                                             metric_params=params, weight_power=3)
    metric = NoiseCorrectedCosineDistance(images, **params)
    ref = images.mean(axis=0)
    a = np.full(len(images), 1/len(images))
    for _ in range(2):
        d = metric.distances(ref, reference_weights=a)
        w = np.clip(1-d, 0, 1)**3
        a = w / w.sum()
        ref = np.tensordot(a, images, axes=(0, 0))
    result = estimator.fit(batch)
    assert_allclose(result.real_estimate, ref)
    assert_allclose(result.weights["real"].reshape(-1), w)
    assert_allclose(estimator.reference_weights, a)
    assert_allclose(estimator.metric.noise_variance, .1)
    assert_allclose(estimator.fit(batch).real_estimate, result.real_estimate)


def test_clipping_unresolved_and_all_zero_weights_are_explicit():
    image = np.arange(35, dtype=float).reshape(5, 7)
    images = image[None]
    raw = NoiseCorrectedCosineDistance(images, noise_variance=1., min_signal_fraction=0)
    assert raw.distances(image)[0] < 0  # Independent clean reference, noisy-image model.
    assert NoiseCorrectedCosineDistance(images, noise_variance=1., clip=True).distances(image)[0] == 0
    # Comparing an image with itself requires numerator debiasing as well.
    assert_allclose(raw.distances(image, reference_weights=np.ones(1)), 0, atol=1e-14)
    est = NoiseCorrectedCosineEstimator(weighting="cosine", max_iter=3,
                                       metric_params={"noise_variance": 1e6})
    with pytest.warns(RuntimeWarning):
        result = est.fit(ImageBatch(np.repeat(images, 3, axis=0)))
    assert_allclose(result.real_estimate, image)
    assert_array_equal(result.weights["real"], np.ones((3, 1, 1)))
    assert est.n_its == 1


@pytest.mark.parametrize("strategy", ["mean", "mean-good", "ground-truth"])
def test_simulation_passes_initial_reference_coefficients(strategy):
    rng = np.random.default_rng(3)
    images = rng.normal(size=(8, 7, 9))
    labels = np.array([0]*6 + [3]*2)
    mask = np.ones((7, 9), dtype=bool)
    dataset = PreparedSimulationDataset(images, labels, images[0], mask)
    cfg = EstimatorConfig("corrected", "noise_corrected_cosine", {
        "weighting": "cosine", "max_iter": 1,
        "metric_params": {"noise_variance": .01, "min_signal_fraction": 0}})
    actual = dict(run_experiment(dataset, [cfg], strategy))[cfg.name]
    if strategy == "ground-truth":
        a, ref = None, images[0]
    else:
        a = np.ones(8) if strategy == "mean" else (labels == 0).astype(float)
        a /= a.sum()
        ref = np.tensordot(a, images, axes=(0, 0))
    expected = build_estimator(cfg.to_method_cfg(), ImageBatch(images, mask=mask)).fit(
        ImageBatch(images, mask=mask), reference=ref, initial_reference_weights=a)
    assert_allclose(actual.real_estimate, expected.real_estimate)
    with pytest.raises(ValueError, match="median"):
        list(run_experiment(dataset, [cfg], "median"))


@pytest.mark.parametrize("filtered", [False, True])
def test_gmm_simulation_saves_and_replays_with_seed_and_unchanged_baseline(write_mrc, tmp_path, filtered):
    y, x = np.mgrid[:17, :19]
    reference = np.exp(-((x-6)**2 + (y-8)**2)/10)
    other = 3*np.exp(-((x-13)**2 + (y-8)**2)/10)
    baseline = EstimatorConfig("baseline", "m_estimator", {
        "weight_function": "norm_smooth", "weight_params": {"delta": 2},
        "solver_params": {"max_iter": 2}})
    new = EstimatorConfig("corrected_gmm", "noise_corrected_cosine", {
        "weighting": "gmm", "max_iter": 2,
        "metric_params": {"min_signal_fraction": 0},
        "gmm_params": {"gmm_max_iter": 100, "check_degenerate_model": False}})
    if filtered:
        new.params["metric_params"].update(filter_sigma=1., max_frequency=.3)
    cfg = SimulationConfig(snrs=[1.], n_runs=1, initialization_strategy="mean", seed=43,
        dataset=DatasetConfig(n_copies=60, n_misclassified=20),
        io=IOConfig(write_mrc("ref.mrc", reference), write_mrc("other.mrc", other), tmp_path/"all"),
        preprocessing=PreprocessingConfig(mask_radius=7, standardize="after"),
        estimators=[baseline, new])
    original = deepcopy(cfg)
    effective = effective_config(cfg)
    assert cfg == original

    assert effective.estimators[1].params["gmm_params"]["random_state"] == cfg.seed
    root = run_simulation(cfg)
    replay = run_simulation(replace(cfg, io=replace(cfg.io, base_results_path=tmp_path/"replay"),
                                    estimators=[new]))
    alone = run_simulation(replace(cfg, io=replace(cfg.io, base_results_path=tmp_path/"baseline"),
                                   estimators=[baseline]))
    for name, other_root in [("corrected_gmm", replay), ("baseline", alone)]:
        relative = f"estimators/{name}/snr_1.0/run_000"
        for filename in ["estimate.npy", "weights_real.npy"]:
            actual = np.load(root/relative/filename)
            assert np.isfinite(actual).all()
            assert_array_equal(actual, np.load(other_root/relative/filename))
    saved = json.loads((root/"estimators/corrected_gmm/config.json").read_text())
    assert saved == effective.estimators[1].to_method_cfg()
    assert cfg == original

@pytest.mark.parametrize("width", [7, 8])
def test_filtered_rfft_matches_full_fft_and_trace_of_dense_operator(width):
    rng = np.random.default_rng(41)
    h = 6
    mask = np.ones((h, width), dtype=bool)
    mask[0, :2] = False
    mask[-1, -2:] = False
    images = rng.normal(size=(11, h, width))
    a = np.arange(1, 12, dtype=float)
    a /= a.sum()
    ref = np.tensordot(a, images, axes=(0, 0))
    noise = np.linspace(.001, .01, len(images))
    sigma = .7
    lower, upper = .1, .45
    metric = NoiseCorrectedCosineDistance(images, mask=mask, noise_variance=noise,
        min_signal_fraction=0, filter_sigma=sigma, min_frequency=lower, max_frequency=upper)
    radius = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(width)[None, :])
    omega = ((radius >= lower) & (radius <= upper)) * np.exp(-4*np.pi**2*sigma**2*radius**2)
    def fft_center(batch):
        centered = np.zeros_like(batch)
        values = batch[:, mask]
        centered[:, mask] = values - values.mean(axis=1, keepdims=True)
        return np.fft.fft2(centered, norm="ortho")
    xfft, rfft = fft_center(images), fft_center(ref[None])[0]
    # Apply the full preprocessing to every spatial basis vector; summing the
    # output energies gives ||L||_F^2 = trace(LL*) without the profile formula.
    basis_fft = fft_center(np.eye(h*width).reshape(h*width, h, width))
    trace = np.sum(np.abs(basis_fft)**2 * omega)
    ex = np.sum(np.abs(xfft)**2 * omega, axis=(1, 2))
    er = np.sum(np.abs(rfft)**2 * omega)
    cross = np.sum((xfft*rfft.conj()).real * omega, axis=(1, 2))
    expected = 1 - (cross-trace*a*noise) / np.sqrt(
        (ex-trace*noise) * (er-trace*np.dot(a*a, noise)))
    assert_allclose(metric.trace, trace, atol=1e-14)
    assert_allclose(metric.image_energy, ex)
    assert_allclose(metric.distances(ref, reference_weights=a), expected)
    # A band covering all frequencies reproduces real-space centering exactly.
    full_band = NoiseCorrectedCosineDistance(images, mask=mask, noise_variance=noise,
                                            min_signal_fraction=0, max_frequency=1.)
    real = NoiseCorrectedCosineDistance(images, mask=mask, noise_variance=noise,
                                       min_signal_fraction=0)
    assert_allclose(full_band.trace, mask.sum()-1)
    assert_allclose(full_band.distances(ref, reference_weights=a),
                    real.distances(ref, reference_weights=a))


def test_filter_changes_only_scoring_not_resolution_of_output_average():
    rng = np.random.default_rng(101)
    images = rng.normal(size=(18, 15, 17))
    estimator = NoiseCorrectedCosineEstimator(weighting="cosine", max_iter=1,
        metric_params={"noise_variance": 0., "filter_sigma": 1.3, "max_frequency": .3})
    result = estimator.fit(ImageBatch(images))
    a = result.weights["real"].reshape(-1)
    a /= a.sum()
    assert_allclose(result.real_estimate, np.tensordot(a, images, axes=(0, 0)))
    assert estimator.metric.features.shape[1] < 15*17
    assert estimator.metric.noise_profile is not None
