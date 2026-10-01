import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from simple_mestimators.domain import ImageBatch
from simple_mestimators.estimators.distances import mean_l2_norm
from simple_mestimators.estimators.gmm import FourierGMM, RecursiveGMMEstimator


def separated_images():
    return np.concatenate([np.linspace(.1, .7, 48), np.linspace(7., 9., 12)])[:, None, None]


def test_recursive_gmm_favors_inliers_and_resets_random_state_between_fits():
    images = separated_images()
    original = images.copy()
    estimator = RecursiveGMMEstimator(mean_l2_norm, max_iter=3, random_state=19, gmm_max_iter=100)
    result = estimator.fit(ImageBatch(images), reference=np.zeros((1, 1)))
    weights = result.weights["real"].ravel()
    assert weights[:48].mean() > .95
    assert weights[48:].mean() < .05
    assert abs(result.real_estimate.item() - .4) < .1
    assert estimator.model.random_state == 19
    assert result.gmm_diagnostics is not None
    assert_allclose(result.gmm_diagnostics.weights, weights)
    distances = result.gmm_diagnostics.distances.ravel()
    assert (np.diff(weights[np.argsort(distances)]) <= 1e-14).all()
    # Contaminate fitted state through another real fit before replaying.
    estimator.fit(ImageBatch(images * 3), reference=np.zeros((1, 1)))
    replay = estimator.fit(ImageBatch(images), reference=np.zeros((1, 1)))
    independent = RecursiveGMMEstimator(mean_l2_norm, max_iter=3, random_state=19, gmm_max_iter=100)
    fresh = independent.fit(ImageBatch(images), reference=np.zeros((1, 1)))
    assert_array_equal(replay.real_estimate, result.real_estimate)
    assert_array_equal(replay.weights["real"], fresh.weights["real"])
    assert_array_equal(images, original)


@pytest.mark.parametrize("good_variance,bad_variance", [(4., .25), (.25, 4.)],
                         ids=["tail-resurgence", "low-distance-dip"])
def test_peak_correction_removes_low_distance_dips_and_tail_resurgence(good_variance, bad_variance):
    estimator = RecursiveGMMEstimator(mean_l2_norm)
    # Component 1 is good. A narrow bad Gaussian produces tail resurgence;
    # a narrow good Gaussian produces a low-distance dip before the peak.
    model = estimator.model
    model.means_ = np.array([[3.], [0.]])
    model.covariances_ = np.array([[[bad_variance]], [[good_variance]]])
    model.precisions_cholesky_ = 1 / np.sqrt(model.covariances_)
    model.weights_ = np.array([.4, .6])
    distances = np.array([5., -3., 3., 0., 2., 4., 1.])[:, None]
    raw, corrected = estimator._responsibility_weights(model, distances)
    d = distances[:, 0]
    good_pdf = .6 / np.sqrt(2 * np.pi * good_variance) * np.exp(-d**2 / (2 * good_variance))
    bad_pdf = .4 / np.sqrt(2 * np.pi * bad_variance) * np.exp(-(d - 3)**2 / (2 * bad_variance))
    assert estimator.get_good_component_idx() == 1
    assert_allclose(raw, good_pdf / (good_pdf + bad_pdf))
    order = np.argsort(d)
    assert np.any(np.diff(raw[order]) > .1)  # This example exercises correction.
    assert (np.diff(corrected[order]) <= 0).all()
    if good_variance > bad_variance:
        assert corrected[d == 5].item() < raw[d == 5].item()
    else:
        assert raw[d == -3].item() < .01
        assert_allclose(corrected[d == -3], raw.max())


@pytest.mark.parametrize("kwargs", [
    {"min_component_separation": 1e6},
    {"min_good_component_weight": .99},
])
def test_degenerate_gmm_returns_mean_and_unit_weights(kwargs):
    images = separated_images()
    estimator = RecursiveGMMEstimator(mean_l2_norm, random_state=8, gmm_max_iter=100, **kwargs)
    result = estimator.fit(ImageBatch(images), reference=np.zeros((1, 1)))
    assert_allclose(result.real_estimate, images.mean(0))
    assert_allclose(result.weights["real"], 1)
    assert estimator.n_its == 1
    assert not estimator.converged
    estimator.check_degenerate_model = False
    fitted = estimator.fit(ImageBatch(images), reference=np.zeros((1, 1)))
    assert abs(fitted.real_estimate.item() - .4) < .1


@pytest.mark.parametrize("norm", ["ortho", "forward"])
def test_fourier_gmm_fits_complex_data_and_converts_real_reference(norm):
    amplitudes = separated_images()
    pattern = np.arange(21.).reshape(3, 7) / 20 + 1
    batch = ImageBatch(amplitudes * pattern, norm=norm)
    solver = RecursiveGMMEstimator(mean_l2_norm, max_iter=2, random_state=12, gmm_max_iter=100)
    estimator = FourierGMM(solver)
    result = estimator.fit(batch, reference=np.zeros(batch.real_shape))
    weights = result.weights["fourier"]
    assert weights.shape == (60, 1, 1)
    assert weights[:48].mean() > .95
    assert weights[48:].mean() < .05
    assert_allclose(result.real_estimate, (batch.real * weights).sum(0) / weights.sum(0), atol=1e-13)
    assert_allclose(result.real_estimate, pattern * .4, atol=.1)
    assert result.weights["real"] is None
    transformed = estimator.fit(batch, reference=np.zeros(batch.rfft_shape, dtype=complex))
    assert_array_equal(result.real_estimate, transformed.real_estimate)
