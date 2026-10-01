from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from simple_mestimators.domain import ImageBatch
from simple_mestimators.estimators.admm import ADMMEstimator
from simple_mestimators.estimators.distances import mean_l1_norm, mean_l2_norm, tagare_distance
from simple_mestimators.estimators.fourier_irls import JointIRLSFourier
from simple_mestimators.estimators.gmm import FourierGMM, RecursiveGMMEstimator
from simple_mestimators.estimators.irls import IRLSSolver
from simple_mestimators.estimators.weights import (
    norm_smooth_redescending_weights, smooth_redescending_weights, tagare_weights,
)
from conftest import image_unit_weights, unit_weights


@pytest.mark.parametrize("width", [6, 7])
@pytest.mark.parametrize("norm", ["ortho", "forward", "backward"])
def test_fourier_roundtrip_and_unit_weights(width, norm):
    images = np.random.default_rng(11).normal(size=(5, 4, width))
    original = images.copy()
    batch = ImageBatch(images, norm=norm)
    spectrum = batch.ensure_fourier()
    assert spectrum.shape == (5, 4, width // 2 + 1)
    assert batch.ensure_fourier() is spectrum
    assert_allclose(spectrum, np.fft.rfft2(images, norm=norm))
    assert_allclose(np.fft.irfft2(spectrum, s=batch.real_shape, norm=norm), images, atol=1e-14)
    # rFFT Parseval needs multiplicity two for non-DC/non-Nyquist columns.
    multiplicity = np.full(width // 2 + 1, 2.0)
    multiplicity[0] = 1
    if width % 2 == 0:
        multiplicity[-1] = 1
    scale = {"ortho": 1, "forward": 1 / (4 * width), "backward": 4 * width}[norm]
    assert_allclose((abs(spectrum)**2 * multiplicity).sum(axis=(1, 2)),
                    (images**2).sum(axis=(1, 2)) * scale)
    solver = JointIRLSFourier(IRLSSolver(unit_weights, max_iter=2))
    result = solver.fit(batch, reference=np.zeros(batch.real_shape))
    assert result.real_estimate.shape == batch.real_shape
    assert_allclose(result.real_estimate, images.mean(axis=0), atol=1e-14)
    assert_allclose(result.weights["fourier"], 1)
    assert result.weights["real"] is None
    assert_array_equal(images, original)


@pytest.mark.parametrize("weight_function", [unit_weights, image_unit_weights])
def test_real_unit_weights_and_zero_iteration_reference(weight_function):
    images = np.random.default_rng(6).normal(size=(4, 3, 5))
    solver = IRLSSolver(weight_function, max_iter=3)
    result = solver.fit(ImageBatch(images), reference=np.zeros((3, 5)))
    assert_allclose(result.real_estimate, images.mean(axis=0))
    assert_allclose(result.weights["real"], 1)
    initial = np.full((3, 5), 7.0)
    estimate, weights = solver.solve(images, reference=initial, max_iter_override=0)
    assert_array_equal(estimate, initial)
    assert weights.shape == (4, 1, 1)
    assert_allclose(weights, 1)


@pytest.mark.parametrize("n_images", [1, 3])
def test_fourier_variance_includes_phase_and_uses_sample_ddof(n_images):
    # Equal magnitudes, different phases: variance of magnitudes would be zero.
    spectra = np.array([1 + 0j, 0 + 1j, -1 + 0j])[:n_images, None, None]
    batch = ImageBatch(np.zeros((n_images, 1, 1)), fourier=spectra)
    denominator = n_images - 1 if n_images > 1 else n_images
    expected = (abs(spectra - spectra.mean(axis=0))**2).sum(axis=0) / denominator
    variance = batch.fourier_residual_variance()
    assert not np.iscomplexobj(variance)
    assert_allclose(variance, expected)
    assert_allclose(batch.fourier_residual_variance("global"), expected.mean())
    if n_images > 1:
        assert variance.item() > 0
    real = np.arange(12.0).reshape(3, 2, 2)
    assert_allclose(ImageBatch(real).real_variance(), ((real - real.mean(0))**2).sum(0) / 2)


def test_complex_distances_and_redescending_weights_use_modulus():
    images = np.array([[[3 + 4j, 0 + 2j]], [[0 + 0j, -6 + 8j]]])
    reference = np.zeros((1, 2), complex)
    std = np.array([[1.0, 2.0]])
    # Scaled residual magnitudes: [5, 1] and [0, 5].
    squared = np.array([[[25., 1.]], [[0., 25.]]])
    assert_allclose(mean_l1_norm(images, reference, std), [3, 2.5])
    assert_allclose(mean_l2_norm(images, reference, std), np.sqrt([13, 12.5]))
    assert_allclose(smooth_redescending_weights(images, reference, std, delta=2), np.exp(-squared / 4))
    assert_allclose(norm_smooth_redescending_weights(images, reference, std, delta=2),
                    np.exp(-squared.mean(axis=(1, 2), keepdims=True) / 4))


def test_tagare_uses_hermitian_projection_and_is_phase_invariant():
    reference = np.array([[1 + 1j, 1 - 1j]])
    orthogonal = np.array([[1 + 1j, -1 + 1j]])
    images = np.stack([reference, 1j * reference, orthogonal, reference + orthogonal])
    expected = np.array([1, 1, 0, np.exp(-0.25 * 4) / np.sqrt(2)])
    assert_allclose(tagare_weights(images, reference, beta=.25).ravel(), expected, atol=1e-14)
    assert_allclose(tagare_distance(images, reference, beta=.25), -expected, atol=1e-14)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("prior_array", [False, True])
def test_masked_fourier_weights_ctf_and_prior_match_closed_form(local, prior_array):
    images = np.random.default_rng(3).normal(size=(4, 3, 5))
    batch = ImageBatch(images)
    fourier = batch.ensure_fourier()
    variance = batch.fourier_residual_variance()
    ctf = np.broadcast_to(np.array([1., -.5, 2., 0.])[:, None, None], fourier.shape).copy()
    batch.ctf = ctf
    mask = np.array([[True, False, True], [False, True, False], [True, False, False]])
    reference = np.zeros(batch.real_shape)
    prior = np.full(batch.real_shape, .4)
    prior_fourier = np.fft.rfft2(prior, norm=batch.norm)
    prior_variance = np.full(batch.rfft_shape, 2.) if prior_array else 2.
    fn = smooth_redescending_weights if local else norm_smooth_redescending_weights
    solver = JointIRLSFourier(IRLSSolver(partial(fn, delta=2), max_iter=1), mask=mask)
    result = solver.fit(batch, reference=reference, prior_mean=prior, prior_variance=prior_variance)
    # Independent one-step oracle: selected residuals yield weights, their mean
    # per image is then used to reconstruct the complete Fourier grid.
    residual_squared = abs(fourier[:, mask])**2 / variance[mask]
    selected = np.exp(-(residual_squared if local else residual_squared.mean(1, keepdims=True)) / 4)
    aggregate = selected.mean(axis=1)[:, None, None]
    numerator = (aggregate * ctf * fourier).sum(0) / variance + prior_fourier / prior_variance
    denominator = (aggregate * ctf**2).sum(0) / variance + 1 / prior_variance
    expected = np.fft.irfft2(numerator / denominator, s=batch.real_shape, norm=batch.norm)
    assert_allclose(result.real_estimate, expected, atol=1e-13)
    weights = result.weights["fourier"]
    if local:
        assert weights.shape == fourier.shape
        assert_allclose(weights[:, mask], selected)
        assert np.isnan(weights[:, ~mask]).all()
    else:
        assert weights.shape == (4, 1, 1)
        assert_allclose(weights[:, 0, 0], selected[:, 0])


def test_masked_zero_iteration_restores_global_weights():
    images = np.random.default_rng(4).normal(size=(3, 3, 5))
    batch = ImageBatch(images)
    mask = np.zeros(batch.rfft_shape, bool)
    mask[0, 0] = True
    result = JointIRLSFourier(IRLSSolver(unit_weights, max_iter=0), mask=mask).fit(batch)
    assert_allclose(result.real_estimate, images.mean(0), atol=1e-14)
    assert result.weights["fourier"].shape == (3, 1, 1)


def test_unit_weights_with_ctf_recover_clean_signal_and_prior_pairs_are_required():
    signal = np.array([[2 + 3j, -1 + 2j]])
    ctf = np.array([1., -2., .5])[:, None, None]
    observed = ctf * signal
    solver = IRLSSolver(image_unit_weights, max_iter=1)
    estimate, _ = solver.solve(observed, ctf=ctf)
    assert_allclose(estimate, signal)
    with pytest.raises(ValueError, match="provided together"):
        solver.solve(observed, prior_mean=signal)


@pytest.mark.parametrize("max_iter", [0, 5])
def test_admm_unit_weights_preserve_mean_and_odd_width(max_iter):
    images = np.random.default_rng(9).normal(size=(5, 4, 7))
    solver = ADMMEstimator(IRLSSolver(unit_weights, 2), JointIRLSFourier(IRLSSolver(unit_weights, 2)),
                           max_iter=max_iter, initial_mu=1., fourier_multiplier=1.)
    result = solver.fit(ImageBatch(images))
    assert result.real_estimate.shape == (4, 7)
    assert_allclose(result.real_estimate, images.mean(0), atol=1e-14)
    for weights in result.weights.values():
        assert_allclose(weights, 1)
    with pytest.raises(ValueError, match="unitary"):
        solver.fit(ImageBatch(images, norm="forward"))


@pytest.mark.parametrize("fourier", [False, True])
def test_gmm_zero_iteration_mean(fourier):
    images = np.random.default_rng(1).normal(size=(5, 3, 7))
    solver = RecursiveGMMEstimator(mean_l2_norm, max_iter=0)
    estimator = FourierGMM(solver) if fourier else solver
    result = estimator.fit(ImageBatch(images))
    assert_allclose(result.real_estimate, images.mean(0), atol=1e-14)
    assert result.gmm_diagnostics is None
    assert_allclose(result.weights["fourier" if fourier else "real"], 1)
