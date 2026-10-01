import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from scripts.simulations.configs import DatasetConfig, PreprocessingConfig
from scripts.simulations.dataset_builder import create_evaluation_dataset, get_dataset_labels
from scripts.simulations.run_simulation import prepare_simulation_dataset, preprocess_images


@pytest.mark.parametrize("standardize_reference", [False, True])
def test_reference_scaled_once_without_centering_and_inputs_preserved(reference, write_mrc, standardize_reference):
    original = reference.copy()
    outlier = reference * 2 + 1
    path = write_mrc("outlier.mrc", outlier)
    cfg = DatasetConfig(n_copies=2, n_misclassified=1, n_noise=1,
                        standardize_reference=standardize_reference)
    scale = reference.std() if standardize_reference else 1.
    expected_reference = reference / scale
    seed = 27
    images, target, labels = create_evaluation_dataset(reference, cfg, np.random.default_rng(seed),
                                                       snr=.5, misclassified_path=path)
    # Account for sampling the single outlier before drawing Gaussian noise.
    rng = np.random.default_rng(seed)
    rng.choice(1, size=1, replace=False)
    clean = np.stack([expected_reference, expected_reference,
                      outlier.astype(np.float32) / scale, np.zeros_like(reference)])
    expected = clean + rng.normal(0, np.sqrt(expected_reference.var() / .5), size=clean.shape)
    assert_allclose(images, expected, atol=2e-6)
    assert_array_equal(reference, original)
    assert_allclose(target, expected_reference)
    assert target.mean() > 0
    assert_array_equal(labels, [0, 0, 3, 4])
    replay = create_evaluation_dataset(reference, cfg, np.random.default_rng(seed), snr=.5, misclassified_path=path)
    for actual, repeated in zip((images, target, labels), replay):
        assert_array_equal(actual, repeated)


def test_mixed_dataset_labels_and_reproducibility(reference, write_mrc):
    path = write_mrc("outliers.mrcs", np.stack([reference * 2, reference[::-1], reference + 3]))
    cfg = DatasetConfig(n_copies=3, n_rotated=2, n_shifted=2, n_misclassified=4, n_noise=2,
                        max_rotation_inliers=5, min_rotation_outliers=20, max_rotation_outliers=40,
                        min_shift_outliers=.5, max_shift_outliers=1.5,
                        min_rotation_misclassified=5, max_rotation_misclassified=10)
    def generate(seed):
        return create_evaluation_dataset(reference, cfg, np.random.default_rng(seed),
                                          snr=.2, misclassified_path=path)
    first = generate(42)
    second = generate(42)
    assert first[0].shape == (13, 5, 7)
    assert_array_equal(first[2], [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    assert_array_equal(first[2], get_dataset_labels(cfg))
    for actual, replay in zip(first, second):
        assert_array_equal(actual, replay)
    assert not np.array_equal(first[0], generate(43)[0])


@pytest.mark.parametrize("standardize", ["none", "before", "after", "both"])
def test_before_and_after_standardization_process_target_like_inputs(reference, standardize):
    original = reference.copy()
    cfg = DatasetConfig(n_copies=2, n_noise=1)
    preprocessing = PreprocessingConfig(standardize=standardize)
    mask = np.ones(reference.shape, bool)
    mask[0] = False
    dataset = prepare_simulation_dataset(ground_truth=reference, misclassified_path=None,
                                         dataset_cfg=cfg, preprocessing_cfg=preprocessing,
                                         mask=mask, snr=1e30, rng=np.random.default_rng(3))
    expected = reference / reference.std()
    if standardize in ("before", "both"):
        expected = (expected - expected.mean()) / expected.std()
    expected = expected * mask
    if standardize in ("after", "both"):
        expected = (expected - expected.mean()) / expected.std()
    assert_allclose(dataset.ground_truth, expected, atol=1e-14)
    assert_allclose(dataset.preprocessed_images[:2], np.broadcast_to(expected, (2, *expected.shape)), atol=1e-13)
    assert_array_equal(reference, original)
    assert np.isfinite(dataset.preprocessed_images).all()
    if standardize in ("after", "both"):
        assert_allclose(dataset.preprocessed_images.mean(axis=(1, 2)), 0, atol=1e-14)
        assert_allclose(dataset.preprocessed_images.std(axis=(1, 2)), 1)


def test_preprocessing_copies_input_and_handles_constant_images():
    images = np.array([np.ones((3, 4)) * 4, np.arange(12).reshape(3, 4)])
    original = images.copy()
    result = preprocess_images(images, PreprocessingConfig(standardize="after"), None)
    assert_array_equal(images, original)
    assert_array_equal(result[0], 0)
    assert_allclose(result[1].mean(), 0, atol=1e-14)
    assert_allclose(result[1].std(), 1)
