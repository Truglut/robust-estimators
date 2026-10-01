from pathlib import Path

import numpy as np
import mrcfile
import scipy

from .configs import DatasetConfig

LABELS = {
    "good": 0,
    "rotated": 1,
    "shifted": 2,
    "misclassified": 3,
    "noise": 4,
}
MAX_LABELS = max(LABELS.values())


def add_noise(
    images: np.ndarray,
    rng: np.random.Generator,
    snr: float,
    signal_var: float | None = None,
) -> np.ndarray:
    """
    Adds Gaussian noise to a batch of images based on a target Signal-to-Noise Ratio (SNR).

    Parameters
    ----------
    images : np.ndarray
        Input image array of shape ``(n_images, height, width)``.
    rng : np.random.Generator
        Random number generator for sampling noise distributions.
    snr : float
        Target Signal-to-Noise Ratio.
    signal_var : float | None, optional
        Pre-calculated variance of the signal. If None, the variance is
        estimated from the input images, by default None.

    Returns
    -------
    np.ndarray
        Images with added Gaussian noise, same shape as input.
    """
    if not np.isfinite(snr) or snr <= 0:
        raise ValueError("snr must be finite and positive")

    if signal_var is None:
        signal_var = float(images.var(axis=(-2, -1)).mean())

    noise_var = signal_var / snr
    noise_std = np.sqrt(noise_var)

    print_noise_specs(signal_var, noise_std, snr)

    return images + rng.normal(0, noise_std, size=images.shape)


def print_noise_specs(signal_var: float, noise_std: np.ndarray, snr: float):
    signal_std = np.sqrt(signal_var)
    print("Adding noise to images:")
    print(f"\t- Signal std:         {signal_std:.4f}\tVariance:  {signal_var:6f}")
    print(f"\t- Noise std:          {noise_std:.4f}\tVariance:  {noise_std**2:.6f}")
    print(f"\t- SNR:                {snr:.4f}\n")


def generate_rotated_copies(
    image: np.ndarray, angles_deg: np.ndarray, interpolation_order: int = 3
) -> np.ndarray:
    """Generate randomly rotated copies of a single 2D reference image."""
    if angles_deg.ndim != 1:
        raise ValueError("Expected a one-dimensional angles_deg array")

    n_copies = len(angles_deg)
    output_images = np.empty(
        (n_copies, image.shape[0], image.shape[1]), dtype=image.dtype
    )

    # Spline interpolation of order > 1 requires prefiltering. Since all
    # rotations use the same source image, compute the spline coefficients
    # once instead of repeating the prefilter for every copy.
    if interpolation_order > 1:
        rotation_input = scipy.ndimage.spline_filter(
            image,
            order=interpolation_order,
            output=np.float64,
            mode="constant",
        )
    else:
        rotation_input = image

    for i, angle in enumerate(angles_deg):
        scipy.ndimage.rotate(
            rotation_input,
            angle,
            order=interpolation_order,
            reshape=False,
            output=output_images[i],
            mode="constant",
            prefilter=False,
        )

    return output_images


def apply_random_rotations(
    images: np.ndarray,
    angles_deg: np.ndarray,
    interpolation_order: int = 3,
    output_images: np.ndarray | None = None,
):
    """
    Applies 2D rotations to a batch of images using pre-sampled angles.

    Parameters
    ----------
    images : np.ndarray
        Input images to rotate. Array of shape ``(n_images, height, width)``.
    angles_deg : np.ndarray
        1D array of rotation angles in degrees. Must match the length of ``images``.
    interpolation_order : int, optional
        Order of spline interpolation for ``scipy.ndimage.rotate``, by default 3.
    output_images : np.ndarray | None, optional
        Pre-allocated array to store the results. If None, a new array is created.

    Returns
    -------
    np.ndarray
        Rotated images array of shape ``(n_images, height, width)``.
    """
    if output_images is None:
        output_images = np.empty_like(images)

    for i, angle in enumerate(angles_deg):
        scipy.ndimage.rotate(
            images[i],
            angle,
            order=interpolation_order,
            reshape=False,
            output=output_images[i],
        )

    return output_images


def generate_shifted_copies(
    image: np.ndarray, shifts: np.ndarray, interpolation_order: int = 3
):
    """
    Generates shifted copies of a single 2D reference image.

    Parameters
    ----------
    image : np.ndarray
        Input 2D reference image of shape ``(height, width)``.
    shifts : np.ndarray
        Array of shape ``(n_copies, 2)`` containing the vertical and horizontal
        shifts in pixels for each copy.
    interpolation_order : int, optional
        Spline interpolation order passed to ``scipy.ndimage.shift``, by default 3.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_copies, height, width)`` containing the shifted images.
    """
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError(f"Expected a shifts array of shape (n, 2), got {shifts.shape}")

    n_copies = len(shifts)
    output = np.empty(shape=(n_copies, *image.shape), dtype=image.dtype)

    # Spline interpolation of order > 1 requires prefiltering. Since all
    # shifts use the same source image, compute the spline coefficients once.
    if interpolation_order > 1:
        shift_input = scipy.ndimage.spline_filter(
            image,
            order=interpolation_order,
            output=np.float64,
            mode="constant",
        )
    else:
        shift_input = image

    for i, shift in enumerate(shifts):
        scipy.ndimage.shift(
            shift_input,
            shift,
            order=interpolation_order,
            output=output[i],
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    return output


def load_misclassified_images(
    image_path: Path, n_copies: int, rng: np.random.Generator
) -> np.ndarray:
    """Loads a file of outlier images and samples the requested number."""
    images = mrcfile.read(image_path)

    # Handle the one image case
    if images.ndim == 2:
        h, w = images.shape
        images = images.reshape(1, h, w)

    n_available = images.shape[0]

    replace = n_available < n_copies

    # Sample indices
    indices = rng.choice(n_available, size=n_copies, replace=replace)

    return images[indices]


def get_dataset_labels(dataset_cfg: DatasetConfig):
    labels = np.zeros(dataset_cfg.total_copies, dtype=np.min_scalar_type(MAX_LABELS))
    start = 0

    count = dataset_cfg.n_copies
    labels[start : start + count] = LABELS["good"]
    start += count

    count = dataset_cfg.n_rotated
    labels[start : start + count] = LABELS["rotated"]
    start += count

    count = dataset_cfg.n_shifted
    labels[start : start + count] = LABELS["shifted"]
    start += count

    count = dataset_cfg.n_misclassified
    labels[start : start + count] = LABELS["misclassified"]
    start += count

    count = dataset_cfg.n_noise
    labels[start : start + count] = LABELS["noise"]
    start += count

    return labels


def create_evaluation_dataset(
    reference: np.ndarray,
    dataset_cfg: DatasetConfig,
    rng: np.random.Generator,
    snr: float,
    misclassified_path: Path | None = None,
    standardize_before_noise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an evaluation dataset with rotated inliers, rotated outliers,
    misclassified outliers, and applies Gaussian noise.
    """
    reference = np.array(reference, copy=True)
    if not np.isfinite(reference).all():
        raise ValueError("'reference' contains non-finite values.")
    if reference.ndim != 2:
        raise ValueError("Expected a 2D reference image.")

    ref_std = reference.std()
    if dataset_cfg.standardize_reference:
        reference /= ref_std

    # Pre-allocate the dataset arrays
    total_copies = dataset_cfg.total_copies
    dataset = np.empty((total_copies, *reference.shape), dtype=reference.dtype)
    labels = np.zeros(total_copies, dtype=np.min_scalar_type(MAX_LABELS))

    current_idx = 0

    def add_block(data: np.ndarray | float, count: int, label: int) -> None:
        nonlocal current_idx
        end_idx = current_idx + count
        dataset[current_idx:end_idx] = data
        labels[current_idx:end_idx] = label
        current_idx = end_idx

    if dataset_cfg.n_copies > 0:
        max_angle = dataset_cfg.max_rotation_inliers
        if max_angle is None:
            add_block(reference, dataset_cfg.n_copies, label=0)
        else:
            angles = rng.uniform(0.0, max_angle, size=dataset_cfg.n_copies)
            add_block(
                generate_rotated_copies(reference, angles),
                dataset_cfg.n_copies,
                label=LABELS["good"],
            )

    if dataset_cfg.n_rotated > 0:
        min_angle = dataset_cfg.min_rotation_outliers
        max_angle = dataset_cfg.max_rotation_outliers
        assert min_angle is not None and max_angle is not None
        angles = rng.uniform(min_angle, max_angle, size=dataset_cfg.n_rotated)

        add_block(
            generate_rotated_copies(reference, angles),
            dataset_cfg.n_rotated,
            label=LABELS["rotated"],
        )

    if dataset_cfg.n_shifted > 0:
        min_shift = dataset_cfg.min_shift_outliers
        max_shift = dataset_cfg.max_shift_outliers
        assert (min_shift is not None) and (max_shift is not None)

        magnitudes = rng.uniform(min_shift, max_shift, size=(dataset_cfg.n_shifted, 2))
        signs = rng.choice((-1.0, 1.0), size=(dataset_cfg.n_shifted, 2))
        shifts = magnitudes * signs
        add_block(
            generate_shifted_copies(reference, shifts),
            dataset_cfg.n_shifted,
            label=LABELS["shifted"],
        )

    if dataset_cfg.n_misclassified > 0:
        assert misclassified_path is not None

        misclassified_images = load_misclassified_images(
            misclassified_path, dataset_cfg.n_misclassified, rng=rng
        )

        min_angle = dataset_cfg.min_rotation_misclassified
        max_angle = dataset_cfg.max_rotation_misclassified
        if min_angle is not None and max_angle is not None:
            angles = rng.uniform(min_angle, max_angle, size=dataset_cfg.n_misclassified)
            misclassified_images = apply_random_rotations(
                misclassified_images, angles, output_images=misclassified_images
            )

        # Misclassified images should be standardized in the same way as the reference
        if dataset_cfg.standardize_reference:
            misclassified_images /= ref_std

        add_block(
            misclassified_images,
            dataset_cfg.n_misclassified,
            label=LABELS["misclassified"],
        )

    # Add images with no signal (all zeros, noise will be added later)
    if dataset_cfg.n_noise > 0:
        add_block(0.0, dataset_cfg.n_noise, label=LABELS["noise"])

    # If requested, standardize images before adding noise
    if standardize_before_noise:
        # Standardize ground truth
        reference = (reference - reference.mean()) / reference.std()

        # Standardize generated images (only those with signal)
        signal_mask = labels != LABELS["noise"]
        if np.any(signal_mask):
            sub_dataset = dataset[signal_mask]
            means = sub_dataset.mean(axis=(1, 2), keepdims=True)
            stds = sub_dataset.std(axis=(1, 2), keepdims=True)

            # Guard against zero-variance images
            stds[stds == 0.0] = 1.0
            dataset[signal_mask] = (sub_dataset - means) / stds

    # Add noise to the array
    final_dataset = add_noise(dataset, rng=rng, signal_var=reference.var(), snr=snr)

    return final_dataset, reference, labels
