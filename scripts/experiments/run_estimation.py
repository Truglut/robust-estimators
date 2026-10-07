"""Run the shared estimator pipeline on a stored stack of aligned 2D images."""

from pathlib import Path

import mrcfile
import numpy as np

from .cli import build_estimation_parser, parse_estimation_cfg
from .configs import EstimationConfig
from .output import initialize_output, save_estimator_result
from .pipeline import (
    PreparedDataset,
    effective_config,
    preprocess_images,
    run_experiment,
    sample_mean_result,
)
from simple_mestimators.masks import create_circular_mask


def load_image_stack(path: Path) -> np.ndarray:
    """Load (n, height, width), treating a 2D MRC as a one-image stack.

    The leading dimension of a 3D file is interpreted as the image index.
    Particle order is retained; no alignment or resampling is performed.
    """
    images = np.asarray(mrcfile.read(path))
    if images.ndim == 2:
        images = images[None, ...]
    if images.ndim != 3 or any(size == 0 for size in images.shape):
        raise ValueError(
            f"Expected a nonempty (n, height, width) MRC stack, got {images.shape}"
        )
    if not np.issubdtype(images.dtype, np.number) or np.iscomplexobj(images):
        raise ValueError("Expected real-valued image data")
    if not np.isfinite(images).all():
        raise ValueError("MRC stack contains non-finite pixel values")
    return images


def run_estimation(cfg: EstimationConfig) -> Path:
    cfg = effective_config(cfg)
    images = load_image_stack(cfg.io.dataset_path)
    source_shape, source_dtype = images.shape, str(images.dtype)
    radius = cfg.preprocessing.mask_radius
    mask = (
        None
        if radius is None
        else create_circular_mask(
            image_shape=images.shape[-2:],
            radius=radius,
        )
    )
    dataset = PreparedDataset(
        preprocessed_images=preprocess_images(images, cfg.preprocessing, mask),
        mask=mask,
    )
    del images
    root = initialize_output(
        cfg,
        mask=mask,
        dataset_metadata={
            "source_path": str(Path(cfg.io.dataset_path).expanduser().resolve()),
            "shape": source_shape,
            "dtype": source_dtype,
            "image_axis": 0,
        },
    )
    save_estimator_result(
        root / "estimators" / "mean" / "run_000",
        sample_mean_result(dataset),
        cfg.io,
    )
    for name, result in run_experiment(
        dataset, cfg.estimators, cfg.initialization_strategy
    ):
        save_estimator_result(root / "estimators" / name / "run_000", result, cfg.io)
        del result
    return root


def main() -> None:
    parser = build_estimation_parser()
    cfg = parse_estimation_cfg(parser.parse_args())
    root = run_estimation(cfg)
    print(f"Saved results to {root}.")


if __name__ == "__main__":
    main()
