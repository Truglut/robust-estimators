"""Inspect initial image distances for a simulated run or stored MRC stack.

Place this module alongside run_simulation.py and run it as a package module.
It accepts the same YAML config and simulation CLI arguments, plus --distance,
--distance-params, --snr-index, --run-index, --bins, --density, and --no-show.
Local experimental_distances.py adds structural metrics, --reference-mode,
--reference-params, and --leave-one-out/--no-leave-one-out.
Use --source dataset for the MRC/config arguments of run_estimation.py.
Stored data produces one histogram because its labels are unavailable.
Only a histogram is saved under --results; no estimators are fitted.

To reproduce a simulation run, pass the same --snr list (in the same order),
--n-runs, --seed, and preprocessing arguments used for that simulation.
The selected SNR and run indices are zero-based.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import mrcfile
import numpy as np

from .cli import (
    build_simulation_parser, parse_simulation_cfg,
    build_estimation_parser, parse_estimation_cfg,
)
from .configs import InitializationStrategy, SimulationConfig
from .dataset_builder import LABELS
from .experimental_distances import (
    EXPERIMENTAL_DISTANCE_FUNCTIONS,
    contrast_balanced_reference,
    get_experimental_distance_function,
)
from .pipeline import (
    PreparedDataset, prepare_initial_reference, prepare_image_dataset,
)
from .run_simulation import prepare_simulation_dataset
from .run_estimation import load_image_stack
from simple_mestimators.estimators.distances import (
    DISTANCE_FUNCTION_REGISTRY,
    DistanceFunction,
    get_distance_function,
)
from simple_mestimators.masks import create_circular_mask


def build_simulation_mask(image_shape, radius):
    """Use the same mask and default radius as run_simulation."""
    if radius is None:
        h, w = image_shape
        radius = max(h // 2, w // 2)
    return create_circular_mask(image_shape=image_shape, radius=radius)


def build_dataset_for_run(
    cfg: SimulationConfig, snr_index: int = 0, run_index: int = 0
) -> PreparedDataset:
    """Reproduce the dataset preparation and seed used by run_simulation."""
    if not 0 <= snr_index < len(cfg.snrs):
        raise ValueError(f"snr_index must be in [0, {len(cfg.snrs)})")
    if not 0 <= run_index < cfg.n_runs:
        raise ValueError(f"run_index must be in [0, {cfg.n_runs})")

    ground_truth = np.asarray(mrcfile.read(cfg.io.reference_path))
    if ground_truth.ndim != 2:
        raise ValueError(f"Expected a 2D reference image, got {ground_truth.shape}")

    mask = build_simulation_mask(ground_truth.shape, cfg.preprocessing.mask_radius)

    seed = int(
        np.random.SeedSequence([cfg.seed, snr_index, run_index]).generate_state(
            1, dtype=np.uint64
        )[0]
    )
    return prepare_simulation_dataset(
        ground_truth=ground_truth,
        misclassified_path=cfg.io.misclassified_path,
        dataset_cfg=cfg.dataset,
        preprocessing_cfg=cfg.preprocessing,
        mask=mask,
        snr=cfg.snrs[snr_index],
        rng=np.random.default_rng(seed),
    )


def calculate_initial_distances(
    dataset: PreparedDataset,
    distance_function: DistanceFunction,
    initialization_strategy: InitializationStrategy = "mean",
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Return one finite real distance per image, without fitting an estimator.

    Parameters
    ----------
    dataset : PreparedDataset
        Images after the shared masking and standardization steps, with
        optional simulation labels and ground truth.
    distance_function : callable
        Function accepting ``(images, reference)``. Custom callables can be
        passed directly when experimenting without registering a new function.
    initialization_strategy : str, optional
        Reference strategy understood by ``prepare_initial_reference``.
        To add a new strategy, extend that shared function.
    reference : np.ndarray, optional
        Explicit reference overriding initialization_strategy. This permits
        experimental references without modifying the simulation's strategies.

    Returns
    -------
    np.ndarray
        Raw distances of shape ``(n_images,)``. No rescaling or GMM weighting
        is applied; negative distances such as Tagare's are preserved.
    """
    if reference is None:
        reference = prepare_initial_reference(dataset, initialization_strategy)
    images = dataset.preprocessed_images
    distances = np.asarray(distance_function(images, reference))
    if np.iscomplexobj(distances) or distances.size != images.shape[0]:
        raise ValueError(
            "The distance function must return one real distance per image"
        )
    distances = distances.reshape(-1)
    if not np.isfinite(distances).all():
        raise ValueError("The distance function returned non-finite values")
    return distances


def plot_distance_histogram(
    distances: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    bins: int = 50,
    density: bool = False,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot labeled histograms with shared bins, or one unlabeled histogram.

    With ``density=True``, each label's histogram integrates to one, making
    distributions easier to compare when label counts differ.
    """
    distances = np.asarray(distances)
    labels = None if labels is None else np.asarray(labels)
    if distances.ndim != 1 or (labels is not None and labels.shape != distances.shape):
        raise ValueError("distances and labels must be matching one-dimensional arrays")
    if distances.size == 0 or not np.isfinite(distances).all():
        raise ValueError("Distances must be nonempty and finite")
    if bins < 1:
        raise ValueError("bins must be positive")

    edges = np.histogram_bin_edges(distances, bins=bins)
    label_names = {value: name for name, value in LABELS.items()}
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    groups = [(None, distances)] if labels is None else [
        (label, distances[labels == label]) for label in np.unique(labels)
    ]
    for label, values in groups:
        color = colors(0 if label is None else int(label) % 10)
        name = "all images" if label is None else label_names.get(label, f"label {label}")
        ax.hist(
            values,
            bins=edges,
            density=density,
            histtype="stepfilled",
            alpha=0.4,
            color=color,
            edgecolor=color,
            label=f"{name} (n={values.size})",
        )
    ax.set_xlabel("Initial distance to reference")
    ax.set_ylabel("Probability density" if density else "Number of images")
    ax.set_title(title or ("Initial distances" if labels is None else "Initial distances by label"))
    ax.legend()
    return fig, ax


def parse_distance_params(value: str) -> dict:
    """Read distance keyword arguments from a JSON object on the CLI."""
    try:
        params = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError("Distance parameters must be a JSON object")
    return params


def build_distance_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    """Select the input-specific parser, retaining the old simulation default."""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--source", choices=("simulation", "dataset"), default="simulation")
    source, _ = selector.parse_known_args(argv)
    parser = build_simulation_parser() if source.source == "simulation" else build_estimation_parser()
    parser.description = "Plot initial image distances without fitting estimators"
    parser.add_argument("--source", choices=("simulation", "dataset"), default="simulation")
    parser.add_argument(
        "--distance",
        choices=sorted(
            set(DISTANCE_FUNCTION_REGISTRY) | set(EXPERIMENTAL_DISTANCE_FUNCTIONS)
        ),
        default="l2",
        help="Distance registry key (default: %(default)s)",
    )
    parser.add_argument(
        "--distance-params",
        type=parse_distance_params,
        default={},
        help='Distance keyword arguments as JSON, e.g. \'{"beta": "auto"}\'',
    )
    if source.source == "simulation":
        parser.add_argument("--snr-index", type=int, default=0, help="Zero-based SNR index")
        parser.add_argument("--run-index", type=int, default=0, help="Zero-based run index")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument(
        "--density", action="store_true", help="Normalize each label separately"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save the plot without displaying it"
    )
    parser.add_argument(
        "--reference-mode",
        choices=["initialization", "simulation", "contrast-balanced"],
        default="initialization",
        help="Use configured initialization or a contrast-balanced reference ('simulation' is a legacy alias)",
    )
    parser.add_argument(
        "--reference-params",
        type=parse_distance_params,
        default={},
        help="JSON keyword arguments for the contrast-balanced reference",
    )
    parser.add_argument(
        "--leave-one-out",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Remove self-contribution; automatic for experimental metrics with a linear reference",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_distance_parser(argv)
    args = parser.parse_args(argv)
    cfg = parse_simulation_cfg(args) if args.source == "simulation" else parse_estimation_cfg(args)

    if args.source == "simulation":
        dataset = build_dataset_for_run(cfg, args.snr_index, args.run_index)
    else:
        dataset = prepare_image_dataset(load_image_stack(cfg.io.dataset_path), cfg.preprocessing)
    images = dataset.preprocessed_images
    mask = dataset.mask
    if args.reference_mode == "contrast-balanced":
        reference, reference_weights = contrast_balanced_reference(
            images, mask=mask, **args.reference_params
        )
        reference_name = "contrast-balanced"
    else:
        if args.reference_params:
            parser.error(
                "--reference-params requires --reference-mode contrast-balanced"
            )
        reference = prepare_initial_reference(dataset, cfg.initialization_strategy)
        reference_name = cfg.initialization_strategy
        reference_weights = None
        if cfg.initialization_strategy == "mean":
            reference_weights = np.full(len(images), 1.0 / len(images))
        elif cfg.initialization_strategy == "mean-good":
            reference_weights = (dataset.labels == LABELS["good"]).astype(float)
            reference_weights /= reference_weights.sum()

    experimental = args.distance in EXPERIMENTAL_DISTANCE_FUNCTIONS
    leave_one_out = args.leave_one_out
    if leave_one_out is None:
        leave_one_out = bool(experimental and reference_weights is not None
                             and np.all(reference_weights < 1.0))
    if leave_one_out and (not experimental or reference_weights is None
                         or np.any(reference_weights >= 1.0)):
        parser.error(
            "Leave-one-out requires an experimental metric and linear reference weights below one"
        )
    if experimental:
        distance_function = get_experimental_distance_function(
            args.distance,
            args.distance_params,
            mask=mask,
            reference_weights=reference_weights,
            leave_one_out=leave_one_out,
        )
        if args.distance == "noise_debiased_cosine" and reference_name == "median":
            if "reference_noise_variance" not in args.distance_params:
                parser.error(
                    "Median reference noise is not calibrated: provide reference_noise_variance"
                )
    else:
        distance_function = get_distance_function(
            args.distance, args.distance_params, imgs=images
        )
    distances = calculate_initial_distances(
        dataset, distance_function, reference=reference
    )
    print(f"Reference={reference_name}; leave-one-out={leave_one_out}")
    groups = [("all images", distances)] if dataset.labels is None else [
        (name, distances[dataset.labels == label]) for name, label in LABELS.items()
    ]
    for name, values in groups:
        if values.size:
            print(
                f"{name:14s} n={values.size:5d} mean={values.mean():.6g} "
                f"std={values.std():.6g} median={np.median(values):.6g}"
            )
    if args.distance == "noise_debiased_cosine":
        print(
            f"Neutral distances: {np.count_nonzero(distances == 1.0)}/{len(distances)}; "
            f"clipped endpoints: {np.count_nonzero((distances == 0.0) | (distances == 2.0))}/{len(distances)}"
        )
    if args.source == "simulation":
        snr = cfg.snrs[args.snr_index]
        description = f"SNR={snr:g} | run={args.run_index}"
        dataset_suffix = f"_snr_{snr:g}_run_{args.run_index:03d}"
    else:
        description = cfg.io.dataset_path.name
        dataset_suffix = "_dataset"
    fig, _ = plot_distance_histogram(
        distances,
        dataset.labels,
        bins=args.bins,
        density=args.density,
        title=(
            f"{args.distance} | {description} | "
            f"reference={reference_name}"
            + (" | leave-one-out" if leave_one_out else "")
        ),
    )
    # Keep legacy filenames; distinguish new reference/parameter experiments.
    suffix = ""
    if experimental or args.reference_mode == "contrast-balanced":
        settings = json.dumps(
            {
                "distance": args.distance_params,
                "reference": args.reference_params,
                "strategy": reference_name,
                "leave_one_out": leave_one_out,
            },
            sort_keys=True,
        )
        suffix = (
            f"_{reference_name}_{hashlib.sha256(settings.encode()).hexdigest()[:8]}"
        )
    output = cfg.io.base_results_path / (
        f"initial_distances_{args.distance}{suffix}{dataset_suffix}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=180)
        print(f"Saved histogram to {output}")
        if not args.no_show:
            plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
