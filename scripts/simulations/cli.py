import argparse
from pathlib import Path


def build_base_parser() -> argparse.ArgumentParser:
    """
    Build the base command-line argument parser, with arguments that are common
    to simulated data and experimental data runs
    """
    parser = argparse.ArgumentParser(description="Run robust estimators on real data")

    # Global arguments
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )

    # Saving
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save ground truth, outliers, and estimates in .mrc files",
    )
    parser.add_argument(
        "--save-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the array of labels for each image.",
    )
    parser.add_argument(
        "--save-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the weights that each estimator produced",
    )

    return parser


def build_simulation_parser() -> argparse.ArgumentParser:
    """
    Builds the simulation argument parser by adding the simulation-specific
    arguments to the base parser.
    """
    parser = build_base_parser()

    parser.add_argument(
        "--initialization-strategy",
        type=str,
        choices=["average", "median", "average-good", "ground-truth"],
        help=(
            "Initialization to use for the estimators that take an initial reference. "
            "'average' means taking the average of all the input images. "
            "'median' means taking the median of all input images. "
            "'average-good' means taking the average of only the good input images. "
            "'ground-truth' means using the ground truth image as the initial reference. "
            "'average-good' and 'ground-truth' exist for testing purposes, but "
            "are clearly not realistic."
        ),
    )

    simulation_group = parser.add_argument_group("Simulation")

    # Simulation
    simulation_group.add_argument(
        "--snr",
        nargs="+",
        type=float,
        help="Target signal to noise ratios in image generation. Default is [0.01]",
        default=[0.01],
    )

    simulation_group.add_argument(
        "--standardize",
        choices=["before", "after", "both", "none"],
        default="after",
        help="When to standardize generated images. Default is %(default)s",
    )
    simulation_group.add_argument(
        "--per-image-noise-std",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add a different noise std to each image to achieve a uniform SNR. Default is False",
    )
    simulation_group.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of simulations to run with the specified configuration.",
    )
    simulation_group.add_argument(
        "--reference-image-path",
        type=Path,
        help="Path to the .mrc file containing the image that will be used as the reference for the simulation. If provided, will override the path in the config file.",
    )
    simulation_group.add_argument(
        "--misclassified-path",
        type=Path,
        help="Path to the .mrc file containing the image that will be used as misclassified images for the simulation. If provided, will override the path in the config file.",
    )
    simulation_group.add_argument(
        "--standardize-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, the reference image will be standardized to zero mean and unit std before generating the dataset",
    )
    return parser


# TODO: parse into nice config objects
