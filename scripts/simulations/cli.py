import argparse
from pathlib import Path
import yaml
from typing import get_args

from .configs import (
    IOConfig,
    DatasetConfig,
    PreprocessingConfig,
    StandardizeOption,
    InitializationStrategy,
    SimulationConfig,
    EstimatorConfig,
)


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
        "--results",
        type=Path,
        required=True,
        help="Path to the directory where the results should be saved",
    )

    for flag in ("weights", "estimates", "labels"):
        parser.add_argument(
            f"--save-{flag}",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=f"Save {flag} (default: %(default)s)",
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
        choices=get_args(InitializationStrategy),
        default="mean",
        help=(
            "Initialization to use for the estimators that take an initial reference. "
            "'mean' means taking the average of all the input images. "
            "'median' means taking the median of all input images. "
            "'mean-good' means taking the average of only the good input images. "
            "'ground-truth' means using the ground truth image as the initial reference. "
            "'mean-good' and 'ground-truth' exist for testing purposes, but "
            "are clearly not realistic."
        ),
    )

    simulation_group = parser.add_argument_group("Simulation")
    simulation_group.add_argument(
        "--seed", type=int, default=42, help="Simulation seed (default: %(default)s)"
    )

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
        choices=get_args(StandardizeOption),
        default="none",
        help="When to standardize generated images. Default is %(default)s",
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
        help="If True, the reference image will be standardized to unit std before generating the dataset",
    )
    return parser


def parse_simulation_cfg(args: argparse.Namespace):
    config_path = Path(args.config)
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    reference_path = args.reference_image_path or cfg["data"]["reference_image_path"]
    
    misclassified_path = None
    if args.misclassified_path is not None:
        misclassified_path = Path(args.misclassified_path)
    elif (m_path := cfg["data"].get("misclassified_path")) is not None:
        misclassified_path = Path(m_path)

    io_cfg = IOConfig(
        reference_path=Path(reference_path),
        misclassified_path=misclassified_path,
        base_results_path=Path(args.results),
        save_weights=args.save_weights,
        save_estimates=args.save_estimates,
        save_labels=args.save_labels,
    )

    mask_cfg = cfg.get("mask")
    radius = None if mask_cfg is None else mask_cfg.get("radius")
    preprocessing_cfg = PreprocessingConfig(
        mask_radius=radius,
        standardize=args.standardize,
    )

    dataset_cfg = DatasetConfig(
        standardize_reference=args.standardize_reference, **cfg["generation"]
    )

    estimator_cfgs: list[EstimatorConfig] = []
    for method_cfg in cfg["experiment"]["methods"]:
        estimator_cfgs.append(
            EstimatorConfig(
                name=method_cfg["name"],
                estimator_type=method_cfg["type"],
                params=method_cfg["params"],
            )
        )

    sim_cfg = SimulationConfig(
        snrs=args.snr,
        n_runs=args.n_runs,
        dataset=dataset_cfg,
        initialization_strategy=args.initialization_strategy,
        io=io_cfg,
        preprocessing=preprocessing_cfg,
        estimators=estimator_cfgs,
        seed=args.seed,
    )

    return sim_cfg
