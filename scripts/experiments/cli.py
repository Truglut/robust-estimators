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
    EstimationIOConfig,
    EstimationConfig,
    DataInitializationStrategy,
)


def build_base_parser(*, simulation: bool = True) -> argparse.ArgumentParser:
    """
    Build the base command-line argument parser, with arguments that are common
    to simulated data and experimental data runs
    """
    parser = argparse.ArgumentParser(
        description="Run robust estimators on image datasets"
    )

    # Global arguments
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )

    # Saving
    parser.add_argument(
        "--results",
        type=Path,
        required=simulation,
        help="Path to the directory where the results should be saved",
    )

    flags = (
        ("weights", "estimates", "labels") if simulation else ("weights", "estimates")
    )
    for flag in flags:
        parser.add_argument(
            f"--save-{flag}",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"Save {flag} (default: %(default)s)",
        )
    return parser


def build_simulation_parser() -> argparse.ArgumentParser:
    """
    Builds the simulation argument parser by adding the simulation-specific
    arguments to the base parser.
    """
    parser = build_base_parser()
    parser.description = "Run simulations; scientific parameters are read from YAML."
    parser.add_argument(
        "--reference-image-path",
        type=Path,
        help="Path to the .mrc file containing the image that will be used as the reference for the simulation. If provided, will override the path in the config file.",
    )
    parser.add_argument(
        "--misclassified-path",
        type=Path,
        help="Path to the .mrc file containing the image that will be used as misclassified images for the simulation. If provided, will override the path in the config file.",
    )
    return parser


def parse_simulation_cfg(args: argparse.Namespace):
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML configuration must be a mapping")
    unknown = set(cfg) - {"data", "output", "experiment", "simulation", "preprocessing", "generation", "mask"}
    if unknown:
        raise ValueError(f"Unknown simulation configuration sections: {sorted(unknown)}")
    data = cfg.get("data", {})
    output = cfg.get("output", {})
    experiment = cfg.get("experiment", {})
    simulation = cfg.get("simulation", {})
    preprocessing = cfg.get("preprocessing", {})
    for section, mapping, allowed in (
        ("data", data, {"reference_image_path", "misclassified_path"}),
        ("output", output, {"save_weights", "save_estimates", "save_labels"}),
        ("preprocessing", preprocessing, {"mask_radius", "standardize"}),
    ):
        if set(mapping) - allowed:
            raise ValueError(f"Unknown {section} settings: {sorted(set(mapping) - allowed)}")
    unknown = set(simulation) - {"snrs", "n_runs"}
    if unknown:
        raise ValueError(f"Unknown simulation settings: {sorted(unknown)}")
    unknown = set(experiment) - {"methods", "seed", "initialization_strategy"}
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    methods = experiment.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("experiment.methods must contain at least one estimator")

    def path_option(cli_path, yaml_path, name, *, required=True):
        if cli_path is not None:
            return Path(cli_path).expanduser().resolve()
        if yaml_path is None:
            if required:
                raise ValueError(f"Provide {name} in YAML or on the command line")
            return None
        path = Path(yaml_path).expanduser()
        return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()

    reference_path = path_option(
        args.reference_image_path, data.get("reference_image_path"), "data.reference_image_path"
    )
    misclassified_path = path_option(
        args.misclassified_path, data.get("misclassified_path"), "data.misclassified_path", required=False
    )

    io_cfg = IOConfig(
        reference_path=Path(reference_path),
        misclassified_path=misclassified_path,
        base_results_path=Path(args.results),
        save_weights=output.get("save_weights", True) if args.save_weights is None else args.save_weights,
        save_estimates=output.get("save_estimates", True) if args.save_estimates is None else args.save_estimates,
        save_labels=output.get("save_labels", True) if args.save_labels is None else args.save_labels,
    )

    if "mask" in cfg and "mask_radius" in preprocessing:
        raise ValueError("Use preprocessing.mask_radius or legacy mask.radius, not both")
    radius = preprocessing.get("mask_radius", (cfg.get("mask") or {}).get("radius"))
    preprocessing_cfg = PreprocessingConfig(
        mask_radius=radius,
        standardize=preprocessing.get("standardize", "none"),
    )
    dataset_cfg = DatasetConfig(**cfg["generation"])

    estimator_cfgs: list[EstimatorConfig] = []
    for method_cfg in methods:
        estimator_cfgs.append(
            EstimatorConfig(
                name=method_cfg["name"],
                estimator_type=method_cfg["type"],
                params=method_cfg["params"],
            )
        )

    sim_cfg = SimulationConfig(
        snrs=simulation.get("snrs", [0.01]),
        n_runs=simulation.get("n_runs", 1),
        dataset=dataset_cfg,
        initialization_strategy=experiment.get("initialization_strategy", "mean"),
        io=io_cfg,
        preprocessing=preprocessing_cfg,
        estimators=estimator_cfgs,
        seed=experiment.get("seed", 42),
    )

    return sim_cfg


def build_estimation_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(simulation=False)
    parser.description = "Run robust estimators on an existing MRC image stack"
    parser.add_argument(
        "--dataset-path", type=Path, help="Override data.dataset_path in YAML"
    )
    parser.add_argument(
        "--initialization-strategy", choices=get_args(DataInitializationStrategy)
    )
    parser.add_argument("--standardize", choices=("none", "after"))
    parser.add_argument(
        "--mask-radius", type=float, help="Circular mask radius in pixels"
    )
    parser.add_argument("--seed", type=int, help="Default GMM random state")
    return parser


def parse_estimation_cfg(args: argparse.Namespace) -> EstimationConfig:
    """CLI overrides YAML; relative YAML paths are resolved beside the YAML file."""
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML configuration must be a mapping")

    def path_option(cli_path, yaml_path, name):
        if cli_path is not None:
            return Path(cli_path).expanduser().resolve()
        if yaml_path is None:
            raise ValueError(f"Provide {name} in YAML or on the command line")
        path = Path(yaml_path).expanduser()
        return (
            path.resolve()
            if path.is_absolute()
            else (config_path.parent / path).resolve()
        )

    def override(value, fallback):
        return fallback if value is None else value

    data = cfg.get("data", {})
    output = cfg.get("output", {})
    experiment = cfg.get("experiment", {})
    preprocessing = cfg.get("preprocessing", {})
    methods = experiment.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("experiment.methods must contain at least one estimator")
    io_cfg = EstimationIOConfig(
        dataset_path=path_option(
            args.dataset_path, data.get("dataset_path"), "data.dataset_path"
        ),
        base_results_path=path_option(
            args.results, output.get("results_path"), "output.results_path"
        ),
        save_weights=override(args.save_weights, output.get("save_weights", True)),
        save_estimates=override(
            args.save_estimates, output.get("save_estimates", True)
        ),
    )
    return EstimationConfig(
        io=io_cfg,
        estimators=[
            EstimatorConfig(m["name"], m["type"], m.get("params", {})) for m in methods
        ],
        preprocessing=PreprocessingConfig(
            mask_radius=override(args.mask_radius, preprocessing.get("mask_radius")),
            standardize=override(
                args.standardize, preprocessing.get("standardize", "none")
            ),
        ),
        initialization_strategy=override(
            args.initialization_strategy,
            experiment.get("initialization_strategy", "mean"),
        ),
        seed=override(args.seed, experiment.get("seed", 42)),
    )
