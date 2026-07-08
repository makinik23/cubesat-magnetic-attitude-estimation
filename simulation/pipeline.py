"""
Main simulation pipeline orchestration.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.config import (
    create_default_attitude_config,
    create_default_orbit,
    create_default_simulation_config,
)
from simulation.plots import (
    animate_attitude_cube,
    plot_angular_velocity_body,
    plot_attitude_orientation,
    plot_attitude_quaternion,
    plot_magnetic_field_body,
    plot_magnetic_field_body_norm,
    plot_magnetic_field_eci,
    plot_magnetic_field_norm,
    plot_magnetometer_measurement,
    plot_orbit_3d,
    plot_position_eci,
    plot_position_norm,
    plot_r_eci_time,
    plot_velocity_norm,
)
from simulation.results import build_results_dataframe, print_sanity_check, save_results
from simulation.runner import SimulationRunner
from simulation.sensors import MagnetometerModel
from simulation.types import AttitudeConfig, ClassicalOrbitalElements, SimulationConfig

Plotter = Callable[[pd.DataFrame, Path], None]

PLOT_OUTPUTS: tuple[tuple[str, Plotter], ...] = (
    ("position_eci.png", plot_position_eci),
    ("r_eci_time.png", plot_r_eci_time),
    ("position_norm.png", plot_position_norm),
    ("velocity_norm.png", plot_velocity_norm),
    ("orbit_3d.png", plot_orbit_3d),
    ("magnetic_field_eci.png", plot_magnetic_field_eci),
    ("magnetic_field_norm.png", plot_magnetic_field_norm),
    ("magnetic_field_body.png", plot_magnetic_field_body),
    ("magnetic_field_body_norm.png", plot_magnetic_field_body_norm),
    ("magnetometer_measurement.png", plot_magnetometer_measurement),
    ("attitude_orientation.png", plot_attitude_orientation),
    ("attitude_quaternion.png", plot_attitude_quaternion),
    ("angular_velocity_body.png", plot_angular_velocity_body),
)

ANIMATION_OUTPUTS: tuple[tuple[str, Plotter], ...] = (("attitude_cube.gif", animate_attitude_cube),)


def load_default_inputs() -> tuple[ClassicalOrbitalElements, SimulationConfig, AttitudeConfig]:
    """Load default orbit, simulation and attitude settings."""

    return (
        create_default_orbit(),
        create_default_simulation_config(),
        create_default_attitude_config(),
    )


def create_default_runner() -> SimulationRunner:
    """Create the default simulation runner."""

    return SimulationRunner(
        magnetometer_model=MagnetometerModel(
            bias_body_t=np.array([0.3e-6, -0.2e-6, 0.1e-6]), noise_std_t=1.0e-6, seed=42
        )
    )


def save_plot_outputs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create static plots and return their paths."""

    paths = []

    for filename, plotter in PLOT_OUTPUTS:
        plotter(df, output_dir)
        paths.append(output_dir / filename)

    return paths


def save_animation_outputs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create animations and return their paths."""

    paths = []

    for filename, plotter in ANIMATION_OUTPUTS:
        plotter(df, output_dir)
        paths.append(output_dir / filename)

    return paths


def print_saved_outputs(
    csv_path: Path, plot_paths: list[Path], animation_paths: list[Path]
) -> None:
    """Print generated output paths."""

    print(f"Saved orbit data to: {csv_path}")

    for path in plot_paths:
        print(f"Saved plot: {path}")

    for path in animation_paths:
        print(f"Saved animation: {path}")


def run_orbit_pipeline(output_dir: Path) -> None:
    """Run the orbit propagation pipeline."""

    elements, simulation_config, attitude_config = load_default_inputs()
    runner = create_default_runner()
    result = runner.run(elements, simulation_config, attitude_config)
    df = build_results_dataframe(result)

    csv_path = save_results(df, output_dir)
    plot_paths = save_plot_outputs(df, output_dir)
    animation_paths = save_animation_outputs(df, output_dir)

    print_saved_outputs(csv_path, plot_paths, animation_paths)
    print_sanity_check(df)
