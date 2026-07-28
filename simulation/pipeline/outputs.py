"""
Main simulation pipeline orchestration.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.config import (
    create_default_attitude_config,
    create_default_orbit,
    create_default_simulation_config,
)
from simulation.estimation import AEKF, AEKFConfig, QuaternionAEKF, QuaternionAEKFConfig
from simulation.visualization import (
    animate_attitude_cube,
    plot_angular_velocity_body,
    plot_attitude_orientation,
    plot_attitude_orientation_lvlh,
    plot_attitude_quaternion,
    plot_attitude_quaternion_one_minus,
    plot_kalman_angular_velocity,
    plot_kalman_magnetometer_bias,
    plot_kalman_state_covariance,
    plot_kalman_state_error,
    plot_kalman_state_quaternion,
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
from simulation.io import build_results_dataframe, print_sanity_check, save_results
from simulation.pipeline.runner import SimulationRunner
from simulation.sensors import MagnetometerModel
from simulation.types import (
    AttitudeConfig,
    ClassicalOrbitalElements,
    KalmanFilterInput,
    SimulationConfig,
    SimulationResult,
)

Plotter = Callable[[pd.DataFrame, Path], None]
PlotOutput = tuple[str, Plotter]

ATTITUDE_AEKF_OUTPUT_DIR = Path("attitude_aekf")
AEKF_OUTPUT_DIR = Path("aekf")
KALMAN_CSV_FILENAME = "kalman_timeseries.csv"

GENERAL_PLOT_OUTPUTS: tuple[PlotOutput, ...] = (
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
    ("attitude_orientation_lvlh.png", plot_attitude_orientation_lvlh),
    ("attitude_quaternion.png", plot_attitude_quaternion),
    ("attitude_quaternion_one_minus.png", plot_attitude_quaternion_one_minus),
    ("angular_velocity_body.png", plot_angular_velocity_body),
)

KALMAN_PLOT_OUTPUTS: tuple[PlotOutput, ...] = (
    ("kalman_state_quaternion.png", plot_kalman_state_quaternion),
    ("kalman_state_error.png", plot_kalman_state_error),
    ("kalman_state_covariance.png", plot_kalman_state_covariance),
    ("kalman_angular_velocity.png", plot_kalman_angular_velocity),
    ("kalman_magnetometer_bias.png", plot_kalman_magnetometer_bias),
)
LEGACY_KALMAN_OUTPUT_FILENAMES = tuple(filename for filename, _ in KALMAN_PLOT_OUTPUTS) + (
    "kalman_gyro_bias.png",
)

PLOT_OUTPUTS: tuple[PlotOutput, ...] = GENERAL_PLOT_OUTPUTS + KALMAN_PLOT_OUTPUTS

ANIMATION_OUTPUTS: tuple[PlotOutput, ...] = (("attitude_cube.gif", animate_attitude_cube),)


def load_default_inputs() -> tuple[ClassicalOrbitalElements, SimulationConfig, AttitudeConfig]:
    """Load default orbit, simulation and attitude settings."""

    return (
        create_default_orbit(),
        create_default_simulation_config(),
        create_default_attitude_config(),
    )


def create_default_runner(attitude_config: AttitudeConfig | None = None) -> SimulationRunner:
    """Create the default simulation runner."""

    initial_omega_body_radps = (
        np.zeros(3, dtype=np.float64)
        if attitude_config is None
        else attitude_config.initial_omega_body_radps
    )

    return SimulationRunner(
        magnetometer_model=MagnetometerModel(
            bias_body_t=np.array([0.3e-6, -0.2e-6, 0.1e-6]), noise_std_t=1.0e-6, seed=42
        ),
        kalman_filter=AEKF(
            AEKFConfig(
                initial_omega_body_radps=initial_omega_body_radps,
                inertia_kg_m2=(
                    np.eye(3, dtype=np.float64)
                    if attitude_config is None
                    else attitude_config.inertia_kg_m2
                ),
                torque_body_nm=(
                    np.zeros(3, dtype=np.float64)
                    if attitude_config is None
                    else attitude_config.torque_body_nm
                ),
                measurement_noise=np.eye(3, dtype=np.float64) * (1.0e-6**2),
            )
        ),
    )


def build_attitude_aekf_dataframe(
    result: SimulationResult, attitude_config: AttitudeConfig
) -> pd.DataFrame:
    """Build a result table for the preserved quaternion-only AEKF baseline."""

    attitude_filter = QuaternionAEKF(
        QuaternionAEKFConfig(
            initial_quaternion_eci_from_body=attitude_config.initial_quaternion_eci_from_body,
            measurement_noise=np.eye(3, dtype=np.float64) * (1.0e-6**2),
        )
    )
    attitude_estimate = attitude_filter.estimate(
        KalmanFilterInput(
            t_s=result.orbit.t_s,
            measurements_body_t=result.b_magnetometer_t,
            reference_vectors_eci_t=result.magnetic_field.b_eci_t,
        )
    )
    attitude_result = replace(result, kalman_estimate=attitude_estimate)

    return build_results_dataframe(attitude_result)


def save_kalman_results(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save filter-specific time-series data to CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / KALMAN_CSV_FILENAME
    df.to_csv(output_path, index=False)

    return output_path


def cleanup_legacy_kalman_outputs(output_dir: Path) -> list[Path]:
    """Remove pre-folder Kalman plots from the root output directory."""

    removed_paths = []

    for filename in LEGACY_KALMAN_OUTPUT_FILENAMES:
        path = output_dir / filename

        if path.is_file():
            path.unlink()
            removed_paths.append(path)

    return removed_paths


def _save_plot_group(
    df: pd.DataFrame, output_dir: Path, plot_outputs: tuple[PlotOutput, ...]
) -> list[Path]:
    """Create a group of static plots and return paths that were produced."""

    paths = []

    for filename, plotter in plot_outputs:
        plotter(df, output_dir)
        path = output_dir / filename

        if path.exists():
            paths.append(path)

    return paths


def save_general_plot_outputs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create non-Kalman static plots and return their paths."""

    return _save_plot_group(df, output_dir, GENERAL_PLOT_OUTPUTS)


def save_kalman_plot_outputs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create Kalman static plots and return their paths."""

    return _save_plot_group(df, output_dir, KALMAN_PLOT_OUTPUTS)


def save_plot_outputs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create static plots and return their paths."""

    cleanup_legacy_kalman_outputs(output_dir)
    paths = save_general_plot_outputs(df, output_dir)
    paths.extend(save_kalman_plot_outputs(df, output_dir / AEKF_OUTPUT_DIR))

    return paths


def save_animation_outputs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create animations and return their paths."""

    paths = []

    for filename, plotter in ANIMATION_OUTPUTS:
        plotter(df, output_dir)
        paths.append(output_dir / filename)

    return paths


def print_saved_outputs(
    csv_path: Path,
    plot_paths: list[Path],
    animation_paths: list[Path],
    kalman_csv_paths: list[Path] | None = None,
) -> None:
    """Print generated output paths."""

    print(f"Saved orbit data to: {csv_path}")

    for path in kalman_csv_paths or []:
        print(f"Saved Kalman data: {path}")

    for path in plot_paths:
        print(f"Saved plot: {path}")

    for path in animation_paths:
        print(f"Saved animation: {path}")


def run_orbit_pipeline(output_dir: Path) -> None:
    """Run the orbit propagation pipeline."""

    elements, simulation_config, attitude_config = load_default_inputs()
    runner = create_default_runner(attitude_config)
    result = runner.run(elements, simulation_config, attitude_config)
    df = build_results_dataframe(result)
    attitude_aekf_df = build_attitude_aekf_dataframe(result, attitude_config)

    cleanup_legacy_kalman_outputs(output_dir)
    csv_path = save_results(df, output_dir)
    aekf_output_dir = output_dir / AEKF_OUTPUT_DIR
    attitude_aekf_output_dir = output_dir / ATTITUDE_AEKF_OUTPUT_DIR
    kalman_csv_paths = [
        save_kalman_results(attitude_aekf_df, attitude_aekf_output_dir),
        save_kalman_results(df, aekf_output_dir),
    ]
    plot_paths = save_general_plot_outputs(df, output_dir)
    plot_paths.extend(save_kalman_plot_outputs(attitude_aekf_df, attitude_aekf_output_dir))
    plot_paths.extend(save_kalman_plot_outputs(df, aekf_output_dir))
    animation_paths = save_animation_outputs(df, output_dir)

    print_saved_outputs(csv_path, plot_paths, animation_paths, kalman_csv_paths)
    print_sanity_check(df)
