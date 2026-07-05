"""
Main simulation pipeline orchestration.
"""

from pathlib import Path

import numpy as np

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


def run_orbit_pipeline(output_dir: Path) -> None:
    """Run the orbit propagation pipeline."""

    elements = create_default_orbit()
    simulation_config = create_default_simulation_config()
    attitude_config = create_default_attitude_config()

    runner = SimulationRunner(
        magnetometer_model=MagnetometerModel(
            bias_body_t=np.array([0.3e-6, -0.2e-6, 0.1e-6]), noise_std_t=1.0e-6, seed=42
        )
    )
    result = runner.run(elements, simulation_config, attitude_config)
    df = build_results_dataframe(result)

    csv_path = save_results(df, output_dir)

    plot_position_eci(df, output_dir)
    plot_r_eci_time(df, output_dir)
    plot_position_norm(df, output_dir)
    plot_velocity_norm(df, output_dir)
    plot_orbit_3d(df, output_dir)
    plot_magnetic_field_eci(df, output_dir)
    plot_magnetic_field_norm(df, output_dir)
    plot_magnetic_field_body(df, output_dir)
    plot_magnetic_field_body_norm(df, output_dir)
    plot_magnetometer_measurement(df, output_dir)
    plot_attitude_orientation(df, output_dir)
    plot_attitude_quaternion(df, output_dir)
    plot_angular_velocity_body(df, output_dir)
    animate_attitude_cube(df, output_dir)

    print(f"Saved orbit data to: {csv_path}")
    print(f"Saved plot: {output_dir / 'position_eci.png'}")
    print(f"Saved plot: {output_dir / 'r_eci_time.png'}")
    print(f"Saved plot: {output_dir / 'position_norm.png'}")
    print(f"Saved plot: {output_dir / 'velocity_norm.png'}")
    print(f"Saved plot: {output_dir / 'orbit_3d.png'}")
    print(f"Saved plot: {output_dir / 'magnetic_field_eci.png'}")
    print(f"Saved plot: {output_dir / 'magnetic_field_norm.png'}")
    print(f"Saved plot: {output_dir / 'magnetic_field_body.png'}")
    print(f"Saved plot: {output_dir / 'magnetic_field_body_norm.png'}")
    print(f"Saved plot: {output_dir / 'magnetometer_measurement.png'}")
    print(f"Saved plot: {output_dir / 'attitude_orientation.png'}")
    print(f"Saved plot: {output_dir / 'attitude_quaternion.png'}")
    print(f"Saved plot: {output_dir / 'angular_velocity_body.png'}")
    print(f"Saved animation: {output_dir / 'attitude_cube.gif'}")

    print_sanity_check(df)
