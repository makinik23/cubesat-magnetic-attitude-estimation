"""
Main pipeline for the first milestone.

Milestone 1:
Classical orbital elements -> poliastro propagation -> ECI position and velocity
time series -> ECEF and geodetic coordinates -> IGRF magnetic field
vector -> CSV and plots.
"""

from pathlib import Path

import pandas as pd

from simulation.attitude import append_attitude_columns, create_default_attitude_config
from simulation.config import create_default_orbit, create_default_simulation_config
from simulation.frames import add_frame_columns
from simulation.geomagnetic import append_igrf_columns
from simulation.orbit_provider import propagate_orbit
from simulation.plots import (
    animate_attitude_cube,
    plot_attitude_orientation,
    plot_magnetic_field_body,
    plot_magnetic_field_body_norm,
    plot_magnetic_field_eci,
    plot_magnetic_field_norm,
    plot_orbit_3d,
    plot_position_eci,
    plot_position_norm,
    plot_r_eci_time,
    plot_velocity_norm,
)
from simulation.visualisation import create_cesium_visualisation


def save_orbit_results(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Save orbit propagation results to CSV.

    Parameters
    ----------
    df:
        Orbit propagation results.
    output_dir:
        Output directory.

    Returns
    -------
    Path
        Path to saved CSV file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "orbit_timeseries.csv"
    df.to_csv(output_path, index=False)

    return output_path


def print_sanity_check(df: pd.DataFrame) -> None:
    """
    Print basic sanity check values.
    """

    print()
    print("Sanity check:")
    print(f"Initial radius: {df['r_norm_m'].iloc[0] / 1000.0:.3f} km")
    print(f"Mean radius:    {df['r_norm_m'].mean() / 1000.0:.3f} km")
    print(f"Min radius:     {df['r_norm_m'].min() / 1000.0:.3f} km")
    print(f"Max radius:     {df['r_norm_m'].max() / 1000.0:.3f} km")
    print(f"Mean velocity:  {df['v_norm_mps'].mean() / 1000.0:.3f} km/s")

    if "det_R_eci_from_body" not in df.columns:
        return

    orthogonality_columns = [
        f"RT_R_minus_I_{row}{col}" for row in range(1, 4) for col in range(1, 4)
    ]
    initial_orthogonality_error = df[orthogonality_columns].iloc[0].to_numpy().reshape(3, 3)
    final_orthogonality_error = df[orthogonality_columns].iloc[-1].to_numpy().reshape(3, 3)

    print()
    print("Attitude sanity check:")
    print("Initial R^T R - I:")
    for row in initial_orthogonality_error:
        print("  [" + " ".join(f"{value: .3e}" for value in row) + "]")
    print("Final R^T R - I:")
    for row in final_orthogonality_error:
        print("  [" + " ".join(f"{value: .3e}" for value in row) + "]")
    print(f"Max ||R^T R - I||_F: {df['RT_R_minus_I_fro'].max():.3e}")
    print(f"det(R) initial:      {df['det_R_eci_from_body'].iloc[0]:.12f}")
    print(f"det(R) final:        {df['det_R_eci_from_body'].iloc[-1]:.12f}")
    print(
        "det(R) min/max:      "
        f"{df['det_R_eci_from_body'].min():.12f} / {df['det_R_eci_from_body'].max():.12f}"
    )


def run_orbit_pipeline(output_dir: Path) -> None:
    """
    Run the orbit propagation pipeline.
    """

    elements = create_default_orbit()
    simulation_config = create_default_simulation_config()
    attitude_config = create_default_attitude_config()

    df = propagate_orbit(elements=elements, config=simulation_config)
    df = add_frame_columns(df)
    df = append_igrf_columns(df)
    df = append_attitude_columns(df, config=attitude_config)

    csv_path = save_orbit_results(df, output_dir)
    czml_path, viewer_path = create_cesium_visualisation(df, output_dir / "visualisation")

    plot_position_eci(df, output_dir)
    plot_r_eci_time(df, output_dir)
    plot_position_norm(df, output_dir)
    plot_velocity_norm(df, output_dir)
    plot_orbit_3d(df, output_dir)
    plot_magnetic_field_eci(df, output_dir)
    plot_magnetic_field_norm(df, output_dir)
    plot_magnetic_field_body(df, output_dir)
    plot_magnetic_field_body_norm(df, output_dir)
    plot_attitude_orientation(df, output_dir)
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
    print(f"Saved plot: {output_dir / 'attitude_orientation.png'}")
    print(f"Saved animation: {output_dir / 'attitude_cube.gif'}")
    print(f"Saved Cesium CZML: {czml_path}")
    print(f"Saved Cesium viewer: {viewer_path}")

    print_sanity_check(df)
