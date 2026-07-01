"""
Main simulation pipeline orchestration.
"""

from pathlib import Path

from simulation.attitude import project_eci_vectors_to_body, propagate_attitude
from simulation.config import (
    create_default_attitude_config,
    create_default_orbit,
    create_default_simulation_config,
)
from simulation.frames import compute_frame_state
from simulation.geomagnetic import compute_magnetic_field_state
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
from simulation.results import build_results_dataframe, print_sanity_check, save_results


def run_orbit_pipeline(output_dir: Path) -> None:
    """
    Run the orbit propagation pipeline.
    """

    elements = create_default_orbit()
    simulation_config = create_default_simulation_config()
    attitude_config = create_default_attitude_config()

    orbit = propagate_orbit(elements=elements, config=simulation_config)
    frame = compute_frame_state(orbit)
    magnetic_field = compute_magnetic_field_state(orbit, frame)
    attitude = propagate_attitude(orbit.t_s, attitude_config)
    b_body_t = project_eci_vectors_to_body(magnetic_field.b_eci_t, attitude)
    df = build_results_dataframe(orbit, frame, magnetic_field, attitude, b_body_t)

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

    print_sanity_check(df)
