"""Plotting utilities for simulation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulation.attitude import quaternion_to_rotation_matrix

Series = tuple[str, str, float]


def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _plot_time_series(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    series: list[Series],
    ylabel: str,
    title: str,
    *,
    xlabel: str = "Time [s]",
) -> None:
    fig, ax = plt.subplots()

    for column, label, scale in series:
        ax.plot(df["t_s"], df[column] * scale, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    if len(series) > 1:
        ax.legend()

    _save_figure(fig, output_dir, filename)


def plot_position_eci(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot ECI position components over time."""

    _plot_time_series(
        df,
        output_dir,
        "position_eci.png",
        [("x_eci_m", "x ECI", 1.0e-3), ("y_eci_m", "y ECI", 1.0e-3), ("z_eci_m", "z ECI", 1.0e-3)],
        "Position [km]",
        "ECI position components",
    )


def plot_r_eci_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot r_eci vector components over time."""

    _plot_time_series(
        df,
        output_dir,
        "r_eci_time.png",
        [
            ("x_eci_m", "r_x ECI", 1.0e-3),
            ("y_eci_m", "r_y ECI", 1.0e-3),
            ("z_eci_m", "r_z ECI", 1.0e-3),
        ],
        "r_eci [km]",
        "r_eci vector components over time",
    )


def plot_position_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the ECI position vector over time."""

    _plot_time_series(
        df,
        output_dir,
        "position_norm.png",
        [("r_norm_m", "|r|", 1.0e-3)],
        "Radius [km]",
        "Position norm",
    )


def plot_velocity_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the ECI velocity vector over time."""

    _plot_time_series(
        df,
        output_dir,
        "velocity_norm.png",
        [("v_norm_mps", "|v|", 1.0e-3)],
        "Velocity [km/s]",
        "Velocity norm",
    )


def plot_orbit_3d(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 3D orbit trajectory in the ECI frame."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(df["x_eci_m"] / 1000.0, df["y_eci_m"] / 1000.0, df["z_eci_m"] / 1000.0)
    ax.set_xlabel("x ECI [km]")
    ax.set_ylabel("y ECI [km]")
    ax.set_zlabel("z ECI [km]")
    ax.set_title("Orbit trajectory in ECI")

    _save_figure(fig, output_dir, "orbit_3d.png")


def plot_ground_track(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot latitude and longitude over time after ECEF/geodetic conversion."""

    _plot_time_series(
        df,
        output_dir,
        "ground_track_timeseries.png",
        [("lat_deg", "latitude", 1.0), ("lon_deg", "longitude", 1.0)],
        "Angle [deg]",
        "Geodetic latitude and longitude",
    )


def plot_altitude(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot geodetic altitude over time."""

    _plot_time_series(
        df,
        output_dir,
        "altitude.png",
        [("alt_m", "altitude", 1.0e-3)],
        "Altitude [km]",
        "Geodetic altitude",
    )


def plot_magnetic_field_eci(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot IGRF magnetic-field components in ECI coordinates."""

    _plot_time_series(
        df,
        output_dir,
        "magnetic_field_eci.png",
        [
            ("Bx_eci_T", "Bx ECI", 1.0e6),
            ("By_eci_T", "By ECI", 1.0e6),
            ("Bz_eci_T", "Bz ECI", 1.0e6),
        ],
        "Magnetic field [uT]",
        "IGRF magnetic field in ECI",
    )


def plot_magnetic_field_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the IGRF magnetic-field vector."""

    _plot_time_series(
        df,
        output_dir,
        "magnetic_field_norm.png",
        [("B_norm_T", "|B|", 1.0e6)],
        "|B| [uT]",
        "IGRF magnetic-field norm",
    )


def plot_magnetic_field_body(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot IGRF magnetic-field components in body coordinates."""

    _plot_time_series(
        df,
        output_dir,
        "magnetic_field_body.png",
        [
            ("Bx_body_T", "Bx body", 1.0e6),
            ("By_body_T", "By body", 1.0e6),
            ("Bz_body_T", "Bz body", 1.0e6),
        ],
        "Magnetic field [uT]",
        "IGRF magnetic field in body",
    )


def plot_magnetic_field_body_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the body-frame IGRF magnetic-field vector."""

    _plot_time_series(
        df,
        output_dir,
        "magnetic_field_body_norm.png",
        [("B_body_norm_T", "|B_body|", 1.0e6)],
        "|B_body| [uT]",
        "Body magnetic-field norm",
    )


def plot_magnetometer_measurement(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot magnetometer measurements in body coordinates."""

    _plot_time_series(
        df,
        output_dir,
        "magnetometer_measurement.png",
        [
            ("Bx_magnetometer_T", "Bx measured", 1.0e6),
            ("By_magnetometer_T", "By measured", 1.0e6),
            ("Bz_magnetometer_T", "Bz measured", 1.0e6),
        ],
        "Magnetometer [uT]",
        "Magnetometer measurement in body",
    )


def plot_attitude_orientation(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot body orientation angles over time."""

    _plot_time_series(
        df,
        output_dir,
        "attitude_orientation.png",
        [
            ("yaw_eci_from_body_deg", "yaw", 1.0),
            ("pitch_eci_from_body_deg", "pitch", 1.0),
            ("roll_eci_from_body_deg", "roll", 1.0),
        ],
        "Angle [deg]",
        "Body orientation over time",
    )


def plot_attitude_quaternion(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot attitude quaternion components and norm over time."""

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    for column, label in [
        ("q_eci_from_body_w", "q_w"),
        ("q_eci_from_body_x", "q_x"),
        ("q_eci_from_body_y", "q_y"),
        ("q_eci_from_body_z", "q_z"),
    ]:
        axes[0].plot(df["t_s"], df[column], label=label)

    axes[0].set_ylabel("Quaternion component [-]")
    axes[0].set_title("Attitude quaternion over time")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df["t_s"], df["q_eci_from_body_norm"], label="|q|")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Quaternion norm [-]")
    axes[1].grid(True)
    axes[1].legend()

    _save_figure(fig, output_dir, "attitude_quaternion.png")


def plot_angular_velocity_body(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot body-frame angular velocity components over time."""

    deg_per_rad = 180.0 / np.pi
    _plot_time_series(
        df,
        output_dir,
        "angular_velocity_body.png",
        [
            ("omega_body_x_radps", "omega_x body", deg_per_rad),
            ("omega_body_y_radps", "omega_y body", deg_per_rad),
            ("omega_body_z_radps", "omega_z body", deg_per_rad),
        ],
        "Angular velocity [deg/s]",
        "Body angular velocity components",
    )


def _cube_vertices() -> np.ndarray:
    """Create unit cube vertices in body coordinates."""

    return np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def _cube_faces(vertices: np.ndarray) -> list[np.ndarray]:
    """Create cube faces from transformed vertices."""

    face_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]

    return [vertices[indices] for indices in face_indices]


def animate_attitude_cube(df: pd.DataFrame, output_dir: Path, max_frames: int = 120) -> None:
    """Create a GIF animation of the body-frame cube attitude."""

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = min(max_frames, len(df))
    frame_indices = np.linspace(0, len(df) - 1, frame_count, dtype=int)
    cube_body = _cube_vertices()

    quaternion_columns = [
        "q_eci_from_body_w",
        "q_eci_from_body_x",
        "q_eci_from_body_y",
        "q_eci_from_body_z",
    ]
    quaternions = df[quaternion_columns].to_numpy(dtype=np.float64)
    times_s = df["t_s"].to_numpy(dtype=np.float64)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(frame_index: int) -> list[Poly3DCollection]:
        ax.clear()

        rotation_eci_from_body = quaternion_to_rotation_matrix(quaternions[frame_index])
        cube_eci = cube_body @ rotation_eci_from_body.T
        faces = _cube_faces(cube_eci)

        collection = Poly3DCollection(
            faces, facecolors="tab:blue", edgecolors="black", linewidths=0.8, alpha=0.35
        )
        ax.add_collection3d(collection)

        axis_length = 1.6
        axis_colors = ["tab:red", "tab:green", "tab:blue"]
        axis_labels = ["+X body", "+Y body", "+Z body"]

        for axis_vector, color, label in zip(rotation_eci_from_body.T, axis_colors, axis_labels):
            ax.plot(
                [0.0, axis_length * axis_vector[0]],
                [0.0, axis_length * axis_vector[1]],
                [0.0, axis_length * axis_vector[2]],
                color=color,
                linewidth=2.0,
                label=label,
            )

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_zlim(-1.8, 1.8)
        ax.set_xlabel("ECI X")
        ax.set_ylabel("ECI Y")
        ax.set_zlabel("ECI Z")
        ax.set_title(f"Body attitude, t = {times_s[frame_index]:.1f} s")
        ax.legend(loc="upper left")
        ax.view_init(elev=25.0, azim=35.0)
        ax.set_box_aspect((1.0, 1.0, 1.0))

        return [collection]

    cube_animation = animation.FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=80, blit=False
    )
    cube_animation.save(output_dir / "attitude_cube.gif", writer=animation.PillowWriter(fps=15))
    plt.close(fig)
