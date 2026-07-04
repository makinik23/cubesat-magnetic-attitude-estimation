"""Plotting utilities for orbit and frame-conversion results."""

from pathlib import Path

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulation.attitude import quaternion_to_rotation_matrix


def plot_position_eci(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot ECI position components over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["x_eci_m"] / 1000.0, label="x ECI")
    plt.plot(df["t_s"], df["y_eci_m"] / 1000.0, label="y ECI")
    plt.plot(df["t_s"], df["z_eci_m"] / 1000.0, label="z ECI")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [km]")
    plt.title("ECI position components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_eci.png", dpi=200)
    plt.close()


def plot_r_eci_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot r_eci vector components over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["x_eci_m"] / 1000.0, label="r_x ECI")
    plt.plot(df["t_s"], df["y_eci_m"] / 1000.0, label="r_y ECI")
    plt.plot(df["t_s"], df["z_eci_m"] / 1000.0, label="r_z ECI")
    plt.xlabel("Time [s]")
    plt.ylabel("r_eci [km]")
    plt.title("r_eci vector components over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "r_eci_time.png", dpi=200)
    plt.close()


def plot_position_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the ECI position vector over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["r_norm_m"] / 1000.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Radius [km]")
    plt.title("Position norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "position_norm.png", dpi=200)
    plt.close()


def plot_velocity_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the ECI velocity vector over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["v_norm_mps"] / 1000.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [km/s]")
    plt.title("Velocity norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_norm.png", dpi=200)
    plt.close()


def plot_orbit_3d(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 3D orbit trajectory in the ECI frame."""

    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(df["x_eci_m"] / 1000.0, df["y_eci_m"] / 1000.0, df["z_eci_m"] / 1000.0)

    ax.set_xlabel("x ECI [km]")
    ax.set_ylabel("y ECI [km]")
    ax.set_zlabel("z ECI [km]")
    ax.set_title("Orbit trajectory in ECI")

    plt.tight_layout()
    plt.savefig(output_dir / "orbit_3d.png", dpi=200)
    plt.close()


def plot_ground_track(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot latitude and longitude over time after ECEF/geodetic conversion."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["lat_deg"], label="latitude")
    plt.plot(df["t_s"], df["lon_deg"], label="longitude")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Geodetic latitude and longitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ground_track_timeseries.png", dpi=200)
    plt.close()


def plot_altitude(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot geodetic altitude over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["alt_m"] / 1000.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [km]")
    plt.title("Geodetic altitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "altitude.png", dpi=200)
    plt.close()


def plot_magnetic_field_eci(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot IGRF magnetic-field components in ECI coordinates."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["Bx_eci_T"] * 1e6, label="Bx ECI")
    plt.plot(df["t_s"], df["By_eci_T"] * 1e6, label="By ECI")
    plt.plot(df["t_s"], df["Bz_eci_T"] * 1e6, label="Bz ECI")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnetic field [uT]")
    plt.title("IGRF magnetic field in ECI")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "magnetic_field_eci.png", dpi=200)
    plt.close()


def plot_magnetic_field_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the IGRF magnetic-field vector."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["B_norm_T"] * 1e6)
    plt.xlabel("Time [s]")
    plt.ylabel("|B| [uT]")
    plt.title("IGRF magnetic-field norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "magnetic_field_norm.png", dpi=200)
    plt.close()


def plot_magnetic_field_body(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot IGRF magnetic-field components in body coordinates."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["Bx_body_T"] * 1e6, label="Bx body")
    plt.plot(df["t_s"], df["By_body_T"] * 1e6, label="By body")
    plt.plot(df["t_s"], df["Bz_body_T"] * 1e6, label="Bz body")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnetic field [uT]")
    plt.title("IGRF magnetic field in body")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "magnetic_field_body.png", dpi=200)
    plt.close()


def plot_magnetic_field_body_norm(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot norm of the body-frame IGRF magnetic-field vector."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["B_body_norm_T"] * 1e6)
    plt.xlabel("Time [s]")
    plt.ylabel("|B_body| [uT]")
    plt.title("Body magnetic-field norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "magnetic_field_body_norm.png", dpi=200)
    plt.close()


def plot_attitude_orientation(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot body orientation angles over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], df["yaw_eci_from_body_deg"], label="yaw")
    plt.plot(df["t_s"], df["pitch_eci_from_body_deg"], label="pitch")
    plt.plot(df["t_s"], df["roll_eci_from_body_deg"], label="roll")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Body orientation over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "attitude_orientation.png", dpi=200)
    plt.close()


def plot_attitude_quaternion(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot attitude quaternion components and norm over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(df["t_s"], df["q_eci_from_body_w"], label="q_w")
    axes[0].plot(df["t_s"], df["q_eci_from_body_x"], label="q_x")
    axes[0].plot(df["t_s"], df["q_eci_from_body_y"], label="q_y")
    axes[0].plot(df["t_s"], df["q_eci_from_body_z"], label="q_z")
    axes[0].set_ylabel("Quaternion component [-]")
    axes[0].set_title("Attitude quaternion over time")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(df["t_s"], df["q_eci_from_body_norm"], label="|q|")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Quaternion norm [-]")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "attitude_quaternion.png", dpi=200)
    plt.close(fig)


def plot_angular_velocity_body(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot body-frame angular velocity components over time."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_s"], np.rad2deg(df["omega_body_x_radps"]), label="omega_x body")
    plt.plot(df["t_s"], np.rad2deg(df["omega_body_y_radps"]), label="omega_y body")
    plt.plot(df["t_s"], np.rad2deg(df["omega_body_z_radps"]), label="omega_z body")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity [deg/s]")
    plt.title("Body angular velocity components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "angular_velocity_body.png", dpi=200)
    plt.close()


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
        body_axes_eci = np.eye(3) @ rotation_eci_from_body.T
        axis_colors = ["tab:red", "tab:green", "tab:blue"]
        axis_labels = ["+X body", "+Y body", "+Z body"]

        for axis_vector, color, label in zip(body_axes_eci, axis_colors, axis_labels):
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
