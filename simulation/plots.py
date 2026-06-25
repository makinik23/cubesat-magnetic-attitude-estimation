"""Plotting utilities for orbit and frame-conversion results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
