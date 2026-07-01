"""
IGRF magnetic-field provider.

This module uses ppigrf as a ready-made IGRF implementation. The project focus
is the estimator, so the geomagnetic model is intentionally treated as an
external reference component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ppigrf

from simulation.frames import ecef_vectors_to_eci, ned_to_ecef_vectors
from simulation.types import FrameState, MagneticFieldState, OrbitState


def compute_igrf_ned_nt(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray, time_utc: np.ndarray
) -> np.ndarray:
    """
    Compute IGRF magnetic field in local NED coordinates.

    Returns
    -------
    np.ndarray
        Magnetic-field vectors [nT] in NED coordinates, shape (N, 3).
    """

    b_ned_nt = np.empty((len(lat_deg), 3), dtype=float)

    for idx, (lat, lon, alt, time_iso) in enumerate(zip(lat_deg, lon_deg, alt_m, time_utc)):
        date = pd.Timestamp(str(time_iso)).to_pydatetime()
        alt_km = alt / 1000.0

        # ppigrf.igrf returns east, north and up components in nT.
        b_east_nt, b_north_nt, b_up_nt = ppigrf.igrf(lon, lat, alt_km, date)

        b_ned_nt[idx] = np.array(
            [
                float(np.asarray(b_north_nt).reshape(-1)[0]),
                float(np.asarray(b_east_nt).reshape(-1)[0]),
                -float(np.asarray(b_up_nt).reshape(-1)[0]),
            ]
        )

    return b_ned_nt


def compute_magnetic_field_state(orbit: OrbitState, frame: FrameState) -> MagneticFieldState:
    """Compute geomagnetic-field vectors in NED, ECEF and ECI frames."""

    b_ned_nt = compute_igrf_ned_nt(frame.lat_deg, frame.lon_deg, frame.alt_m, orbit.t_utc)
    b_ned_t = b_ned_nt * 1e-9
    b_ecef_t = ned_to_ecef_vectors(b_ned_t, frame.lat_deg, frame.lon_deg)
    b_eci_t = ecef_vectors_to_eci(b_ecef_t, frame.r_ecef_m, orbit.t_utc)

    return MagneticFieldState(b_ned_nt=b_ned_nt, b_ecef_t=b_ecef_t, b_eci_t=b_eci_t)


def append_igrf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append IGRF magnetic-field columns in NED, ECEF and ECI frames."""

    lat_deg = df["lat_deg"].to_numpy()
    lon_deg = df["lon_deg"].to_numpy()
    alt_m = df["alt_m"].to_numpy()
    time_utc = df["t_utc"].to_numpy()
    r_ecef_m = df[["x_ecef_m", "y_ecef_m", "z_ecef_m"]].to_numpy()

    b_ned_nt = compute_igrf_ned_nt(lat_deg, lon_deg, alt_m, time_utc)
    b_ned_t = b_ned_nt * 1e-9
    b_ecef_t = ned_to_ecef_vectors(b_ned_t, lat_deg, lon_deg)
    b_eci_t = ecef_vectors_to_eci(b_ecef_t, r_ecef_m, time_utc)

    df = df.copy()
    df["B_N_nT"] = b_ned_nt[:, 0]
    df["B_E_nT"] = b_ned_nt[:, 1]
    df["B_D_nT"] = b_ned_nt[:, 2]
    df["Bx_ecef_T"] = b_ecef_t[:, 0]
    df["By_ecef_T"] = b_ecef_t[:, 1]
    df["Bz_ecef_T"] = b_ecef_t[:, 2]
    df["Bx_eci_T"] = b_eci_t[:, 0]
    df["By_eci_T"] = b_eci_t[:, 1]
    df["Bz_eci_T"] = b_eci_t[:, 2]
    df["B_norm_T"] = np.linalg.norm(b_eci_t, axis=1)
    df["B_norm_nT"] = df["B_norm_T"] * 1e9

    return df
