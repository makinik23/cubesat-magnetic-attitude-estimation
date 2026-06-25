"""
Reference-frame and geodetic conversion utilities.

The current pipeline assumes that the inertial vectors produced by poliastro
can be interpreted as GCRS-like ECI vectors for this early project milestone.
This assumption is good enough for the first engineering plots, but should be
revisited if the project later requires high-fidelity frame consistency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import astropy.units as u
import numpy as np
import pymap3d as pm
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers


def _configure_iers_offline() -> None:
    """Keep Astropy frame transforms deterministic without network access."""

    iers.conf.auto_download = False


def _as_time_array(time_utc: Any) -> Time:
    """Convert UTC timestamps to an astropy Time array."""

    _configure_iers_offline()

    if isinstance(time_utc, Time):
        return time_utc

    time_values = np.atleast_1d(time_utc)

    if np.issubdtype(time_values.dtype, np.datetime64):
        return Time(time_values, format="datetime64", scale="utc")

    values = time_values.tolist()
    if not isinstance(values, list):
        values = [values]

    if values and isinstance(values[0], datetime):
        return Time(values, scale="utc")

    return Time([str(value) for value in values], format="isot", scale="utc")


def _as_position_array(r_m: Any, name: str) -> np.ndarray:
    """Convert position values to a float64 array with shape (N, 3)."""

    array = np.asarray(r_m, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")

    return array


def _check_vector_inputs(vectors: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> None:
    """Validate vector and coordinate array shapes."""

    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3).")

    if lat_deg.shape != lon_deg.shape or lat_deg.shape[0] != vectors.shape[0]:
        raise ValueError("Latitude and longitude arrays must match vector length.")


def eci_to_ecef_positions(r_eci_m: np.ndarray, time_utc: np.ndarray) -> np.ndarray:
    """
    Convert ECI/GCRS-like position vectors to ECEF/ITRS positions.

    Parameters
    ----------
    r_eci_m:
        Position vectors in ECI [m], shape (N, 3).
    time_utc:
        UTC timestamps compatible with astropy Time, shape (N,).

    Returns
    -------
    np.ndarray
        Position vectors in ECEF/ITRS [m], shape (N, 3).
    """

    r_eci_m = _as_position_array(r_eci_m, "r_eci_m")
    times = _as_time_array(time_utc)

    if len(times) != len(r_eci_m):
        raise ValueError("time_utc must have the same length as r_eci_m.")

    representation = CartesianRepresentation(
        x=r_eci_m[:, 0] * u.m, y=r_eci_m[:, 1] * u.m, z=r_eci_m[:, 2] * u.m
    )

    gcrs = GCRS(representation, obstime=times)
    itrs = gcrs.transform_to(ITRS(obstime=times))
    cart = itrs.cartesian

    return np.column_stack([cart.x.to_value(u.m), cart.y.to_value(u.m), cart.z.to_value(u.m)])


def ecef_to_geodetic(r_ecef_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECEF positions to geodetic latitude, longitude and altitude.

    Returns latitude and longitude in degrees, altitude in meters.
    """

    r_ecef_m = _as_position_array(r_ecef_m, "r_ecef_m")

    lat_deg, lon_deg, alt_m = pm.ecef2geodetic(
        r_ecef_m[:, 0], r_ecef_m[:, 1], r_ecef_m[:, 2], deg=True
    )

    return (
        np.asarray(lat_deg, dtype=np.float64),
        np.asarray(lon_deg, dtype=np.float64),
        np.asarray(alt_m, dtype=np.float64),
    )


def append_ecef_and_geodetic_columns(df):
    """Append ECEF position and geodetic coordinates to an orbit DataFrame."""

    r_eci_m = df[["x_eci_m", "y_eci_m", "z_eci_m"]].to_numpy()
    time_utc = df["t_utc"].to_numpy()

    r_ecef_m = eci_to_ecef_positions(r_eci_m, time_utc)
    lat_deg, lon_deg, alt_m = ecef_to_geodetic(r_ecef_m)

    df = df.copy()
    df["x_ecef_m"] = r_ecef_m[:, 0]
    df["y_ecef_m"] = r_ecef_m[:, 1]
    df["z_ecef_m"] = r_ecef_m[:, 2]
    df["lat_deg"] = lat_deg
    df["lon_deg"] = lon_deg
    df["alt_m"] = alt_m
    df["lat_rad"] = np.deg2rad(lat_deg)
    df["lon_rad"] = np.deg2rad(lon_deg)
    df["alt_km"] = alt_m / 1000.0

    return df


def ned_to_ecef_vectors(b_ned: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """
    Convert vectors from local NED coordinates to ECEF coordinates.

    Parameters
    ----------
    b_ned:
        Vectors in NED coordinates, shape (N, 3).
    lat_deg:
        Geodetic latitude [deg], shape (N,).
    lon_deg:
        Geodetic longitude [deg], shape (N,).

    Returns
    -------
    np.ndarray
        Vectors in ECEF coordinates, shape (N, 3).
    """

    b_ned = np.asarray(b_ned, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_deg = np.asarray(lon_deg, dtype=np.float64)

    _check_vector_inputs(b_ned, lat_deg, lon_deg)

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    b_ecef = np.empty_like(b_ned, dtype=float)

    for idx, (vector_ned, lat, lon) in enumerate(zip(b_ned, lat_rad, lon_rad)):
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        east = np.array([-sin_lon, cos_lon, 0.0])
        down = np.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat])

        b_ecef[idx] = vector_ned[0] * north + vector_ned[1] * east + vector_ned[2] * down

    return b_ecef


def ecef_vectors_to_eci(
    vectors_ecef: np.ndarray, r_ecef_m: np.ndarray, time_utc: np.ndarray
) -> np.ndarray:
    """
    Convert vectors from ECEF/ITRS to ECI/GCRS-like coordinates.

    Astropy handles coordinates rather than free vectors directly here, so this
    function estimates the local rotation by transforming a base point and three
    one-meter displaced points. This is slower than using a precomputed rotation
    matrix, but it is explicit and robust enough for this early milestone.
    """

    vectors_ecef = np.asarray(vectors_ecef, dtype=np.float64)
    r_ecef_m = _as_position_array(r_ecef_m, "r_ecef_m")
    times = _as_time_array(time_utc)

    if vectors_ecef.shape != r_ecef_m.shape:
        raise ValueError("vectors_ecef and r_ecef_m must both have shape (N, 3).")

    if len(times) != len(vectors_ecef):
        raise ValueError("time_utc must have the same length as vectors_ecef.")

    vectors_eci = np.empty_like(vectors_ecef, dtype=float)

    basis_ecef = np.eye(3)

    for idx, (vector_ecef, position_ecef, time) in enumerate(zip(vectors_ecef, r_ecef_m, times)):
        base_itrs = ITRS(
            CartesianRepresentation(
                position_ecef[0] * u.m, position_ecef[1] * u.m, position_ecef[2] * u.m
            ),
            obstime=time,
        )
        base_gcrs = base_itrs.transform_to(GCRS(obstime=time)).cartesian
        base_eci = np.array(
            [base_gcrs.x.to_value(u.m), base_gcrs.y.to_value(u.m), base_gcrs.z.to_value(u.m)]
        )

        rotation_eci_from_ecef = np.empty((3, 3), dtype=float)

        for col, basis_vector in enumerate(basis_ecef):
            shifted = position_ecef + basis_vector
            shifted_itrs = ITRS(
                CartesianRepresentation(shifted[0] * u.m, shifted[1] * u.m, shifted[2] * u.m),
                obstime=time,
            )
            shifted_gcrs = shifted_itrs.transform_to(GCRS(obstime=time)).cartesian
            shifted_eci = np.array(
                [
                    shifted_gcrs.x.to_value(u.m),
                    shifted_gcrs.y.to_value(u.m),
                    shifted_gcrs.z.to_value(u.m),
                ]
            )
            rotation_eci_from_ecef[:, col] = shifted_eci - base_eci

        vectors_eci[idx] = rotation_eci_from_ecef @ vector_ecef

    return vectors_eci


def add_frame_columns(df):
    """Backward-compatible alias for appending ECEF and geodetic columns."""

    return append_ecef_and_geodetic_columns(df)
