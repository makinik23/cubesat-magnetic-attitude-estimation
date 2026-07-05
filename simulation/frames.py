"""
Reference-frame and geodetic conversion utilities.

The current pipeline delegates ECI/ECEF transforms to pymap3d. With Astropy
available, pymap3d uses the GCRS <-> ITRS transform path instead of the lower
accuracy Vallado/GMST-only fallback.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pymap3d as pm

from simulation.helpers import as_datetime_array, as_float_vector_array
from simulation.types import FrameState, OrbitState


def _pymap3d_astropy_kwargs(function: Any) -> dict[str, bool]:
    """Return keyword arguments that select pymap3d's Astropy-backed path."""

    parameters = inspect.signature(function).parameters

    if "force_non_astropy" in parameters:
        return {"force_non_astropy": False}

    if "useastropy" in parameters:
        return {"useastropy": True}

    return {}


def _check_coordinate_inputs(vectors: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> None:
    """Validate geodetic coordinate array shapes against vector length."""

    if lat_deg.ndim != 1 or lon_deg.ndim != 1:
        raise ValueError("Latitude and longitude arrays must be one-dimensional.")

    if lat_deg.shape != lon_deg.shape or lat_deg.shape[0] != vectors.shape[0]:
        raise ValueError("Latitude and longitude arrays must match vector length.")


def eci_to_ecef_positions(r_eci_m: np.ndarray, time_utc: Any) -> np.ndarray:
    """
    Convert ECI position vectors to ECEF positions via pymap3d.

    Parameters
    ----------
    r_eci_m:
        Position vectors in ECI [m], shape (N, 3).
    time_utc:
        UTC timestamps compatible with astropy Time, shape (N,).

    Returns
    -------
    np.ndarray
        Position vectors in ECEF [m], shape (N, 3).
    """

    r_eci_m = as_float_vector_array(r_eci_m, "r_eci_m")
    times = as_datetime_array(time_utc)

    if len(times) != len(r_eci_m):
        raise ValueError("time_utc must have the same length as r_eci_m.")

    x_ecef, y_ecef, z_ecef = pm.eci2ecef(
        r_eci_m[:, 0], r_eci_m[:, 1], r_eci_m[:, 2], times, **_pymap3d_astropy_kwargs(pm.eci2ecef)
    )

    return np.column_stack(
        [
            np.atleast_1d(np.asarray(x_ecef, dtype=np.float64)),
            np.atleast_1d(np.asarray(y_ecef, dtype=np.float64)),
            np.atleast_1d(np.asarray(z_ecef, dtype=np.float64)),
        ]
    )


def ecef_to_geodetic(r_ecef_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECEF positions to geodetic latitude, longitude and altitude.

    Returns latitude and longitude in degrees, altitude in meters.
    """

    r_ecef_m = as_float_vector_array(r_ecef_m, "r_ecef_m")

    lat_deg, lon_deg, alt_m = pm.ecef2geodetic(
        r_ecef_m[:, 0], r_ecef_m[:, 1], r_ecef_m[:, 2], deg=True
    )

    return (
        np.asarray(lat_deg, dtype=np.float64),
        np.asarray(lon_deg, dtype=np.float64),
        np.asarray(alt_m, dtype=np.float64),
    )


def compute_frame_state(orbit: OrbitState) -> FrameState:
    """Compute ECEF positions and geodetic coordinates from an orbit state."""

    r_ecef_m = eci_to_ecef_positions(orbit.r_eci_m, orbit.t_utc)
    lat_deg, lon_deg, alt_m = ecef_to_geodetic(r_ecef_m)

    return FrameState(r_ecef_m=r_ecef_m, lat_deg=lat_deg, lon_deg=lon_deg, alt_m=alt_m)


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

    b_ned = as_float_vector_array(b_ned, "b_ned")

    _check_coordinate_inputs(b_ned, lat_deg, lon_deg)

    x_ecef, y_ecef, z_ecef = pm.enu2ecefv(
        b_ned[:, 1], b_ned[:, 0], -b_ned[:, 2], lat_deg, lon_deg, deg=True
    )

    return np.column_stack([x_ecef, y_ecef, z_ecef]).astype(np.float64)


def eci_vectors_to_ecef(vectors_eci: np.ndarray, time_utc: Any) -> np.ndarray:
    """
    Convert free vectors from ECI coordinates to ECEF coordinates via pymap3d.
    """

    vectors_eci = as_float_vector_array(vectors_eci, "vectors_eci")
    times = as_datetime_array(time_utc)

    if len(times) != len(vectors_eci):
        raise ValueError("time_utc must have the same length as vectors_eci.")

    x_ecef, y_ecef, z_ecef = pm.eci2ecef(
        vectors_eci[:, 0],
        vectors_eci[:, 1],
        vectors_eci[:, 2],
        times,
        **_pymap3d_astropy_kwargs(pm.eci2ecef),
    )

    return np.column_stack(
        [
            np.atleast_1d(np.asarray(x_ecef, dtype=np.float64)),
            np.atleast_1d(np.asarray(y_ecef, dtype=np.float64)),
            np.atleast_1d(np.asarray(z_ecef, dtype=np.float64)),
        ]
    )


def ecef_vectors_to_eci(vectors_ecef: np.ndarray, time_utc: Any) -> np.ndarray:
    """
    Convert free vectors from ECEF coordinates to ECI coordinates via pymap3d.
    """

    vectors_ecef = as_float_vector_array(vectors_ecef, "vectors_ecef")
    times = as_datetime_array(time_utc)

    if len(times) != len(vectors_ecef):
        raise ValueError("time_utc must have the same length as vectors_ecef.")

    x_eci, y_eci, z_eci = pm.ecef2eci(
        vectors_ecef[:, 0],
        vectors_ecef[:, 1],
        vectors_ecef[:, 2],
        times,
        **_pymap3d_astropy_kwargs(pm.ecef2eci),
    )

    return np.column_stack(
        [
            np.atleast_1d(np.asarray(x_eci, dtype=np.float64)),
            np.atleast_1d(np.asarray(y_eci, dtype=np.float64)),
            np.atleast_1d(np.asarray(z_eci, dtype=np.float64)),
        ]
    )


class Pymap3dFrameTransformer:
    """Adapter exposing the current pymap3d frame transformation path."""

    def compute(self, orbit: OrbitState) -> FrameState:
        """Compute ECEF and geodetic data from an orbit state."""

        return compute_frame_state(orbit)
