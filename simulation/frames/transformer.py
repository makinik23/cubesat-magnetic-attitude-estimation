"""
Reference-frame and geodetic conversion utilities.

The current pipeline delegates ECI/ECEF transforms to pymap3d. With Astropy
available, pymap3d uses the GCRS <-> ITRS transform path instead of the lower
accuracy Vallado/GMST-only fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymap3d as pm

from simulation.helpers import as_datetime_array
from simulation.types import FrameState, OrbitState


def _stack_components(x: Any, y: Any, z: Any) -> np.ndarray:
    """Stack x, y and z components returned by pymap3d."""

    return np.column_stack((np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)))


def _normalize_rows(vectors: np.ndarray, name: str) -> np.ndarray:
    """Normalize row vectors and reject degenerate directions."""

    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms <= 0.0):
        raise ValueError(f"{name} contains a zero-length vector.")

    return vectors / norms[:, np.newaxis]


def compute_rotation_eci_from_lvlh(r_eci_m: np.ndarray, v_eci_mps: np.ndarray) -> np.ndarray:
    """
    Compute LVLH-to-ECI rotation matrices from inertial position and velocity.

    The LVLH convention is +X along-track, +Z nadir and +Y completing a
    right-handed frame. Returned matrices map LVLH coordinates into ECI
    coordinates, so each matrix column is one LVLH axis expressed in ECI.
    """

    r_eci = np.asarray(r_eci_m, dtype=np.float64)
    v_eci = np.asarray(v_eci_mps, dtype=np.float64)

    if r_eci.shape != v_eci.shape or r_eci.ndim != 2 or r_eci.shape[1] != 3:
        raise ValueError("r_eci_m and v_eci_mps must both have shape (N, 3).")

    radius_direction_eci = _normalize_rows(r_eci, "r_eci_m")
    angular_momentum_direction_eci = _normalize_rows(np.cross(r_eci, v_eci), "r_eci_m x v_eci_mps")

    z_lvlh_eci = -radius_direction_eci
    y_lvlh_eci = -angular_momentum_direction_eci
    x_lvlh_eci = _normalize_rows(np.cross(y_lvlh_eci, z_lvlh_eci), "x LVLH axis")

    return np.stack((x_lvlh_eci, y_lvlh_eci, z_lvlh_eci), axis=2)


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

    times = as_datetime_array(time_utc)

    x_ecef, y_ecef, z_ecef = pm.eci2ecef(
        r_eci_m[:, 0], r_eci_m[:, 1], r_eci_m[:, 2], times, force_non_astropy=False
    )

    return _stack_components(x_ecef, y_ecef, z_ecef)


def ecef_to_geodetic(r_ecef_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ECEF positions to geodetic latitude, longitude and altitude.

    Returns latitude and longitude in degrees, altitude in meters.
    """

    lat_deg, lon_deg, alt_m = pm.ecef2geodetic(
        r_ecef_m[:, 0], r_ecef_m[:, 1], r_ecef_m[:, 2], deg=True
    )

    return lat_deg, lon_deg, alt_m


def compute_frame_state(orbit: OrbitState) -> FrameState:
    """Compute ECEF positions and geodetic coordinates from an orbit state."""

    r_ecef_m = eci_to_ecef_positions(orbit.r_eci_m, orbit.t_utc)
    lat_deg, lon_deg, alt_m = ecef_to_geodetic(r_ecef_m)
    rotation_eci_from_lvlh = compute_rotation_eci_from_lvlh(orbit.r_eci_m, orbit.v_eci_mps)

    return FrameState(
        r_ecef_m=r_ecef_m,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        rotation_eci_from_lvlh=rotation_eci_from_lvlh,
    )


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

    x_ecef, y_ecef, z_ecef = pm.enu2ecefv(
        b_ned[:, 1], b_ned[:, 0], -b_ned[:, 2], lat_deg, lon_deg, deg=True
    )

    return _stack_components(x_ecef, y_ecef, z_ecef)


def eci_vectors_to_ecef(vectors_eci: np.ndarray, time_utc: Any) -> np.ndarray:
    """
    Convert free vectors from ECI coordinates to ECEF coordinates via pymap3d.
    """

    times = as_datetime_array(time_utc)

    x_ecef, y_ecef, z_ecef = pm.eci2ecef(
        vectors_eci[:, 0], vectors_eci[:, 1], vectors_eci[:, 2], times, force_non_astropy=False
    )

    return _stack_components(x_ecef, y_ecef, z_ecef)


def ecef_vectors_to_eci(vectors_ecef: np.ndarray, time_utc: Any) -> np.ndarray:
    """
    Convert free vectors from ECEF coordinates to ECI coordinates via pymap3d.
    """

    times = as_datetime_array(time_utc)

    x_eci, y_eci, z_eci = pm.ecef2eci(
        vectors_ecef[:, 0], vectors_ecef[:, 1], vectors_ecef[:, 2], times, force_non_astropy=False
    )

    return _stack_components(x_eci, y_eci, z_eci)


class Pymap3dFrameTransformer:
    """Adapter exposing the current pymap3d frame transformation path."""

    def compute(self, orbit: OrbitState) -> FrameState:
        """Compute ECEF and geodetic data from an orbit state."""

        return compute_frame_state(orbit)
