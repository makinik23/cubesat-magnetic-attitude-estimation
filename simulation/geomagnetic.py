"""
IGRF magnetic-field provider.

This module uses ppigrf as a ready-made IGRF implementation. The project focus
is the estimator, so the geomagnetic model is intentionally treated as an
external reference component.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ppigrf
from astropy.time import Time

from simulation.frames import ecef_vectors_to_eci, ned_to_ecef_vectors
from simulation.types import FrameState, MagneticFieldState, OrbitState


def _as_time_array(time_utc: Any) -> Time:
    """Convert UTC timestamps to an astropy Time array."""

    if isinstance(time_utc, Time):
        if time_utc.isscalar:
            return Time([time_utc])
        return time_utc

    return Time([str(value) for value in np.atleast_1d(time_utc)], format="isot", scale="utc")


def compute_igrf_ned_nt(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray, time_utc: Any
) -> np.ndarray:
    """
    Compute IGRF magnetic field in local NED coordinates.

    Returns
    -------
    np.ndarray
        Magnetic-field vectors [nT] in NED coordinates, shape (N, 3).
    """

    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    alt_m = np.asarray(alt_m, dtype=np.float64)
    times = _as_time_array(time_utc)

    if lat_deg.ndim != 1 or lon_deg.ndim != 1 or alt_m.ndim != 1:
        raise ValueError("lat_deg, lon_deg and alt_m must be one-dimensional arrays.")

    if lat_deg.shape != lon_deg.shape or lat_deg.shape != alt_m.shape:
        raise ValueError("lat_deg, lon_deg and alt_m must have the same length.")

    if len(times) != len(lat_deg):
        raise ValueError("time_utc must have the same length as lat_deg, lon_deg and alt_m.")

    b_ned_nt = np.empty((len(lat_deg), 3), dtype=float)

    for idx, (lat, lon, alt, time) in enumerate(zip(lat_deg, lon_deg, alt_m, times)):
        date = time.to_datetime()
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
    b_eci_t = ecef_vectors_to_eci(b_ecef_t, orbit.t_utc)

    return MagneticFieldState(b_ned_nt=b_ned_nt, b_ecef_t=b_ecef_t, b_eci_t=b_eci_t)


class IgrfMagneticFieldModel:
    """Adapter exposing the current ppigrf IGRF model."""

    def compute(self, orbit: OrbitState, frame: FrameState) -> MagneticFieldState:
        """Compute magnetic-field vectors along an orbit."""

        return compute_magnetic_field_state(orbit, frame)
