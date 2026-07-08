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

from simulation.frames import ecef_vectors_to_eci, ned_to_ecef_vectors
from simulation.helpers import as_time_array
from simulation.types import FrameState, MagneticFieldState, OrbitState


def _first_value(value: Any) -> float:
    """Return the scalar value from a ppigrf component."""

    return float(np.ravel(value)[0])


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

    times = as_time_array(time_utc)

    b_ned_nt = np.empty((len(lat_deg), 3), dtype=float)

    for idx, (lat, lon, alt, time) in enumerate(zip(lat_deg, lon_deg, alt_m, times)):
        date = time.to_datetime()
        alt_km = alt / 1000.0

        # ppigrf.igrf returns east, north and up components in nT.
        b_east_nt, b_north_nt, b_up_nt = ppigrf.igrf(lon, lat, alt_km, date)

        b_ned_nt[idx] = [_first_value(b_north_nt), _first_value(b_east_nt), -_first_value(b_up_nt)]

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
