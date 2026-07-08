"""Shared helpers for simulation modules."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from astropy.time import Time
from astropy.utils import iers


def configure_iers_offline() -> None:
    """Keep Astropy-backed transforms deterministic without network access."""

    iers.conf.auto_download = False


def as_time_array(time_utc: Any) -> Time:
    """Convert UTC timestamps to an astropy Time array."""

    configure_iers_offline()

    if isinstance(time_utc, Time):
        if time_utc.isscalar:
            return Time([time_utc])
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


def as_datetime_array(time_utc: Any) -> np.ndarray:
    """Convert UTC timestamps to Python datetimes."""

    times = as_time_array(time_utc)
    return np.atleast_1d(times.to_datetime())


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a scalar-first quaternion."""

    norm = np.linalg.norm(quaternion)

    if norm <= 0.0:
        raise ValueError("Quaternion must have non-zero norm.")

    return quaternion / norm
