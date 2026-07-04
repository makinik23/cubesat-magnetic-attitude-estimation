"""Shared conversion and validation helpers for simulation modules."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from astropy.time import Time
from astropy.utils import iers

from simulation.types import ArrayFloat64


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


def as_float_vector(
    values: Any, name: str, *, length: int = 3, finite: bool = False
) -> ArrayFloat64:
    """Convert values to a float64 vector with shape ``(length,)``."""

    array = np.asarray(values, dtype=np.float64)

    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},).")

    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def as_float_matrix(
    values: Any, name: str, *, shape: tuple[int, int] = (3, 3), finite: bool = False
) -> ArrayFloat64:
    """Convert values to a float64 matrix with the requested shape."""

    array = np.asarray(values, dtype=np.float64)

    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")

    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def as_float_vector_array(values: Any, name: str, *, finite: bool = False) -> ArrayFloat64:
    """Convert values to a float64 array with shape ``(N, 3)``."""

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")

    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def normalize_quaternion(quaternion: ArrayFloat64) -> ArrayFloat64:
    """Normalize a scalar-first quaternion."""

    quaternion = as_float_vector(quaternion, "quaternion", length=4)
    norm = np.linalg.norm(quaternion)

    if norm <= 0.0:
        raise ValueError("Quaternion must have non-zero norm.")

    return quaternion / norm
