"""Sensor models used to corrupt ideal simulation signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulation.types import ArrayFloat64


def _as_body_vector(values: Any, name: str) -> ArrayFloat64:
    """Convert values to a finite float64 vector with shape (3,)."""

    array = np.asarray(values, dtype=np.float64)

    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _as_noise_std(values: Any) -> ArrayFloat64:
    """Convert a scalar or per-axis standard deviation to float64."""

    array = np.asarray(values, dtype=np.float64)

    if array.shape not in ((), (3,)):
        raise ValueError("noise_std_t must be a scalar or have shape (3,).")

    if np.any(array < 0.0):
        raise ValueError("noise_std_t must be non-negative.")

    if not np.all(np.isfinite(array)):
        raise ValueError("noise_std_t must contain only finite values.")

    return array


def _as_vector_array(values: Any, name: str) -> ArrayFloat64:
    """Convert values to a finite float64 array with shape (N, 3)."""

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


@dataclass(slots=True)
class MagnetometerModel:
    """Simple body-frame magnetometer with constant bias and Gaussian noise."""

    bias_body_t: ArrayFloat64 = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    noise_std_t: float | ArrayFloat64 = 0.0
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters and initialize the noise generator."""

        self.bias_body_t = _as_body_vector(self.bias_body_t, "bias_body_t")
        self.noise_std_t = _as_noise_std(self.noise_std_t)
        self._rng = np.random.default_rng(self.seed)

    def measure(self, b_body_t: ArrayFloat64) -> ArrayFloat64:
        """Return magnetometer measurements for ideal body-frame field vectors."""

        b_body_t = _as_vector_array(b_body_t, "b_body_t")
        noise_t = self._rng.normal(loc=0.0, scale=self.noise_std_t, size=b_body_t.shape)

        return np.asarray(b_body_t + self.bias_body_t + noise_t, dtype=np.float64)
