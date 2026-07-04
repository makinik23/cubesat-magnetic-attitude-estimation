"""Sensor models used to corrupt ideal simulation signals."""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation.helpers import as_float_vector, as_float_vector_array
from simulation.interfaces import Magnetometer
from simulation.types import ArrayFloat64


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


class MagnetometerModel(Magnetometer):
    """Simple body-frame magnetometer with constant bias and Gaussian noise."""

    def __init__(
        self,
        bias_body_t: ArrayFloat64 | None = None,
        noise_std_t: float | ArrayFloat64 = 0.0,
        seed: int | None = None,
    ) -> None:
        """Initialize magnetometer parameters and noise generator."""

        if bias_body_t is None:
            bias_body_t = np.zeros(3, dtype=np.float64)

        self.bias_body_t = as_float_vector(bias_body_t, "bias_body_t", finite=True)
        self.noise_std_t = _as_noise_std(noise_std_t)
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def measure(self, b_body_t: ArrayFloat64) -> ArrayFloat64:
        """Return magnetometer measurements for ideal body-frame field vectors."""

        b_body_t = as_float_vector_array(b_body_t, "b_body_t", finite=True)
        noise_t = self._rng.normal(loc=0.0, scale=self.noise_std_t, size=b_body_t.shape)

        return np.asarray(b_body_t + self.bias_body_t + noise_t, dtype=np.float64)
