"""Sensor models used to corrupt ideal simulation signals."""

from __future__ import annotations

import numpy as np

from simulation.interfaces import Magnetometer
from simulation.types import ArrayFloat64


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

        self.bias_body_t = bias_body_t
        self.noise_std_t = noise_std_t
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    def measure(self, b_body_t: ArrayFloat64) -> ArrayFloat64:
        """Return magnetometer measurements for ideal body-frame field vectors."""

        noise_t = self._rng.normal(loc=0.0, scale=self.noise_std_t, size=b_body_t.shape)

        return b_body_t + self.bias_body_t + noise_t
