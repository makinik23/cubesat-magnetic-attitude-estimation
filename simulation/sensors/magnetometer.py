"""Sensor models used to corrupt ideal simulation signals."""

from __future__ import annotations

from numbers import Real
from typing import cast

import numpy as np

from simulation.interfaces import Magnetometer
from simulation.types import ArrayFloat64


def _default_vector() -> ArrayFloat64:
    return np.zeros(3, dtype=np.float64)


def _default_sensor_axes() -> ArrayFloat64:
    return np.eye(3, dtype=np.float64)


def _default_positions() -> ArrayFloat64:
    return np.zeros((3, 3), dtype=np.float64)


class MagnetometerModel(Magnetometer):
    """
    Three-axis magnetometer triad with fixed mounting, bias and Gaussian noise.

    Rows of ``sensor_axes_from_body`` are the three sensor sensitive axes
    expressed in body-frame coordinates. Rows of ``positions_body_m`` are the
    corresponding sensor positions in body-frame coordinates.
    """

    def __init__(
        self,
        bias_sensor_t: ArrayFloat64 | None = None,
        noise_std_t: float | ArrayFloat64 = 0.0,
        seed: int | None = None,
        sensor_axes_from_body: ArrayFloat64 | None = None,
        positions_body_m: ArrayFloat64 | None = None,
        rotation_sensor_from_body: ArrayFloat64 | None = None,
        bias_body_t: ArrayFloat64 | None = None,
    ) -> None:
        """Initialize magnetometer parameters and noise generator."""

        if bias_sensor_t is not None and bias_body_t is not None:
            raise ValueError("Use either bias_sensor_t or bias_body_t, not both.")
        if bias_sensor_t is None and bias_body_t is not None:
            bias_sensor_t = bias_body_t
        if bias_sensor_t is None:
            bias_sensor_t = _default_vector()
        if sensor_axes_from_body is not None and rotation_sensor_from_body is not None:
            raise ValueError(
                "Use either sensor_axes_from_body or rotation_sensor_from_body, not both."
            )
        if sensor_axes_from_body is None:
            sensor_axes_from_body = (
                rotation_sensor_from_body
                if rotation_sensor_from_body is not None
                else _default_sensor_axes()
            )
        if positions_body_m is None:
            positions_body_m = _default_positions()

        self.bias_sensor_t = _as_vector(bias_sensor_t, "bias_sensor_t")
        self.noise_std_t = _as_noise_std(noise_std_t)
        self.seed = seed
        self.sensor_axes_from_body = _as_sensor_axes_matrix(
            sensor_axes_from_body, "sensor_axes_from_body"
        )
        self.rotation_sensor_from_body = self.sensor_axes_from_body
        self.positions_body_m = _as_positions_matrix(positions_body_m, "positions_body_m")
        self._rng = np.random.default_rng(self.seed)

    def measure(self, b_body_t: ArrayFloat64) -> ArrayFloat64:
        """Return sensor-frame magnetometer measurements for ideal body-frame fields."""

        body_field = np.asarray(b_body_t, dtype=np.float64)

        if body_field.shape[-1:] != (3,):
            raise ValueError("b_body_t must have trailing dimension 3.")

        sensor_field = np.einsum("ij,...j->...i", self.sensor_axes_from_body, body_field)
        noise_t = self._rng.normal(loc=0.0, scale=self.noise_std_t, size=sensor_field.shape)

        return sensor_field + self.bias_sensor_t + noise_t


def _as_vector(values: ArrayFloat64, name: str) -> ArrayFloat64:
    vector = np.asarray(values, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")

    return vector


def _as_noise_std(values: float | ArrayFloat64) -> float | ArrayFloat64:
    if isinstance(values, Real):
        noise_std = float(values)

        if noise_std < 0.0:
            raise ValueError("noise_std_t must be nonnegative.")

        return noise_std

    noise_std_vector = _as_vector(cast(ArrayFloat64, values), "noise_std_t")

    if np.any(noise_std_vector < 0.0):
        raise ValueError("noise_std_t must be nonnegative.")

    return noise_std_vector


def _as_sensor_axes_matrix(values: ArrayFloat64, name: str) -> ArrayFloat64:
    matrix = np.asarray(values, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    if not np.allclose(matrix @ matrix.T, np.eye(3, dtype=np.float64), atol=1.0e-9):
        raise ValueError(f"{name} must be orthonormal.")

    return matrix


def _as_positions_matrix(values: ArrayFloat64, name: str) -> ArrayFloat64:
    matrix = np.asarray(values, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")

    return matrix
