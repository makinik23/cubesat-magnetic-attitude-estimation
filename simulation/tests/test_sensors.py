"""Tests for simple sensor models."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.sensors import MagnetometerModel


class MagnetometerModelTests(unittest.TestCase):
    """Check bias and Gaussian noise behavior."""

    def test_measure_adds_bias_without_noise(self) -> None:
        b_body_t = np.array([[20e-6, -5e-6, 30e-6], [21e-6, -4e-6, 29e-6]], dtype=np.float64)
        bias_body_t = np.array([100e-9, -50e-9, 25e-9], dtype=np.float64)

        model = MagnetometerModel(bias_body_t=bias_body_t, noise_std_t=0.0)

        np.testing.assert_allclose(model.measure(b_body_t), b_body_t + bias_body_t)

    def test_seed_makes_noise_reproducible(self) -> None:
        b_body_t = np.ones((4, 3), dtype=np.float64) * 20e-6

        first_model = MagnetometerModel(noise_std_t=10e-9, seed=123)
        second_model = MagnetometerModel(noise_std_t=10e-9, seed=123)

        np.testing.assert_allclose(first_model.measure(b_body_t), second_model.measure(b_body_t))

    def test_supports_per_axis_noise(self) -> None:
        b_body_t = np.zeros((3, 3), dtype=np.float64)
        model = MagnetometerModel(
            noise_std_t=np.array([1e-9, 2e-9, 3e-9], dtype=np.float64), seed=1
        )

        measurement = model.measure(b_body_t)

        self.assertEqual(measurement.shape, b_body_t.shape)


if __name__ == "__main__":
    unittest.main()
