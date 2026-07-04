"""Tests for simple sensor models."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.sensors import MagnetometerModel


class MagnetometerModelTests(unittest.TestCase):
    """Check bias, Gaussian noise and input validation."""

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

    def test_rejects_invalid_bias_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "bias_body_t must have shape"):
            MagnetometerModel(bias_body_t=np.zeros((1, 3), dtype=np.float64))

    def test_rejects_invalid_noise_std(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise_std_t must be non-negative"):
            MagnetometerModel(noise_std_t=-1.0)

        with self.assertRaisesRegex(ValueError, "noise_std_t must be a scalar"):
            MagnetometerModel(noise_std_t=np.ones((1, 3), dtype=np.float64))

    def test_rejects_invalid_measurement_input_shape(self) -> None:
        model = MagnetometerModel()

        with self.assertRaisesRegex(ValueError, "b_body_t must have shape"):
            model.measure(np.zeros(3, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
