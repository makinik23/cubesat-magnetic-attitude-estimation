"""Tests for simple sensor models."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.sensors import MagnetometerModel


class MagnetometerModelTests(unittest.TestCase):
    """Check bias and Gaussian noise behavior."""

    def test_measure_adds_bias_without_noise(self) -> None:
        b_body_t = np.array([[20e-6, -5e-6, 30e-6], [21e-6, -4e-6, 29e-6]], dtype=np.float64)
        bias_sensor_t = np.array([100e-9, -50e-9, 25e-9], dtype=np.float64)

        model = MagnetometerModel(bias_sensor_t=bias_sensor_t, noise_std_t=0.0)

        np.testing.assert_allclose(model.measure(b_body_t), b_body_t + bias_sensor_t)

    def test_accepts_legacy_body_bias_name_for_body_aligned_sensor(self) -> None:
        b_body_t = np.array([[20e-6, -5e-6, 30e-6]], dtype=np.float64)
        bias_t = np.array([100e-9, -50e-9, 25e-9], dtype=np.float64)

        model = MagnetometerModel(bias_body_t=bias_t, noise_std_t=0.0)

        np.testing.assert_allclose(model.measure(b_body_t), b_body_t + bias_t)

    def test_measure_projects_body_field_to_sensor_axes(self) -> None:
        b_body_t = np.array([[1.0e-6, 2.0e-6, 3.0e-6]], dtype=np.float64)
        sensor_axes_from_body = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
        model = MagnetometerModel(noise_std_t=0.0, sensor_axes_from_body=sensor_axes_from_body)

        measurement = model.measure(b_body_t)

        np.testing.assert_allclose(measurement, np.array([[2.0e-6, 3.0e-6, 1.0e-6]]))

    def test_stores_sensor_positions_in_body_frame(self) -> None:
        positions_body_m = np.array(
            [[0.02, 0.03, 0.0], [0.0, -0.02, 0.03], [-0.03, 0.0, -0.02]], dtype=np.float64
        )

        model = MagnetometerModel(positions_body_m=positions_body_m)

        np.testing.assert_allclose(model.positions_body_m, positions_body_m)

    def test_accepts_legacy_sensor_rotation_name(self) -> None:
        b_body_t = np.array([[1.0e-6, 2.0e-6, 3.0e-6]], dtype=np.float64)
        rotation_sensor_from_body = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
        model = MagnetometerModel(
            noise_std_t=0.0, rotation_sensor_from_body=rotation_sensor_from_body
        )

        measurement = model.measure(b_body_t)

        np.testing.assert_allclose(measurement, np.array([[2.0e-6, 3.0e-6, 1.0e-6]]))

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

    def test_rejects_invalid_sensor_axes(self) -> None:
        invalid_rotation = np.diag([1.0, 1.0, 2.0])

        with self.assertRaises(ValueError):
            MagnetometerModel(sensor_axes_from_body=invalid_rotation)

    def test_rejects_collinear_sensor_axes(self) -> None:
        collinear_axes = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )

        with self.assertRaises(ValueError):
            MagnetometerModel(sensor_axes_from_body=collinear_axes)


if __name__ == "__main__":
    unittest.main()
