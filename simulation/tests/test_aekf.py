"""Tests for the quaternion-only additive EKF base class."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.estimation import AEKF, AEKFConfig
from simulation.types import KalmanFilterInput


class AEKFTests(unittest.TestCase):
    """Check the AEKF process and measurement conventions."""

    def test_predict_quaternion_uses_body_rate(self) -> None:
        aekf = AEKF()

        quaternion = aekf.predict_quaternion(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64),
            1.0,
        )

        expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float64)
        np.testing.assert_allclose(quaternion, expected, atol=1e-12)

    def test_predict_measurement_projects_eci_field_to_body(self) -> None:
        aekf = AEKF()
        angle_rad = np.pi / 2.0
        quaternion = np.array(
            [np.cos(angle_rad / 2.0), 0.0, 0.0, np.sin(angle_rad / 2.0)], dtype=np.float64
        )

        measurement = aekf.predict_measurement(
            quaternion, np.array([0.0, 1.0, 0.0], dtype=np.float64)
        )

        np.testing.assert_allclose(measurement, np.array([1.0, 0.0, 0.0]), atol=1e-12)

    def test_estimate_returns_normalized_quaternion_trajectory(self) -> None:
        aekf = AEKF(
            AEKFConfig(
                initial_quaternion_eci_from_body=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                initial_covariance=np.eye(4, dtype=np.float64) * 1e-3,
                process_noise=np.eye(4, dtype=np.float64) * 1e-12,
                measurement_noise=np.eye(3, dtype=np.float64) * 1e-12,
            )
        )
        times_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        reference_vectors_eci_t = np.tile(
            np.array([2e-5, -1e-5, 4e-5], dtype=np.float64), (len(times_s), 1)
        )
        measurements_body_t = reference_vectors_eci_t.copy()

        estimate = aekf.estimate(
            KalmanFilterInput(
                t_s=times_s,
                measurements_body_t=measurements_body_t,
                reference_vectors_eci_t=reference_vectors_eci_t,
                angular_rate_body_radps=np.zeros((len(times_s), 3), dtype=np.float64),
            )
        )

        np.testing.assert_allclose(estimate.state[:, 0], np.ones(len(times_s)), atol=1e-12)
        np.testing.assert_allclose(estimate.state[:, 1:], np.zeros((len(times_s), 3)), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(estimate.state, axis=1), 1.0, atol=1e-12)
        assert estimate.innovation is not None
        np.testing.assert_allclose(estimate.innovation, np.zeros((len(times_s), 3)), atol=1e-12)
        self.assertEqual(estimate.covariance.shape, (len(times_s), 4, 4))


if __name__ == "__main__":
    unittest.main()
