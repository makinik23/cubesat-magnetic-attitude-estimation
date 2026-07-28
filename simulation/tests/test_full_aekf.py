"""Tests for the full attitude, rate and magnetometer-bias AEKF."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.estimation import AEKF, AEKFConfig
from simulation.types import KalmanFilterInput


class FullAEKFTests(unittest.TestCase):
    """Check the 10-state AEKF model conventions."""

    def test_predict_state_rotates_quaternion_and_preserves_random_walk_states(self) -> None:
        aekf = AEKF()
        state = np.zeros(10, dtype=np.float64)
        state[0] = 1.0
        state[6] = np.pi / 2.0
        state[7:] = np.array([0.01, -0.02, 0.03], dtype=np.float64)

        predicted = aekf.predict_state(state, 1.0)

        expected_quaternion = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        np.testing.assert_allclose(predicted[:4], expected_quaternion, atol=1e-3)
        np.testing.assert_allclose(predicted[4:7], state[4:7])
        np.testing.assert_allclose(predicted[7:10], state[7:10])

    def test_measurement_includes_magnetometer_bias(self) -> None:
        aekf = AEKF()
        state = np.zeros(10, dtype=np.float64)
        state[0] = 1.0
        state[4:7] = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        state[7:10] = np.array([0.1e-6, 0.2e-6, -0.3e-6], dtype=np.float64)

        measurement = aekf.predict_measurement_from_state(
            state, np.array([2e-5, -1e-5, 4e-5], dtype=np.float64)
        )

        np.testing.assert_allclose(measurement, np.array([20.1e-6, -9.8e-6, 39.7e-6]))

    def test_estimate_returns_full_state_trajectory(self) -> None:
        aekf = AEKF(
            AEKFConfig(
                initial_quaternion_eci_from_body=np.array([1.0, 0.0, 0.0, 0.0]),
                initial_omega_body_radps=np.zeros(3, dtype=np.float64),
                initial_magnetometer_bias_body_t=np.zeros(3, dtype=np.float64),
                initial_covariance=np.eye(10, dtype=np.float64) * 1e-3,
                process_noise=np.eye(10, dtype=np.float64) * 1e-12,
                measurement_noise=np.eye(3, dtype=np.float64) * 1e-12,
            )
        )
        times_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        reference_vectors_eci_t = np.tile(
            np.array([2e-5, -1e-5, 4e-5], dtype=np.float64), (len(times_s), 1)
        )

        estimate = aekf.estimate(
            KalmanFilterInput(
                t_s=times_s,
                measurements_body_t=reference_vectors_eci_t.copy(),
                reference_vectors_eci_t=reference_vectors_eci_t,
            )
        )

        self.assertEqual(estimate.state.shape, (len(times_s), 10))
        self.assertEqual(estimate.covariance.shape, (len(times_s), 10, 10))
        np.testing.assert_allclose(np.linalg.norm(estimate.state[:, :4], axis=1), 1.0)
        assert estimate.innovation is not None
        self.assertEqual(estimate.innovation.shape, (len(times_s), 3))


if __name__ == "__main__":
    unittest.main()
