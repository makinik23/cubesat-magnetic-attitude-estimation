"""Tests for the full attitude, rate and magnetometer-bias AEKF."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.estimation import AEKF, AEKFConfig
from simulation.helpers import normalize_quaternion
from simulation.types import KalmanFilterInput


class FullAEKFTests(unittest.TestCase):
    """Check the 10-state AEKF model conventions."""

    def assert_jacobian_matches_finite_difference(
        self,
        actual: np.ndarray,
        function,
        point: np.ndarray,
        output_size: int,
        step: float,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        expected = np.empty((output_size, len(point)), dtype=np.float64)

        for column in range(len(point)):
            perturbation = np.zeros(len(point), dtype=np.float64)
            perturbation[column] = step
            upper = function(point + perturbation)
            lower = function(point - perturbation)
            expected[:, column] = (upper - lower) / (2.0 * step)

        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

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

    def test_measurement_includes_sensor_mounting_and_magnetometer_bias(self) -> None:
        sensor_axes_from_body = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        aekf = AEKF(AEKFConfig(sensor_axes_from_body=sensor_axes_from_body))
        state = np.zeros(10, dtype=np.float64)
        state[0] = 1.0
        state[4:7] = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        state[7:10] = np.array([0.1e-6, 0.2e-6, -0.3e-6], dtype=np.float64)

        measurement = aekf.predict_measurement_from_state(
            state, np.array([2e-5, -1e-5, 4e-5], dtype=np.float64)
        )

        np.testing.assert_allclose(measurement, np.array([10.1e-6, 20.2e-6, 39.7e-6]))

    def test_rejects_invalid_sensor_axes(self) -> None:
        with self.assertRaises(ValueError):
            AEKF(AEKFConfig(sensor_axes_from_body=np.diag([1.0, 1.0, 0.5])))

    def test_accepts_legacy_initial_magnetometer_bias_name(self) -> None:
        aekf = AEKF(
            AEKFConfig(initial_magnetometer_bias_body_t=np.array([0.1e-6, -0.2e-6, 0.3e-6]))
        )

        np.testing.assert_allclose(aekf.initial_state[7:10], np.array([0.1e-6, -0.2e-6, 0.3e-6]))

    def test_rejects_ambiguous_initial_magnetometer_bias_names(self) -> None:
        with self.assertRaises(ValueError):
            AEKF(
                AEKFConfig(
                    initial_magnetometer_bias_sensor_t=np.zeros(3, dtype=np.float64),
                    initial_magnetometer_bias_body_t=np.zeros(3, dtype=np.float64),
                )
            )

    def test_predict_projects_covariance_after_quaternion_normalization(self) -> None:
        process_noise = np.eye(10, dtype=np.float64) * 1e-8
        aekf = AEKF(AEKFConfig(process_noise=process_noise))
        state = np.zeros(10, dtype=np.float64)
        state[0] = 1.0
        covariance = np.eye(10, dtype=np.float64) * 1e-6

        predicted_state, predicted_covariance, _transition = aekf.predict(state, covariance, 0.0)
        quaternion = predicted_state[:4]
        radial_variance = float(quaternion @ predicted_covariance[:4, :4] @ quaternion)

        self.assertAlmostEqual(radial_variance, 0.0, places=18)

    def test_measurement_jacobian_matches_finite_difference(self) -> None:
        sensor_axes_from_body = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        aekf = AEKF(AEKFConfig(sensor_axes_from_body=sensor_axes_from_body))
        state = np.array(
            [0.9, 0.2, -0.1, 0.35, 0.003, -0.004, 0.002, 0.1e-6, -0.2e-6, 0.05e-6], dtype=np.float64
        )
        state[:4] = normalize_quaternion(state[:4])
        reference_vector_eci_t = np.array([2e-5, -1e-5, 4e-5], dtype=np.float64)

        jacobian = aekf.measurement_jacobian(state, reference_vector_eci_t)

        self.assert_jacobian_matches_finite_difference(
            jacobian,
            lambda perturbed_state: aekf.predict_measurement_from_state(
                perturbed_state, reference_vector_eci_t
            ),
            state,
            3,
            1e-8,
            atol=1e-11,
            rtol=1e-6,
        )

    def test_state_transition_jacobian_matches_finite_difference(self) -> None:
        aekf = AEKF(
            AEKFConfig(
                inertia_kg_m2=np.diag([0.0409, 0.0403, 0.0073]),
                torque_body_nm=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )
        )
        state = np.array(
            [0.9, 0.2, -0.1, 0.35, 0.003, -0.004, 0.002, 0.1e-6, -0.2e-6, 0.05e-6], dtype=np.float64
        )
        state[:4] = normalize_quaternion(state[:4])

        jacobian = aekf.state_transition_jacobian(state, 10.0)

        self.assert_jacobian_matches_finite_difference(
            jacobian,
            lambda perturbed_state: aekf.predict_state(perturbed_state, 10.0),
            state,
            10,
            1e-7,
            atol=1e-8,
            rtol=1e-6,
        )

    def test_estimate_returns_full_state_trajectory(self) -> None:
        aekf = AEKF(
            AEKFConfig(
                initial_quaternion_eci_from_body=np.array([1.0, 0.0, 0.0, 0.0]),
                initial_omega_body_radps=np.zeros(3, dtype=np.float64),
                initial_magnetometer_bias_sensor_t=np.zeros(3, dtype=np.float64),
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
