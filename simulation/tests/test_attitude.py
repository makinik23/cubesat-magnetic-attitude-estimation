"""Tests for attitude kinematics and projection conventions."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.attitude import (
    project_eci_vectors_to_body,
    propagate_attitude,
    quaternion_to_rotation_matrix,
)
from simulation.types import AttitudeConfig


def _attitude_config(initial_omega_body_radps: np.ndarray) -> AttitudeConfig:
    return AttitudeConfig(
        mass_kg=4.0,
        inertia_kg_m2=np.diag([0.02, 0.018, 0.015]).astype(np.float64),
        initial_quaternion_eci_from_body=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        initial_omega_body_radps=np.asarray(initial_omega_body_radps, dtype=np.float64),
        torque_body_nm=np.zeros(3, dtype=np.float64),
        integration_method="DOP853",
        rtol=1e-11,
        atol=1e-12,
    )


class AttitudeConventionTests(unittest.TestCase):
    """Check quaternion orientation and rigid-body propagation conventions."""

    def test_identity_quaternion_returns_identity_rotation(self) -> None:
        rotation = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    def test_positive_z_quaternion_rotates_body_x_to_eci_y(self) -> None:
        angle_rad = np.pi / 2.0
        quaternion = np.array([np.cos(angle_rad / 2.0), 0.0, 0.0, np.sin(angle_rad / 2.0)])
        rotation = quaternion_to_rotation_matrix(quaternion)

        rotated = rotation @ np.array([1.0, 0.0, 0.0])

        np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)

    def test_project_eci_vectors_to_body_uses_inverse_rotation(self) -> None:
        attitude = propagate_attitude(np.array([0.0]), _attitude_config(np.zeros(3)))
        vector_eci = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

        vector_body = project_eci_vectors_to_body(vector_eci, attitude)

        np.testing.assert_allclose(vector_body, vector_eci, atol=1e-12)

    def test_principal_axis_torque_free_rotation_matches_closed_form(self) -> None:
        omega_x = 0.1
        times_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)

        attitude = propagate_attitude(times_s, _attitude_config(np.array([omega_x, 0.0, 0.0])))

        expected = np.column_stack(
            (
                np.cos(0.5 * omega_x * times_s),
                np.sin(0.5 * omega_x * times_s),
                np.zeros_like(times_s),
                np.zeros_like(times_s),
            )
        )
        np.testing.assert_allclose(attitude.q_eci_from_body, expected, atol=1e-9)
        np.testing.assert_allclose(
            attitude.omega_body_radps,
            np.tile(np.array([omega_x, 0.0, 0.0], dtype=np.float64), (len(times_s), 1)),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
