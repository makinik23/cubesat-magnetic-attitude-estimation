"""Rigid-body attitude propagation."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from simulation.attitude.rotations import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_zyx_euler,
)
from simulation.helpers import normalize_quaternion
from simulation.types import ArrayFloat64, AttitudeConfig, AttitudeState


def rigid_body_derivative(
    quaternion_eci_from_body: ArrayFloat64,
    omega_body_radps: ArrayFloat64,
    inertia_kg_m2: ArrayFloat64,
    torque_body_nm: ArrayFloat64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute quaternion and angular-velocity derivatives.

    This solves the classical Euler rigid-body equation:
    I * omega_dot + omega x (I * omega) = torque.
    """

    omega_quaternion = np.array([0.0, *omega_body_radps], dtype=np.float64)
    quaternion_dot = 0.5 * quaternion_multiply(quaternion_eci_from_body, omega_quaternion)

    angular_momentum_body = inertia_kg_m2 @ omega_body_radps
    omega_dot_body = np.linalg.solve(
        inertia_kg_m2, torque_body_nm - np.cross(omega_body_radps, angular_momentum_body)
    )

    return quaternion_dot, omega_dot_body


def attitude_state_derivative(
    _time_s: float, state: ArrayFloat64, config: AttitudeConfig
) -> ArrayFloat64:
    """Compute the solve_ivp state derivative for attitude propagation."""

    quaternion = normalize_quaternion(state[:4])
    omega = state[4:]
    quaternion_dot, omega_dot = rigid_body_derivative(
        quaternion, omega, config.inertia_kg_m2, config.torque_body_nm
    )

    return np.concatenate((quaternion_dot, omega_dot))


def propagate_attitude(times_s: ArrayFloat64, config: AttitudeConfig) -> AttitudeState:
    """
    Propagate torque-free rigid-body attitude over the simulation time grid.
    """

    initial_state = np.concatenate(
        (config.initial_quaternion_eci_from_body, config.initial_omega_body_radps)
    )

    if len(times_s) == 1:
        states = initial_state.reshape(1, -1)
    else:
        solution = solve_ivp(
            fun=lambda time_s, state: attitude_state_derivative(time_s, state, config),
            t_span=(float(times_s[0]), float(times_s[-1])),
            y0=initial_state,
            t_eval=times_s,
            method=config.integration_method,
            rtol=config.rtol,
            atol=config.atol,
        )

        if not solution.success:
            raise RuntimeError(f"Attitude integration failed: {solution.message}")

        states = solution.y.T

    quaternions = states[:, :4]
    quaternion_norms = np.linalg.norm(quaternions, axis=1)

    if np.any(quaternion_norms <= 0.0):
        raise RuntimeError("Attitude integration produced a zero-norm quaternion.")

    quaternions = quaternions / quaternion_norms[:, np.newaxis]
    omegas = states[:, 4:]

    rotation_matrices = np.asarray(
        [quaternion_to_rotation_matrix(quaternion) for quaternion in quaternions], dtype=np.float64
    )
    euler_zyx_rad = np.asarray(
        [rotation_matrix_to_zyx_euler(rotation_matrix) for rotation_matrix in rotation_matrices],
        dtype=np.float64,
    )
    rt_r_minus_i = np.einsum("nji,njk->nik", rotation_matrices, rotation_matrices) - np.eye(3)
    det_rotation = np.linalg.det(rotation_matrices)

    return AttitudeState(
        q_eci_from_body=quaternions,
        omega_body_radps=omegas,
        rotation_eci_from_body=rotation_matrices,
        euler_zyx_rad=euler_zyx_rad,
        rt_r_minus_i=rt_r_minus_i,
        det_rotation=det_rotation,
    )


class SolveIvpAttitudePropagator:
    """Adapter exposing the current solve_ivp attitude propagator."""

    def propagate(self, times_s: ArrayFloat64, config: AttitudeConfig) -> AttitudeState:
        """Propagate attitude over a time grid."""

        return propagate_attitude(times_s, config)
