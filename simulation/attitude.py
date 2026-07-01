"""
Rigid-body attitude propagation and body-frame magnetic-field projection.

The initial attitude is provided as a scalar-first quaternion. The propagated
quaternion maps body-frame vectors into ECI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from simulation.types import ArrayFloat64, AttitudeConfig, AttitudeState


def _as_vector_array(values: Any, name: str) -> ArrayFloat64:
    """Convert values to a float64 array with shape (N, 3)."""

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")

    return array


def _normalize_quaternion(quaternion: ArrayFloat64) -> ArrayFloat64:
    """Normalize a scalar-first quaternion."""

    norm = np.linalg.norm(quaternion)

    if norm <= 0.0:
        raise ValueError("Quaternion must have non-zero norm.")

    return quaternion / norm


def quaternion_multiply(left: ArrayFloat64, right: ArrayFloat64) -> ArrayFloat64:
    """Multiply scalar-first quaternions."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right

    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation_matrix(quaternion: ArrayFloat64) -> ArrayFloat64:
    """
    Convert a scalar-first quaternion to an ECI-from-body rotation matrix.
    """

    qw, qx, qy, qz = _normalize_quaternion(quaternion)

    return np.array(
        [
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation_matrix: ArrayFloat64) -> ArrayFloat64:
    """
    Convert a rotation matrix to a scalar-first quaternion.
    """

    matrix = np.asarray(rotation_matrix, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3).")

    trace = np.trace(matrix)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))

        if diagonal_index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )

    quaternion = _normalize_quaternion(quaternion)

    if quaternion[0] < 0.0:
        quaternion = -quaternion

    return quaternion


def rotation_matrix_to_zyx_euler(rotation_matrix: ArrayFloat64) -> ArrayFloat64:
    """
    Convert an ECI-from-body rotation matrix to ZYX yaw, pitch and roll angles.
    """

    matrix = np.asarray(rotation_matrix, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3).")

    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-12:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])

    return np.array([yaw, pitch, roll], dtype=np.float64)


def rigid_body_derivative(
    quaternion_eci_from_body: ArrayFloat64,
    omega_body_radps: ArrayFloat64,
    inertia_kg_m2: ArrayFloat64,
    torque_body_nm: ArrayFloat64,
) -> tuple[ArrayFloat64, ArrayFloat64]:
    """
    Compute quaternion and angular-velocity derivatives.

    This solves the classical Euler rigid-body equation:
    I * omega_dot + omega x (I * omega) = torque.
    """

    omega_quaternion = np.array([0.0, *omega_body_radps], dtype=np.float64)
    quaternion_dot = 0.5 * quaternion_multiply(quaternion_eci_from_body, omega_quaternion)

    angular_momentum_body = inertia_kg_m2 @ omega_body_radps
    omega_dot_body = np.asarray(
        np.linalg.solve(
            inertia_kg_m2, torque_body_nm - np.cross(omega_body_radps, angular_momentum_body)
        ),
        dtype=np.float64,
    )

    return quaternion_dot, omega_dot_body


def attitude_state_derivative(
    _time_s: float, state: ArrayFloat64, config: AttitudeConfig
) -> ArrayFloat64:
    """Compute the solve_ivp state derivative for attitude propagation."""

    quaternion = _normalize_quaternion(state[:4])
    omega = state[4:]
    quaternion_dot, omega_dot = rigid_body_derivative(
        quaternion, omega, config.inertia_kg_m2, config.torque_body_nm
    )

    return np.concatenate((quaternion_dot, omega_dot))


def propagate_attitude(times_s: ArrayFloat64, config: AttitudeConfig) -> AttitudeState:
    """
    Propagate torque-free rigid-body attitude over the simulation time grid.
    """

    times_s = np.asarray(times_s, dtype=np.float64)

    if times_s.ndim != 1:
        raise ValueError("times_s must be a one-dimensional array.")

    if len(times_s) == 0:
        raise ValueError("times_s must not be empty.")

    if len(times_s) > 1 and np.any(np.diff(times_s) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")

    initial_state = np.concatenate(
        (
            config.initial_quaternion_eci_from_body,
            np.asarray(config.initial_omega_body_radps, dtype=np.float64),
        )
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

        states = np.asarray(solution.y.T, dtype=np.float64)

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


def project_eci_vectors_to_body(vectors_eci: ArrayFloat64, attitude: AttitudeState) -> ArrayFloat64:
    """Project ECI-frame vectors into the spacecraft body frame."""

    vectors_eci = _as_vector_array(vectors_eci, "vectors_eci")

    if vectors_eci.shape[0] != attitude.rotation_eci_from_body.shape[0]:
        raise ValueError("vectors_eci and attitude state must have the same length.")

    rotation_body_from_eci = np.transpose(attitude.rotation_eci_from_body, axes=(0, 2, 1))

    return np.einsum("nij,nj->ni", rotation_body_from_eci, vectors_eci)


class SolveIvpAttitudePropagator:
    """Adapter exposing the current solve_ivp attitude propagator."""

    def propagate(self, times_s: ArrayFloat64, config: AttitudeConfig) -> AttitudeState:
        """Propagate attitude over a time grid."""

        return propagate_attitude(times_s, config)


class RotationBodyFieldProjector:
    """Adapter exposing the current ECI-to-body vector projection."""

    def project(self, vectors_eci: ArrayFloat64, attitude: AttitudeState) -> ArrayFloat64:
        """Project inertial vectors into body coordinates."""

        return project_eci_vectors_to_body(vectors_eci, attitude)
