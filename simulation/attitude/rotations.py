"""Quaternion and rotation-matrix utilities."""

from __future__ import annotations

import numpy as np

from simulation.helpers import normalize_quaternion
from simulation.types import ArrayFloat64


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

    qw, qx, qy, qz = normalize_quaternion(quaternion)

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

    matrix = rotation_matrix
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

    quaternion = normalize_quaternion(quaternion)

    if quaternion[0] < 0.0:
        quaternion = -quaternion

    return quaternion


def rotation_matrix_to_zyx_euler(rotation_matrix: ArrayFloat64) -> ArrayFloat64:
    """
    Convert an ECI-from-body rotation matrix to ZYX yaw, pitch and roll angles.
    """

    matrix = rotation_matrix
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-12:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])

    return np.array([yaw, pitch, roll], dtype=np.float64)
