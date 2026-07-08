"""Attitude propagation, rotations and vector projection."""

from simulation.attitude.dynamics import (
    SolveIvpAttitudePropagator,
    attitude_state_derivative,
    propagate_attitude,
    rigid_body_derivative,
)
from simulation.attitude.projection import RotationBodyFieldProjector, project_eci_vectors_to_body
from simulation.attitude.rotations import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_zyx_euler,
)

__all__ = [
    "RotationBodyFieldProjector",
    "SolveIvpAttitudePropagator",
    "attitude_state_derivative",
    "project_eci_vectors_to_body",
    "propagate_attitude",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rigid_body_derivative",
    "rotation_matrix_to_quaternion",
    "rotation_matrix_to_zyx_euler",
]
