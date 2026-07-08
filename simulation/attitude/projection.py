"""Body-frame vector projection helpers."""

from __future__ import annotations

import numpy as np

from simulation.types import ArrayFloat64, AttitudeState


def project_eci_vectors_to_body(vectors_eci: ArrayFloat64, attitude: AttitudeState) -> ArrayFloat64:
    """Project ECI-frame vectors into the spacecraft body frame."""

    rotation_body_from_eci = np.transpose(attitude.rotation_eci_from_body, axes=(0, 2, 1))

    return np.einsum("nij,nj->ni", rotation_body_from_eci, vectors_eci)


class RotationBodyFieldProjector:
    """Adapter exposing the current ECI-to-body vector projection."""

    def project(self, vectors_eci: ArrayFloat64, attitude: AttitudeState) -> ArrayFloat64:
        """Project inertial vectors into body coordinates."""

        return project_eci_vectors_to_body(vectors_eci, attitude)
