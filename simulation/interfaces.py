"""Strategy interfaces for replaceable simulation components."""

from __future__ import annotations

from typing import Protocol

from simulation.types import (
    ArrayFloat64,
    AttitudeConfig,
    AttitudeState,
    ClassicalOrbitalElements,
    FrameState,
    MagneticFieldState,
    OrbitState,
    SimulationConfig,
)


class OrbitPropagator(Protocol):
    """Propagates orbital elements into an inertial orbit state."""

    def propagate(self, elements: ClassicalOrbitalElements, config: SimulationConfig) -> OrbitState:
        """Return the orbit state for a simulation configuration."""
        raise NotImplementedError


class FrameTransformer(Protocol):
    """Computes derived frame quantities from an orbit state."""

    def compute(self, orbit: OrbitState) -> FrameState:
        """Return ECEF and geodetic data for an orbit state."""
        raise NotImplementedError


class MagneticFieldModel(Protocol):
    """Computes magnetic-field vectors along an orbit."""

    def compute(self, orbit: OrbitState, frame: FrameState) -> MagneticFieldState:
        """Return magnetic-field vectors in supported frames."""
        raise NotImplementedError


class AttitudePropagator(Protocol):
    """Propagates the spacecraft attitude state."""

    def propagate(self, times_s: ArrayFloat64, config: AttitudeConfig) -> AttitudeState:
        """Return the attitude state over the requested time grid."""
        raise NotImplementedError


class BodyFieldProjector(Protocol):
    """Projects inertial vectors into the spacecraft body frame."""

    def project(self, vectors_eci: ArrayFloat64, attitude: AttitudeState) -> ArrayFloat64:
        """Return vectors expressed in body-frame coordinates."""
        raise NotImplementedError


class Magnetometer(Protocol):
    """Measures body-frame magnetic-field vectors."""

    def measure(self, b_body_t: ArrayFloat64) -> ArrayFloat64:
        """Return magnetic-field measurements in body-frame coordinates."""
        raise NotImplementedError
