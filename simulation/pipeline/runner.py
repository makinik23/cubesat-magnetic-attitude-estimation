"""Simulation orchestration built from replaceable strategy components."""

from __future__ import annotations

from dataclasses import dataclass, field

from simulation.attitude import RotationBodyFieldProjector, SolveIvpAttitudePropagator
from simulation.frames import Pymap3dFrameTransformer
from simulation.magnetic import IgrfMagneticFieldModel
from simulation.interfaces import (
    AttitudePropagator,
    BodyFieldProjector,
    FrameTransformer,
    KalmanFilter,
    Magnetometer,
    MagneticFieldModel,
    OrbitPropagator,
)
from simulation.orbit import PoliastroKeplerPropagator
from simulation.sensors import MagnetometerModel
from simulation.types import (
    AttitudeConfig,
    ClassicalOrbitalElements,
    KalmanFilterInput,
    SimulationConfig,
    SimulationResult,
)


@dataclass(frozen=True, slots=True)
class SimulationRunner:
    """Run the full simulation with injectable numerical strategies."""

    orbit_propagator: OrbitPropagator = field(default_factory=PoliastroKeplerPropagator)
    frame_transformer: FrameTransformer = field(default_factory=Pymap3dFrameTransformer)
    magnetic_field_model: MagneticFieldModel = field(default_factory=IgrfMagneticFieldModel)
    attitude_propagator: AttitudePropagator = field(default_factory=SolveIvpAttitudePropagator)
    body_field_projector: BodyFieldProjector = field(default_factory=RotationBodyFieldProjector)
    magnetometer_model: Magnetometer = field(default_factory=MagnetometerModel)
    kalman_filter: KalmanFilter | None = None

    def run(
        self,
        elements: ClassicalOrbitalElements,
        simulation_config: SimulationConfig,
        attitude_config: AttitudeConfig,
    ) -> SimulationResult:
        """Run orbit, frame, magnetic-field and attitude computations."""
        orbit = self.orbit_propagator.propagate(elements, simulation_config)
        frame = self.frame_transformer.compute(orbit)
        magnetic_field = self.magnetic_field_model.compute(orbit, frame)
        attitude = self.attitude_propagator.propagate(orbit.t_s, attitude_config)
        b_body_t = self.body_field_projector.project(magnetic_field.b_eci_t, attitude)
        b_magnetometer_t = self.magnetometer_model.measure(b_body_t)
        kalman_estimate = None

        if self.kalman_filter is not None:
            kalman_estimate = self.kalman_filter.estimate(
                KalmanFilterInput(
                    t_s=orbit.t_s,
                    measurements_body_t=b_magnetometer_t,
                    reference_vectors_eci_t=magnetic_field.b_eci_t,
                    angular_rate_body_radps=attitude.omega_body_radps,
                )
            )

        return SimulationResult(
            orbit=orbit,
            frame=frame,
            magnetic_field=magnetic_field,
            attitude=attitude,
            b_body_t=b_body_t,
            b_magnetometer_t=b_magnetometer_t,
            kalman_estimate=kalman_estimate,
        )
