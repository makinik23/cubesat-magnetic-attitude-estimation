"""Tests for simulation orchestration with injectable strategies."""

from __future__ import annotations

import unittest

import astropy.units as u
import numpy as np
from astropy.time import Time

from simulation.pipeline import SimulationRunner
from simulation.types import (
    AttitudeConfig,
    AttitudeState,
    ClassicalOrbitalElements,
    FrameState,
    KalmanFilterEstimate,
    KalmanFilterInput,
    MagneticFieldState,
    OrbitState,
    SimulationConfig,
)


class SimulationRunnerTests(unittest.TestCase):
    """Check that the runner wires strategies without hard-coding implementations."""

    def test_runner_accepts_replaceable_strategies(self) -> None:
        test_case = self
        calls: list[str] = []

        orbit_state = OrbitState(
            t_s=np.array([0.0, 1.0], dtype=np.float64),
            t_utc=Time(["2026-01-01T00:00:00.000", "2026-01-01T00:00:01.000"], scale="utc"),
            r_eci_m=np.ones((2, 3), dtype=np.float64),
            v_eci_mps=np.ones((2, 3), dtype=np.float64) * 2.0,
        )
        frame_state = FrameState(
            r_ecef_m=np.ones((2, 3), dtype=np.float64) * 3.0,
            lat_deg=np.zeros(2, dtype=np.float64),
            lon_deg=np.zeros(2, dtype=np.float64),
            alt_m=np.zeros(2, dtype=np.float64),
        )
        magnetic_state = MagneticFieldState(
            b_ned_nt=np.ones((2, 3), dtype=np.float64),
            b_ecef_t=np.ones((2, 3), dtype=np.float64) * 1e-6,
            b_eci_t=np.ones((2, 3), dtype=np.float64) * 2e-6,
        )
        attitude_state = AttitudeState(
            q_eci_from_body=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (2, 1)),
            omega_body_radps=np.zeros((2, 3), dtype=np.float64),
            rotation_eci_from_body=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
            euler_zyx_rad=np.zeros((2, 3), dtype=np.float64),
            rt_r_minus_i=np.zeros((2, 3, 3), dtype=np.float64),
            det_rotation=np.ones(2, dtype=np.float64),
        )
        b_body_t = np.ones((2, 3), dtype=np.float64) * 4e-6
        b_magnetometer_t = np.ones((2, 3), dtype=np.float64) * 5e-6
        kalman_estimate = KalmanFilterEstimate(
            t_s=orbit_state.t_s,
            state=np.zeros((2, 4), dtype=np.float64),
            covariance=np.tile(np.eye(4, dtype=np.float64), (2, 1, 1)),
            innovation=np.zeros((2, 3), dtype=np.float64),
            innovation_covariance=np.tile(np.eye(3, dtype=np.float64), (2, 1, 1)),
        )

        class OrbitStrategy:
            def propagate(
                self, elements: ClassicalOrbitalElements, config: SimulationConfig
            ) -> OrbitState:
                calls.append("orbit")
                return orbit_state

        class FrameStrategy:
            def compute(self, orbit: OrbitState) -> FrameState:
                test_case.assertIs(orbit, orbit_state)
                calls.append("frame")
                return frame_state

        class MagneticStrategy:
            def compute(self, orbit: OrbitState, frame: FrameState) -> MagneticFieldState:
                test_case.assertIs(orbit, orbit_state)
                test_case.assertIs(frame, frame_state)
                calls.append("magnetic")
                return magnetic_state

        class AttitudeStrategy:
            def propagate(self, times_s: np.ndarray, config: AttitudeConfig) -> AttitudeState:
                np.testing.assert_allclose(times_s, orbit_state.t_s)
                calls.append("attitude")
                return attitude_state

        class BodyProjector:
            def project(self, vectors_eci: np.ndarray, attitude: AttitudeState) -> np.ndarray:
                np.testing.assert_allclose(vectors_eci, magnetic_state.b_eci_t)
                test_case.assertIs(attitude, attitude_state)
                calls.append("body")
                return b_body_t

        class MagnetometerStrategy:
            def measure(self, vectors_body: np.ndarray) -> np.ndarray:
                np.testing.assert_allclose(vectors_body, b_body_t)
                calls.append("magnetometer")
                return b_magnetometer_t

        class KalmanStrategy:
            def estimate(self, inputs: KalmanFilterInput) -> KalmanFilterEstimate:
                np.testing.assert_allclose(inputs.t_s, orbit_state.t_s)
                np.testing.assert_allclose(inputs.measurements_body_t, b_magnetometer_t)
                np.testing.assert_allclose(inputs.reference_vectors_eci_t, magnetic_state.b_eci_t)
                angular_rate_body_radps = inputs.angular_rate_body_radps
                test_case.assertIsNotNone(angular_rate_body_radps)
                assert angular_rate_body_radps is not None
                np.testing.assert_allclose(angular_rate_body_radps, attitude_state.omega_body_radps)
                calls.append("kalman")
                return kalman_estimate

        elements = ClassicalOrbitalElements(
            semi_major_axis=7_000_000.0 * u.m,
            eccentricity=0.0 * u.one,
            inclination=0.0 * u.rad,
            raan=0.0 * u.rad,
            argument_of_perigee=0.0 * u.rad,
            true_anomaly=0.0 * u.rad,
            epoch=Time("2026-01-01T00:00:00.000", scale="utc"),
        )
        simulation_config = SimulationConfig(duration_s=1.0, time_step_s=1.0)
        attitude_config = AttitudeConfig(
            mass_kg=1.0,
            inertia_kg_m2=np.eye(3, dtype=np.float64),
            initial_quaternion_eci_from_body=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            initial_omega_body_radps=np.zeros(3, dtype=np.float64),
            torque_body_nm=np.zeros(3, dtype=np.float64),
            integration_method="DOP853",
            rtol=1e-9,
            atol=1e-12,
        )
        runner = SimulationRunner(
            orbit_propagator=OrbitStrategy(),
            frame_transformer=FrameStrategy(),
            magnetic_field_model=MagneticStrategy(),
            attitude_propagator=AttitudeStrategy(),
            body_field_projector=BodyProjector(),
            magnetometer_model=MagnetometerStrategy(),
            kalman_filter=KalmanStrategy(),
        )

        result = runner.run(elements, simulation_config, attitude_config)

        self.assertEqual(
            calls, ["orbit", "frame", "magnetic", "attitude", "body", "magnetometer", "kalman"]
        )
        self.assertIs(result.orbit, orbit_state)
        self.assertIs(result.frame, frame_state)
        self.assertIs(result.magnetic_field, magnetic_state)
        self.assertIs(result.attitude, attitude_state)
        self.assertIs(result.kalman_estimate, kalman_estimate)
        np.testing.assert_allclose(result.b_body_t, b_body_t)
        np.testing.assert_allclose(result.b_magnetometer_t, b_magnetometer_t)


if __name__ == "__main__":
    unittest.main()
