"""Tests for result table assembly."""

from __future__ import annotations

import unittest

import numpy as np
from astropy.time import Time

from simulation.io import build_results_dataframe
from simulation.types import (
    AttitudeState,
    FrameState,
    MagneticFieldState,
    OrbitState,
    SimulationResult,
)


class ResultTableTests(unittest.TestCase):
    """Check derived result columns."""

    def test_body_aligned_with_lvlh_has_zero_lvlh_euler_angles(self) -> None:
        sample_count = 1
        rotation_eci_from_lvlh = np.array(
            [[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]], dtype=np.float64
        )

        result = SimulationResult(
            orbit=OrbitState(
                t_s=np.array([0.0], dtype=np.float64),
                t_utc=Time(["2026-01-01T00:00:00.000"], scale="utc"),
                r_eci_m=np.array([[7000e3, 0.0, 0.0]], dtype=np.float64),
                v_eci_mps=np.array([[0.0, 7500.0, 0.0]], dtype=np.float64),
            ),
            frame=FrameState(
                r_ecef_m=np.array([[7000e3, 0.0, 0.0]], dtype=np.float64),
                lat_deg=np.zeros(sample_count, dtype=np.float64),
                lon_deg=np.zeros(sample_count, dtype=np.float64),
                alt_m=np.zeros(sample_count, dtype=np.float64),
                rotation_eci_from_lvlh=rotation_eci_from_lvlh,
            ),
            magnetic_field=MagneticFieldState(
                b_ned_nt=np.zeros((sample_count, 3), dtype=np.float64),
                b_ecef_t=np.zeros((sample_count, 3), dtype=np.float64),
                b_eci_t=np.zeros((sample_count, 3), dtype=np.float64),
            ),
            attitude=AttitudeState(
                q_eci_from_body=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
                omega_body_radps=np.zeros((sample_count, 3), dtype=np.float64),
                rotation_eci_from_body=rotation_eci_from_lvlh,
                euler_zyx_rad=np.zeros((sample_count, 3), dtype=np.float64),
                rt_r_minus_i=np.zeros((sample_count, 3, 3), dtype=np.float64),
                det_rotation=np.ones(sample_count, dtype=np.float64),
            ),
            b_body_t=np.zeros((sample_count, 3), dtype=np.float64),
            b_magnetometer_t=np.zeros((sample_count, 3), dtype=np.float64),
        )

        df = build_results_dataframe(result)

        np.testing.assert_allclose(df["yaw_lvlh_from_body_rad"], np.zeros(sample_count), atol=1e-12)
        np.testing.assert_allclose(
            df["pitch_lvlh_from_body_rad"], np.zeros(sample_count), atol=1e-12
        )
        np.testing.assert_allclose(
            df["roll_lvlh_from_body_rad"], np.zeros(sample_count), atol=1e-12
        )
        np.testing.assert_allclose(df["R_eci_from_lvlh_13"], np.array([-1.0]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
