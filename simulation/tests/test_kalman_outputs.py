"""Tests for Kalman estimate table columns and plots."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.time import Time

from simulation.io import build_results_dataframe
from simulation.types import (
    AttitudeState,
    FrameState,
    KalmanFilterEstimate,
    MagneticFieldState,
    OrbitState,
    SimulationResult,
)
from simulation.visualization import (
    plot_kalman_state_covariance,
    plot_kalman_state_error,
    plot_kalman_state_quaternion,
)


def _simulation_result_with_kalman_estimate() -> SimulationResult:
    sample_count = 2
    times_s = np.array([0.0, 1.0], dtype=np.float64)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (sample_count, 1))

    return SimulationResult(
        orbit=OrbitState(
            t_s=times_s,
            t_utc=Time(["2026-01-01T00:00:00.000", "2026-01-01T00:00:01.000"], scale="utc"),
            r_eci_m=np.ones((sample_count, 3), dtype=np.float64),
            v_eci_mps=np.ones((sample_count, 3), dtype=np.float64),
        ),
        frame=FrameState(
            r_ecef_m=np.ones((sample_count, 3), dtype=np.float64),
            lat_deg=np.zeros(sample_count, dtype=np.float64),
            lon_deg=np.zeros(sample_count, dtype=np.float64),
            alt_m=np.zeros(sample_count, dtype=np.float64),
        ),
        magnetic_field=MagneticFieldState(
            b_ned_nt=np.ones((sample_count, 3), dtype=np.float64),
            b_ecef_t=np.ones((sample_count, 3), dtype=np.float64) * 1e-6,
            b_eci_t=np.ones((sample_count, 3), dtype=np.float64) * 2e-6,
        ),
        attitude=AttitudeState(
            q_eci_from_body=quaternions,
            omega_body_radps=np.zeros((sample_count, 3), dtype=np.float64),
            rotation_eci_from_body=np.tile(np.eye(3, dtype=np.float64), (sample_count, 1, 1)),
            euler_zyx_rad=np.zeros((sample_count, 3), dtype=np.float64),
            rt_r_minus_i=np.zeros((sample_count, 3, 3), dtype=np.float64),
            det_rotation=np.ones(sample_count, dtype=np.float64),
        ),
        b_body_t=np.ones((sample_count, 3), dtype=np.float64) * 4e-6,
        b_magnetometer_t=np.ones((sample_count, 3), dtype=np.float64) * 5e-6,
        kalman_estimate=KalmanFilterEstimate(
            t_s=times_s,
            state=quaternions.copy(),
            covariance=np.tile(np.eye(4, dtype=np.float64) * 1e-4, (sample_count, 1, 1)),
            innovation=np.zeros((sample_count, 3), dtype=np.float64),
            innovation_covariance=np.tile(np.eye(3, dtype=np.float64), (sample_count, 1, 1)),
        ),
    )


class KalmanOutputTests(unittest.TestCase):
    """Check exported Kalman estimate data and plot files."""

    def test_results_dataframe_includes_kalman_state_columns(self) -> None:
        df = build_results_dataframe(_simulation_result_with_kalman_estimate())

        np.testing.assert_allclose(df["q_kalman_w"], np.ones(2))
        np.testing.assert_allclose(df["q_kalman_norm"], np.ones(2))
        np.testing.assert_allclose(df["q_kalman_error_angle_deg"], np.zeros(2))
        np.testing.assert_allclose(df["sigma_kalman_w"], np.ones(2) * 1e-2)
        np.testing.assert_allclose(df["innovation_kalman_norm_T"], np.zeros(2))

    def test_kalman_plotters_save_png_files(self) -> None:
        df = build_results_dataframe(_simulation_result_with_kalman_estimate())

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            plot_kalman_state_quaternion(df, output_dir)
            plot_kalman_state_error(df, output_dir)
            plot_kalman_state_covariance(df, output_dir)

            self.assertTrue((output_dir / "kalman_state_quaternion.png").is_file())
            self.assertTrue((output_dir / "kalman_state_error.png").is_file())
            self.assertTrue((output_dir / "kalman_state_covariance.png").is_file())


if __name__ == "__main__":
    unittest.main()
