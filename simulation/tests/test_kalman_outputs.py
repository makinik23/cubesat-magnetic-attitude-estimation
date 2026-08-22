"""Tests for Kalman estimate table columns and plots."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.time import Time

from simulation.config import create_default_magnetometer_config
from simulation.io import build_results_dataframe
from simulation.pipeline import (
    AEKF_OUTPUT_DIR,
    KALMAN_CSV_FILENAME,
    build_attitude_aekf_dataframe,
    cleanup_legacy_kalman_outputs,
    save_kalman_results,
    save_plot_outputs,
)
from simulation.types import (
    AttitudeConfig,
    AttitudeState,
    FrameState,
    KalmanFilterEstimate,
    MagneticFieldState,
    OrbitState,
    SimulationResult,
)
from simulation.visualization import (
    plot_kalman_angular_velocity,
    plot_kalman_innovation_consistency,
    plot_kalman_magnetometer_bias,
    plot_kalman_state_covariance,
    plot_kalman_state_error,
    plot_kalman_state_quaternion,
)


def _simulation_result_with_kalman_estimate() -> SimulationResult:
    sample_count = 2
    times_s = np.array([0.0, 1.0], dtype=np.float64)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (sample_count, 1))
    omega_body_radps = np.tile(np.array([0.01, -0.02, 0.03], dtype=np.float64), (sample_count, 1))
    mag_bias_t = np.tile(np.array([0.1e-6, -0.2e-6, 0.3e-6], dtype=np.float64), (sample_count, 1))
    kalman_state = np.column_stack((quaternions, omega_body_radps, mag_bias_t))

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
            rotation_eci_from_lvlh=np.tile(np.eye(3, dtype=np.float64), (sample_count, 1, 1)),
        ),
        magnetic_field=MagneticFieldState(
            b_ned_nt=np.ones((sample_count, 3), dtype=np.float64),
            b_ecef_t=np.ones((sample_count, 3), dtype=np.float64) * 1e-6,
            b_eci_t=np.ones((sample_count, 3), dtype=np.float64) * 2e-6,
        ),
        attitude=AttitudeState(
            q_eci_from_body=quaternions,
            omega_body_radps=omega_body_radps,
            rotation_eci_from_body=np.tile(np.eye(3, dtype=np.float64), (sample_count, 1, 1)),
            euler_zyx_rad=np.zeros((sample_count, 3), dtype=np.float64),
            rt_r_minus_i=np.zeros((sample_count, 3, 3), dtype=np.float64),
            det_rotation=np.ones(sample_count, dtype=np.float64),
        ),
        b_body_t=np.ones((sample_count, 3), dtype=np.float64) * 4e-6,
        b_magnetometer_t=np.ones((sample_count, 3), dtype=np.float64) * 5e-6,
        kalman_estimate=KalmanFilterEstimate(
            t_s=times_s,
            state=kalman_state,
            covariance=np.tile(np.eye(10, dtype=np.float64) * 1e-4, (sample_count, 1, 1)),
            innovation=np.zeros((sample_count, 3), dtype=np.float64),
            innovation_covariance=np.tile(np.eye(3, dtype=np.float64), (sample_count, 1, 1)),
        ),
    )


class KalmanOutputTests(unittest.TestCase):
    """Check exported Kalman estimate data and plot files."""

    def test_default_magnetometers_are_axis_parallel_but_not_collinear(self) -> None:
        config = create_default_magnetometer_config()
        sensor_axes = config.sensor_axes_from_body
        positions_body_m = config.positions_body_m

        np.testing.assert_allclose(sensor_axes, np.eye(3, dtype=np.float64), atol=1e-12)
        self.assertEqual(positions_body_m.shape, (3, 3))
        self.assertTrue(np.isclose(positions_body_m[0, 2], 0.0))
        self.assertTrue(np.isclose(positions_body_m[1, 0], 0.0))
        self.assertTrue(np.isclose(positions_body_m[2, 1], 0.0))
        self.assertGreater(abs(float(positions_body_m[0, 1])), 0.0)
        self.assertGreater(abs(float(positions_body_m[1, 2])), 0.0)
        self.assertGreater(abs(float(positions_body_m[2, 0])), 0.0)

    def test_results_dataframe_includes_kalman_state_columns(self) -> None:
        df = build_results_dataframe(_simulation_result_with_kalman_estimate())

        np.testing.assert_allclose(df["q_kalman_w"], np.ones(2))
        np.testing.assert_allclose(df["q_kalman_norm"], np.ones(2))
        np.testing.assert_allclose(df["q_kalman_error_angle_deg"], np.zeros(2))
        np.testing.assert_allclose(df["sigma_kalman_w"], np.ones(2) * 1e-2)
        np.testing.assert_allclose(df["innovation_kalman_norm_T"], np.zeros(2))
        np.testing.assert_allclose(df["nis_kalman"], np.zeros(2))
        np.testing.assert_allclose(df["omega_kalman_x_radps"], np.ones(2) * 0.01)
        np.testing.assert_allclose(df["mag_bias_kalman_z_T"], np.ones(2) * 0.3e-6)
        np.testing.assert_allclose(df["mag_bias_kalman_z_uT"], np.ones(2) * 0.3)

    def test_kalman_plotters_save_png_files(self) -> None:
        df = build_results_dataframe(_simulation_result_with_kalman_estimate())

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            plot_kalman_state_quaternion(df, output_dir)
            plot_kalman_state_error(df, output_dir)
            plot_kalman_state_covariance(df, output_dir)
            plot_kalman_angular_velocity(df, output_dir)
            plot_kalman_magnetometer_bias(df, output_dir)
            plot_kalman_innovation_consistency(df, output_dir)

            self.assertTrue((output_dir / "kalman_state_quaternion.png").is_file())
            self.assertTrue((output_dir / "kalman_state_error.png").is_file())
            self.assertTrue((output_dir / "kalman_state_covariance.png").is_file())
            self.assertTrue((output_dir / "kalman_angular_velocity.png").is_file())
            self.assertTrue((output_dir / "kalman_magnetometer_bias.png").is_file())
            self.assertTrue((output_dir / "kalman_innovation_consistency.png").is_file())

    def test_save_plot_outputs_routes_kalman_plots_to_aekf_subdirectory(self) -> None:
        df = build_results_dataframe(_simulation_result_with_kalman_estimate())

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            paths = save_plot_outputs(df, output_dir)
            kalman_path = output_dir / AEKF_OUTPUT_DIR / "kalman_state_quaternion.png"

            self.assertTrue((output_dir / "position_eci.png").is_file())
            self.assertTrue(kalman_path.is_file())
            self.assertFalse((output_dir / "kalman_state_quaternion.png").exists())
            self.assertIn(kalman_path, paths)

    def test_save_kalman_results_uses_filter_specific_csv_name(self) -> None:
        df = build_results_dataframe(_simulation_result_with_kalman_estimate())

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / AEKF_OUTPUT_DIR
            csv_path = save_kalman_results(df, output_dir)

            self.assertEqual(csv_path, output_dir / KALMAN_CSV_FILENAME)
            self.assertTrue(csv_path.is_file())

    def test_cleanup_legacy_kalman_outputs_removes_root_kalman_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            old_quaternion_plot = output_dir / "kalman_state_quaternion.png"
            old_gyro_bias_plot = output_dir / "kalman_gyro_bias.png"
            regular_plot = output_dir / "position_eci.png"

            old_quaternion_plot.touch()
            old_gyro_bias_plot.touch()
            regular_plot.touch()

            removed_paths = cleanup_legacy_kalman_outputs(output_dir)

            self.assertCountEqual(removed_paths, [old_quaternion_plot, old_gyro_bias_plot])
            self.assertFalse(old_quaternion_plot.exists())
            self.assertFalse(old_gyro_bias_plot.exists())
            self.assertTrue(regular_plot.exists())

    def test_attitude_aekf_dataframe_uses_quaternion_only_estimate(self) -> None:
        result = _simulation_result_with_kalman_estimate()
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

        df = build_attitude_aekf_dataframe(result, attitude_config)

        self.assertTrue(np.isfinite(df["q_kalman_w"]).all())
        self.assertTrue(df["omega_kalman_x_radps"].isna().all())
        self.assertTrue(df["mag_bias_kalman_x_T"].isna().all())


if __name__ == "__main__":
    unittest.main()
