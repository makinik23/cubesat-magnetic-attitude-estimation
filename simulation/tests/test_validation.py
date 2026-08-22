"""Tests for AEKF validation metrics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from simulation.config import create_default_orbit
from simulation.validation.aekf import (
    orbital_period_s,
    summarize_aekf_dataframe,
    summarize_monte_carlo,
)


class AEKFValidationTests(unittest.TestCase):
    """Check validation metric helpers without running full simulations."""

    def test_orbital_period_matches_default_leo(self) -> None:
        period_s = orbital_period_s(create_default_orbit())

        self.assertGreater(period_s, 5600.0)
        self.assertLess(period_s, 5800.0)

    def test_summarize_aekf_dataframe_computes_settling_and_nis_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "t_s": np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
                "q_kalman_error_angle_deg": np.array([2.5, 1.0, 0.5, 0.4], dtype=np.float64),
                "innovation_kalman_norm_T": np.array(
                    [2.0e-6, 1.0e-6, 0.5e-6, 0.25e-6], dtype=np.float64
                ),
                "mag_bias_kalman_x_uT": np.array([0.0, 0.1, 0.2, 0.3]),
                "mag_bias_kalman_y_uT": np.array([0.0, -0.1, -0.2, -0.2]),
                "mag_bias_kalman_z_uT": np.array([0.0, 0.0, 0.05, 0.1]),
                "nis_kalman": np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float64),
                "omega_kalman_error_norm_degps": np.array([0.2, 0.1, 0.05, 0.01], dtype=np.float64),
            }
        )

        summary = summarize_aekf_dataframe(
            df, np.array([0.3e-6, -0.2e-6, 0.1e-6], dtype=np.float64), 100.0
        )

        self.assertEqual(summary["attitude_settled_below_1_deg_s"], 10.0)
        self.assertAlmostEqual(summary["bias_final_error_norm_uT"], 0.0)
        self.assertEqual(summary["nis_mean"], 3.0)
        self.assertEqual(summary["nis_central_95_fraction"], 1.0)

    def test_summarize_monte_carlo_aggregates_per_run_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "attitude_final_deg": [0.5, 1.0],
                "attitude_rms_deg": [0.7, 1.1],
                "bias_final_error_norm_uT": [0.1, 0.2],
                "innovation_mean_norm_uT": [1.0, 2.0],
                "nis_mean": [2.5, 3.5],
                "nis_central_95_fraction": [0.9, 1.0],
                "attitude_settled_below_2_deg_s": [5.0, 15.0],
                "attitude_settled_below_1p5_deg_s": [8.0, 18.0],
                "attitude_settled_below_1_deg_s": [10.0, np.nan],
                "bias_settled_below_0p15_uT_s": [20.0, np.nan],
            }
        )

        summary = summarize_monte_carlo(df, period_s=100.0)

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["attitude_final_deg_mean"], 0.75)
        self.assertEqual(summary["nis_mean_mean"], 3.0)
        self.assertEqual(summary["attitude_settled_below_1_deg_fraction"], 0.5)
        self.assertEqual(summary["attitude_settled_below_1_deg_count"], 1)
        self.assertEqual(summary["attitude_settled_below_1_deg_mean_s"], 10.0)
        self.assertEqual(summary["attitude_settled_below_2_deg_mean_s"], 10.0)
        self.assertEqual(summary["attitude_settled_below_2_deg_mean_orbits"], 0.1)


if __name__ == "__main__":
    unittest.main()
