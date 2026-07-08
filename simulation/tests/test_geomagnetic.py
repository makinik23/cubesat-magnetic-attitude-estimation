"""Tests for geomagnetic-field computation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import unittest

import numpy as np
from astropy.time import Time

from simulation.frames import Pymap3dFrameTransformer
from simulation.geomagnetic import IgrfMagneticFieldModel, compute_igrf_ned_nt
from simulation.types import OrbitState

REFERENCE_DATA_PATH = Path(__file__).parent / "data" / "magnetic_injections.json"


@dataclass(frozen=True, slots=True)
class MagneticReferenceStats:
    """Error statistics for B_eci reference-data comparison."""

    sample_count: int
    mean_error_t: float
    median_error_t: float
    p95_error_t: float
    max_error_t: float
    rmse_error_t: float
    mean_relative_error: float
    max_relative_error: float
    component_mean_error_t: np.ndarray
    component_rmse_error_t: np.ndarray
    component_max_abs_error_t: np.ndarray


def _load_reference_injections() -> list[dict[str, object]]:
    data = json.loads(REFERENCE_DATA_PATH.read_text(encoding="utf-8"))
    return data["test_scenario_0"]["injections"]


def _reference_orbit_state(injections: list[dict[str, object]]) -> OrbitState:
    r_eci_m = np.array([injection["r_eci_given"] for injection in injections], dtype=float) * 1000.0
    times_utc = Time(
        [
            datetime.strptime(str(injection["epoch"]), "%d-%b-%Y %H:%M:%S")
            for injection in injections
        ],
        scale="utc",
    )

    return OrbitState(
        t_s=np.arange(len(injections), dtype=float),
        t_utc=times_utc,
        r_eci_m=r_eci_m,
        v_eci_mps=np.zeros_like(r_eci_m),
    )


def _expected_b_eci_t(injections: list[dict[str, object]]) -> np.ndarray:
    return np.array([injection["expected_beci"] for injection in injections], dtype=float)


def _compute_reference_stats() -> MagneticReferenceStats:
    injections = _load_reference_injections()
    orbit = _reference_orbit_state(injections)
    frame = Pymap3dFrameTransformer().compute(orbit)
    magnetic_field = IgrfMagneticFieldModel().compute(orbit, frame)

    expected_b_eci_t = _expected_b_eci_t(injections)
    error_t = magnetic_field.b_eci_t - expected_b_eci_t
    error_norm_t = np.linalg.norm(error_t, axis=1)
    reference_norm_t = np.linalg.norm(expected_b_eci_t, axis=1)
    relative_error = error_norm_t / reference_norm_t

    return MagneticReferenceStats(
        sample_count=len(injections),
        mean_error_t=float(np.mean(error_norm_t)),
        median_error_t=float(np.median(error_norm_t)),
        p95_error_t=float(np.percentile(error_norm_t, 95.0)),
        max_error_t=float(np.max(error_norm_t)),
        rmse_error_t=float(np.sqrt(np.mean(error_norm_t**2))),
        mean_relative_error=float(np.mean(relative_error)),
        max_relative_error=float(np.max(relative_error)),
        component_mean_error_t=np.mean(error_t, axis=0),
        component_rmse_error_t=np.sqrt(np.mean(error_t**2, axis=0)),
        component_max_abs_error_t=np.max(np.abs(error_t), axis=0),
    )


def _format_reference_stats(stats: MagneticReferenceStats) -> str:
    return (
        f"samples={stats.sample_count}, "
        f"mean={stats.mean_error_t * 1e9:.3f} nT, "
        f"median={stats.median_error_t * 1e9:.3f} nT, "
        f"p95={stats.p95_error_t * 1e9:.3f} nT, "
        f"max={stats.max_error_t * 1e9:.3f} nT, "
        f"rmse={stats.rmse_error_t * 1e9:.3f} nT, "
        f"mean_rel={stats.mean_relative_error * 100.0:.4f}%, "
        f"max_rel={stats.max_relative_error * 100.0:.4f}%, "
        f"component_mean={stats.component_mean_error_t * 1e9} nT, "
        f"component_rmse={stats.component_rmse_error_t * 1e9} nT, "
        f"component_max_abs={stats.component_max_abs_error_t * 1e9} nT"
    )


class GeomagneticFieldTests(unittest.TestCase):
    """Check basic IGRF output behavior."""

    def test_igrf_returns_ned_vector_for_valid_point(self) -> None:
        lat_deg = np.array([0.0], dtype=np.float64)
        lon_deg = np.array([0.0], dtype=np.float64)
        alt_m = np.array([400_000.0], dtype=np.float64)
        time_utc = Time(["2026-01-01T00:00:00.000"], scale="utc")

        b_ned_nt = compute_igrf_ned_nt(lat_deg, lon_deg, alt_m, time_utc)

        self.assertEqual(b_ned_nt.shape, (1, 3))
        self.assertTrue(np.all(np.isfinite(b_ned_nt)))

    def test_magnetic_reference_file_is_valid(self) -> None:
        injections = _load_reference_injections()

        self.assertEqual(len(injections), 1696)

    def test_pipeline_b_eci_matches_reference_injections(self) -> None:
        stats = _compute_reference_stats()
        stats_message = _format_reference_stats(stats)

        self.assertEqual(stats.sample_count, 1696)
        self.assertLess(stats.mean_error_t, 2.5e-7, stats_message)
        self.assertLess(stats.p95_error_t, 2.5e-7, stats_message)
        self.assertLess(stats.max_error_t, 3.0e-7, stats_message)
        self.assertLess(stats.max_relative_error, 0.006, stats_message)


if __name__ == "__main__":
    unittest.main()
