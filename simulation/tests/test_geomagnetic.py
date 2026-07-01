"""Tests for geomagnetic-field input validation."""

from __future__ import annotations

import unittest

import numpy as np
from astropy.time import Time

from simulation.geomagnetic import compute_igrf_ned_nt


class GeomagneticInputTests(unittest.TestCase):
    """Check that geomagnetic input arrays cannot be silently truncated."""

    def test_igrf_rejects_time_length_mismatch(self) -> None:
        lat_deg = np.array([0.0, 1.0], dtype=np.float64)
        lon_deg = np.array([0.0, 1.0], dtype=np.float64)
        alt_m = np.array([400_000.0, 400_000.0], dtype=np.float64)
        time_utc = Time(["2026-01-01T00:00:00.000"], scale="utc")

        with self.assertRaisesRegex(ValueError, "time_utc"):
            compute_igrf_ned_nt(lat_deg, lon_deg, alt_m, time_utc)


if __name__ == "__main__":
    unittest.main()
