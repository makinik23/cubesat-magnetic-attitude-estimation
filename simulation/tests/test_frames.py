"""Tests for frame-conversion conventions."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.frames import ecef_vectors_to_eci, ned_to_ecef_vectors


class FrameConversionTests(unittest.TestCase):
    """Check axis signs and rotation-like vector transforms."""

    def test_ned_to_ecef_axes_at_equator_greenwich(self) -> None:
        lat_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        lon_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ned_basis = np.eye(3, dtype=np.float64)

        ecef_basis = ned_to_ecef_vectors(ned_basis, lat_deg, lon_deg)

        expected = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
        np.testing.assert_allclose(ecef_basis, expected, atol=1e-12)

    def test_ecef_to_eci_vector_transform_preserves_norm(self) -> None:
        r_ecef_m = np.array([[6_878_137.0, 0.0, 0.0]], dtype=np.float64)
        vector_ecef = np.array([[10e-6, -20e-6, 30e-6]], dtype=np.float64)
        time_utc = np.array(["2026-01-01T12:00:00.000"], dtype=str)

        vector_eci = ecef_vectors_to_eci(vector_ecef, r_ecef_m, time_utc)

        np.testing.assert_allclose(
            np.linalg.norm(vector_eci, axis=1), np.linalg.norm(vector_ecef, axis=1), rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
