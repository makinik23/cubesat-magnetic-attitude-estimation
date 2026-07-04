"""Tests for frame-conversion conventions."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.frames import ecef_vectors_to_eci, eci_vectors_to_ecef, ned_to_ecef_vectors


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
        vector_ecef = np.array([[10e-6, -20e-6, 30e-6]], dtype=np.float64)
        time_utc = np.array(["2026-01-01T12:00:00.000"], dtype=str)

        vector_eci = ecef_vectors_to_eci(vector_ecef, time_utc)

        np.testing.assert_allclose(
            np.linalg.norm(vector_eci, axis=1), np.linalg.norm(vector_ecef, axis=1), rtol=1e-6
        )

    def test_eci_ecef_vector_round_trip(self) -> None:
        vectors_eci = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64)
        time_utc = np.array(["2026-01-01T12:00:00.000", "2026-01-01T12:10:00.000"], dtype=str)

        vectors_ecef = eci_vectors_to_ecef(vectors_eci, time_utc)
        round_trip = ecef_vectors_to_eci(vectors_ecef, time_utc)

        np.testing.assert_allclose(round_trip, vectors_eci, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
