"""Tests for shared simulation helpers."""

from __future__ import annotations

import unittest

import numpy as np
from astropy.time import Time

from simulation.helpers import as_time_array, normalize_quaternion


class HelperTests(unittest.TestCase):
    """Check shared helper contracts used across simulation modules."""

    def test_as_time_array_wraps_scalar_time(self) -> None:
        times = as_time_array(Time("2026-01-01T00:00:00.000", scale="utc"))

        self.assertEqual(len(times), 1)
        self.assertEqual(times[0].isot, "2026-01-01T00:00:00.000")

    def test_normalize_quaternion_returns_unit_quaternion(self) -> None:
        quaternion = normalize_quaternion(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64))

        np.testing.assert_allclose(quaternion, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_normalize_quaternion_rejects_zero_norm(self) -> None:
        with self.assertRaisesRegex(ValueError, "Quaternion must have non-zero norm"):
            normalize_quaternion(np.zeros(4, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
