"""Geomagnetic-field models."""

from simulation.magnetic.geomagnetic import (
    IgrfMagneticFieldModel,
    compute_igrf_ned_nt,
    compute_magnetic_field_state,
)

__all__ = ["IgrfMagneticFieldModel", "compute_igrf_ned_nt", "compute_magnetic_field_state"]
