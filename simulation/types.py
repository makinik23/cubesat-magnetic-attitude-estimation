"""Shared typed containers for simulation core data."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.time import Time

ArrayFloat64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ClassicalOrbitalElements:
    """
    Classical Keplerian orbital elements.

    All angular quantities are stored as astropy quantities.
    """

    semi_major_axis: u.Quantity
    eccentricity: u.Quantity
    inclination: u.Quantity
    raan: u.Quantity
    argument_of_perigee: u.Quantity
    true_anomaly: u.Quantity
    epoch: Time


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Time configuration of the simulation."""

    duration_s: float
    time_step_s: float


@dataclass(frozen=True, slots=True)
class AttitudeConfig:
    """
    Rigid-body attitude propagation settings.

    The initial quaternion is scalar-first and maps body coordinates into ECI.
    """

    mass_kg: float
    inertia_kg_m2: ArrayFloat64
    initial_quaternion_eci_from_body: ArrayFloat64
    initial_omega_body_radps: ArrayFloat64
    torque_body_nm: ArrayFloat64
    integration_method: str
    rtol: float
    atol: float


@dataclass(frozen=True, slots=True)
class OrbitState:
    """Orbit propagation results in ECI coordinates."""

    t_s: ArrayFloat64
    t_utc: Time
    r_eci_m: ArrayFloat64
    v_eci_mps: ArrayFloat64


@dataclass(frozen=True, slots=True)
class FrameState:
    """Position and geodetic data derived from the orbit state."""

    r_ecef_m: ArrayFloat64
    lat_deg: ArrayFloat64
    lon_deg: ArrayFloat64
    alt_m: ArrayFloat64


@dataclass(frozen=True, slots=True)
class MagneticFieldState:
    """Geomagnetic field vectors in the frames used by the simulation."""

    b_ned_nt: ArrayFloat64
    b_ecef_t: ArrayFloat64
    b_eci_t: ArrayFloat64


@dataclass(frozen=True, slots=True)
class AttitudeState:
    """Rigid-body attitude propagation results."""

    q_eci_from_body: ArrayFloat64
    omega_body_radps: ArrayFloat64
    rotation_eci_from_body: ArrayFloat64
    euler_zyx_rad: ArrayFloat64
    rt_r_minus_i: ArrayFloat64
    det_rotation: ArrayFloat64


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Complete simulation result before tabular export or plotting."""

    orbit: OrbitState
    frame: FrameState
    magnetic_field: MagneticFieldState
    attitude: AttitudeState
    b_body_t: ArrayFloat64
    b_magnetometer_t: ArrayFloat64
