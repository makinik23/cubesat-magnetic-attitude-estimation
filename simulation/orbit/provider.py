"""
Orbit propagation provider based on poliastro.

This module converts classical orbital elements into ECI position and velocity
time series. The generated ECI-like frame follows the inertial frame convention
used internally by poliastro for the two-body problem.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from simulation.types import ClassicalOrbitalElements, OrbitState, SimulationConfig


def create_poliastro_orbit(elements: ClassicalOrbitalElements) -> Orbit:
    """
    Create a poliastro Orbit object from classical orbital elements.

    Parameters
    ----------
    elements:
        Classical orbital elements.

    Returns
    -------
    Orbit
        Poliastro orbit object.
    """

    return Orbit.from_classical(
        attractor=Earth,
        a=elements.semi_major_axis,
        ecc=elements.eccentricity,
        inc=elements.inclination,
        raan=elements.raan,
        argp=elements.argument_of_perigee,
        nu=elements.true_anomaly,
        epoch=elements.epoch,
    )


def generate_time_grid(config: SimulationConfig) -> np.ndarray:
    """
    Generate simulation time grid.

    Parameters
    ----------
    config:
        Simulation configuration.

    Returns
    -------
    np.ndarray
        Time vector in seconds.
    """

    if config.duration_s <= 0.0:
        raise ValueError("Simulation duration must be positive.")

    if config.time_step_s <= 0.0:
        raise ValueError("Simulation time step must be positive.")

    return np.arange(0.0, config.duration_s + config.time_step_s, config.time_step_s)


def propagate_orbit(elements: ClassicalOrbitalElements, config: SimulationConfig) -> OrbitState:
    """
    Propagate orbit and return ECI position and velocity time series.

    Parameters
    ----------
    elements:
        Classical orbital elements.
    config:
        Simulation configuration.

    Returns
    -------
    OrbitState
        Time, ECI position and ECI velocity arrays.
    """

    orbit = create_poliastro_orbit(elements)
    times_s = generate_time_grid(config)

    r_eci_list = []
    v_eci_list = []

    for t_s in times_s:
        propagated_orbit = orbit.propagate(t_s * u.s)

        r_eci_m = propagated_orbit.r.to(u.m).value
        v_eci_mps = propagated_orbit.v.to(u.m / u.s).value

        r_eci_list.append(r_eci_m)
        v_eci_list.append(v_eci_mps)

    r_eci = np.array(r_eci_list)
    v_eci = np.array(v_eci_list)
    times_utc = elements.epoch + times_s * u.s

    return OrbitState(t_s=times_s, t_utc=times_utc, r_eci_m=r_eci, v_eci_mps=v_eci)


class PoliastroKeplerPropagator:
    """Adapter exposing the current poliastro two-body propagator."""

    def propagate(self, elements: ClassicalOrbitalElements, config: SimulationConfig) -> OrbitState:
        """Propagate an orbit with poliastro."""

        return propagate_orbit(elements, config)
