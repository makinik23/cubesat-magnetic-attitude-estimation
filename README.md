# CubeSat Magnetic Attitude Estimation

Small simulation pipeline for a CubeSat orbit, geomagnetic field, and rigid-body
attitude motion.

## What It Does

- Propagates a default LEO orbit from classical orbital elements.
- Converts ECI position to ECEF and geodetic coordinates.
- Computes the IGRF magnetic field with `ppigrf`.
- Propagates rigid-body attitude with `scipy.integrate.solve_ivp`.
- Projects the magnetic-field vector into the spacecraft body frame.
- Exports a CSV, static plots, and an attitude cube animation.

## Project Layout

- `main.py` - entry point.
- `simulation/config.py` - configuration loading and validation.
- `simulation/types.py` - shared configuration and state dataclasses.
- `simulation/interfaces.py` - strategy contracts for replaceable components.
- `simulation/runner.py` - end-to-end simulation orchestration.
- `simulation/orbit_provider.py` - poliastro orbit propagation adapter.
- `simulation/frames.py` - Astropy/pymap3d frame transformations.
- `simulation/geomagnetic.py` - ppigrf IGRF magnetic-field adapter.
- `simulation/attitude.py` - solve_ivp quaternion attitude dynamics.
- `simulation/results.py` - result table assembly, CSV export, and sanity checks.
- `simulation/plots.py` - plots and GIF animation.
- `simulation/settings/` - YAML input files for orbit and satellite parameters.
- `docs/` - concise mathematical notes.
- `outputs/` - generated CSV, plots, and animation.

## Setup

Requires Python 3.11+ and Hatch.

```bash
hatch env create
```

## Run

```bash
hatch run simulate
```

Generated outputs:

- `outputs/orbit_timeseries.csv`
- `outputs/position_eci.png`
- `outputs/r_eci_time.png`
- `outputs/position_norm.png`
- `outputs/velocity_norm.png`
- `outputs/orbit_3d.png`
- `outputs/magnetic_field_eci.png`
- `outputs/magnetic_field_norm.png`
- `outputs/magnetic_field_body.png`
- `outputs/magnetic_field_body_norm.png`
- `outputs/magnetometer_measurement.png`
- `outputs/attitude_orientation.png`
- `outputs/attitude_quaternion.png`
- `outputs/angular_velocity_body.png`
- `outputs/attitude_cube.gif`

## Checks

```bash
hatch run format
hatch run check
hatch run unit-tests
```

`hatch run check` runs:

- `black --check .`
- `flake8 .`
- `mypy .`

The GitHub Actions CI workflow runs format checks, static analysis, and
unit tests on pushes and pull requests.

## Notes

- Quaternions use scalar-first convention: $\mathbf{q} = [q_w, q_x, q_y, q_z]^{\mathsf{T}}$.
- Rotation matrices are named as `R_target_from_source`.
- Body-frame magnetic field is computed as $\mathbf{B}_b = \mathbf{R}_{eb}^{\mathsf{T}}\mathbf{B}_e$, where $e$ is ECI and $b$ is body frame.
- The CSV includes attitude sanity checks:
  $\mathbf{R}^{\mathsf{T}}\mathbf{R} - \mathbf{I}$ and $\det(\mathbf{R})$.
