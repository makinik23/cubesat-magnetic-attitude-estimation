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
- `simulation/config.py` - default orbit and simulation settings.
- `simulation/orbit_provider.py` - orbit propagation.
- `simulation/frames.py` - ECI/ECEF/geodetic transformations.
- `simulation/geomagnetic.py` - IGRF magnetic-field computation.
- `simulation/attitude.py` - quaternion attitude dynamics.
- `simulation/plots.py` - plots and GIF animation.
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
- `outputs/attitude_orientation.png`
- `outputs/attitude_cube.gif`

## Checks

```bash
hatch run format
hatch run check
```

`hatch run check` runs:

- `black --check .`
- `flake8 .`
- `mypy .`

## Notes

- Quaternions use scalar-first convention: `[w, x, y, z]`.
- Rotation matrices are named as `R_target_from_source`.
- Body-frame magnetic field is computed as `B_body = R_eci_from_body.T @ B_eci`.
- The CSV includes attitude sanity checks: `R^T R - I` and `det(R)`.
