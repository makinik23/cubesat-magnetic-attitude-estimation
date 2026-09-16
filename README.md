# CubeSat Magnetic Attitude Estimation

Magnetometer-only attitude estimation for a CubeSat 3U.

## Estimation Pipeline

The end-to-end flow is:

1. Propagate a default LEO orbit from classical orbital elements.
2. Convert position through ECI, ECEF, and geodetic frames.
3. Compute the IGRF magnetic field with `ppigrf`.
4. Propagate the truth attitude with rigid-body dynamics.
5. Project the magnetic-field vector into the spacecraft body frame.
6. Generate biased and noisy magnetometer measurements in the configured sensor frame.
7. Estimate attitude with the selected Kalman filter.
8. Export time-series data, plots, validation summaries, and consistency metrics.

## 10-State AEKF

The current primary estimator is an additive extended Kalman filter with the
state:

```text
x = [q, omega_body, mag_bias_sensor]
```

The prediction step propagates the quaternion and body angular velocity with
the rigid-body dynamics model. The update step uses only the 3-axis
magnetometer measurement, comparing it with the IGRF field projected into the
sensor frame through the estimated attitude. After each prediction and update,
the quaternion part of the state is normalized back to unit length.

## Estimator Convergence

The table is intentionally shaped so later estimator variants can be added next
to the current AEKF.

| Estimator | State | Measurement | Main tuning | Long-run final attitude error | Long-run RMS attitude error | Monte Carlo final attitude error, mean | Monte Carlo final attitude error, p95 | Runs settled below 1 deg | Mean NIS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10-state AEKF | `q`, `omega_body`, `mag_bias_sensor` | 3-axis magnetometer | `Q_omega = (1e-5 deg/s)^2` | 0.770 deg | 0.427 deg | 0.418 deg | 0.925 deg | 97/100 | 3.020 |

Angular-rate convergence uses the norm of the `omega_body` estimation error:

| Estimator | Long-run settled below 0.005 deg/s | Long-run final omega error | Long-run mean omega error | Monte Carlo final omega error, mean | Monte Carlo final omega error, p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10-state AEKF | 1730 s (0.30 orbit) | 0.00262 deg/s | 0.000991 deg/s | 0.00139 deg/s | 0.00290 deg/s |

Validation setup:

- Monte Carlo validation uses 100 runs with randomized measurement-noise seeds
  and randomized initial AEKF errors.
- The long-run validation covers 5.00 orbits.
- The 10-state AEKF settles below 1 deg in the long run after 1920 s.
- Its angular-rate error norm settles below 0.005 deg/s after 1730 s in the
  same long-run validation.
- For a 3D magnetometer measurement, the expected NIS mean is close to 3; the
  current Monte Carlo mean is 3.020.

## Project Layout

- `main.py` - default run entry point.
- `simulation/estimation/` - Kalman-family attitude estimators.
- `simulation/validation/` - Monte Carlo, NIS, and long-run validation workflow.
- `simulation/pipeline/` - orchestration of truth generation, measurements, and estimators.
- `simulation/config/` - YAML configuration loading and validation.
- `simulation/types.py` - shared configuration and state dataclasses.
- `simulation/interfaces.py` - strategy contracts for replaceable components.
- `simulation/orbit/` - poliastro orbit propagation adapter.
- `simulation/frames/` - Astropy/pymap3d frame transformations.
- `simulation/magnetic/` - ppigrf IGRF magnetic-field adapter.
- `simulation/attitude/` - rotations, body projection, and rigid-body dynamics.
- `simulation/sensors/` - magnetometer measurement model.
- `simulation/io/` - result table assembly, CSV export, and sanity checks.
- `simulation/visualization/` - plots and GIF animation.
- `simulation/settings/` - YAML input files for orbit and satellite parameters.
- `docs/` - concise mathematical notes.
- `outputs/` - generated CSV files, plots, animations, and validation artifacts.

## Setup

Requires Python 3.11+ and Hatch.

```bash
hatch env create
```

## Run The Default Estimation Case

```bash
hatch run simulate
```

Main generated outputs:

- `outputs/orbit_timeseries.csv` - full truth, measurement, and estimator table.
- `outputs/aekf/kalman_timeseries.csv` - 10-state AEKF time series.
- `outputs/aekf/` - 10-state AEKF plots.
- `outputs/attitude_aekf/` - quaternion-only AEKF baseline CSV and plots.
- `outputs/attitude_cube.gif` - truth attitude animation.

## Run Validation

```bash
hatch run validate-aekf --monte-carlo-runs 100
```

This regenerates the convergence table inputs in `outputs/validation/`.

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
- `python -m unittest discover simulation/tests`

## Conventions

- Quaternions use scalar-first convention:
  $\mathbf{q} = [q_w, q_x, q_y, q_z]^{\mathsf{T}}$.
- Rotation matrices are named as `R_target_from_source`.
- Body-frame magnetic field is computed as
  $\mathbf{B}_b = \mathbf{R}_{eb}^{\mathsf{T}}\mathbf{B}_e$, where `e` is ECI
  and `b` is body frame.
- The default 10-state AEKF uses a tight angular-rate process noise of
  `1e-5 deg/s` per sample because the default truth model is torque-free and
  uses known rigid-body dynamics.
- CSV outputs include quaternion norm, rotation orthogonality, determinant,
  estimator error, covariance, magnetometer bias estimate, innovation, and NIS.
