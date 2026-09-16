# AEKF Validation Summary

## Scope

This validation run covers four checks:

- Monte Carlo runs with different measurement-noise seeds and initial AEKF errors.
- Normalized innovation squared (NIS) consistency against chi-square bounds.
- A longer AEKF run covering 5 orbits.
- Current magnetometer geometry and limitations.

## Setup

- Estimated orbital period: 5676.978 s.
- Monte Carlo runs: 100.
- Initial attitude error range: 0 to 5 deg.
- Initial omega error std: 0.05 deg/s per axis.
- Initial bias estimate std: 0.5 uT per axis.
- AEKF angular-rate process-noise std: 1e-05 deg/s per sample.

## Monte Carlo Aggregate

- Mean final attitude error: 0.418 deg.
- 95th percentile final attitude error: 0.925 deg.
- Mean final angular-rate error: 0.00139 deg/s.
- 95th percentile final angular-rate error: 0.00290 deg/s.
- Mean final bias error norm: 0.079 uT.
- 95th percentile final bias error norm: 0.135 uT.
- Mean NIS over runs: 3.020.
- Mean central-95% NIS fraction: 0.948.

## Monte Carlo Attitude Settling

| Threshold | Settled runs | Mean [s] | Median [s] | P95 [s] | Mean [orbits] |
| --- | ---: | ---: | ---: | ---: | ---: |
| < 2 deg | 100/100 | 1151.4 | 950.0 | 2111.5 | 0.20 |
| < 1.5 deg | 100/100 | 1541.3 | 1180.0 | 4081.0 | 0.27 |
| < 1 deg | 97/100 | 2573.1 | 1970.0 | 4696.0 | 0.45 |

## Long Run

- Duration: 28390.0 s (5.00 orbits).
- Final attitude error: 0.770 deg.
- RMS attitude error: 0.427 deg.
- Final angular-rate error: 0.00262 deg/s.
- Mean angular-rate error: 0.000991 deg/s.
- Angular-rate settling below 0.005 deg/s: 1730.0 s.
- Final bias error norm: 0.039 uT.
- Best bias error norm: 0.028 uT at 22290.0 s.
- Mean NIS: 2.993.
- Central-95% NIS fraction: 0.947.

## Magnetometer Model

The current measurement model is:

```text
B_sensor = C_sensor_from_body * R_eci_from_body(q).T * B_eci
           + bias_sensor + noise
```

The current sensor-axis matrix is:

```text
[ 1.000000,  0.000000,  0.000000]
[ 0.000000,  1.000000,  0.000000]
[ 0.000000,  0.000000,  1.000000]
```

The current sensor positions in body coordinates are:

```text
[ 0.020000,  0.030000,  0.000000]
[ 0.000000, -0.020000,  0.030000]
[-0.030000,  0.000000, -0.020000]
```

The positions are stored and validated, but the current field model supplies one
uniform body-frame field vector at the spacecraft center. Therefore these small
translations do not yet alter the measurement unless a field-gradient or local
magnetic-disturbance model is added.
