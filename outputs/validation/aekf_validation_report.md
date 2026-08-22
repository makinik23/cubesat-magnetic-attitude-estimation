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

## Monte Carlo Aggregate

- Mean final attitude error: 0.575 deg.
- 95th percentile final attitude error: 1.133 deg.
- Mean final bias error norm: 0.090 uT.
- 95th percentile final bias error norm: 0.152 uT.
- Mean NIS over runs: 2.985.
- Mean central-95% NIS fraction: 0.949.

## Monte Carlo Attitude Settling

| Threshold | Settled runs | Mean [s] | Median [s] | P95 [s] | Mean [orbits] |
| --- | ---: | ---: | ---: | ---: | ---: |
| < 2 deg | 100/100 | 1193.6 | 950.0 | 2230.0 | 0.21 |
| < 1.5 deg | 99/100 | 2026.5 | 1220.0 | 5705.0 | 0.36 |
| < 1 deg | 91/100 | 4678.7 | 5640.0 | 5980.0 | 0.82 |

## Long Run

- Duration: 28390.0 s (5.00 orbits).
- Final attitude error: 1.248 deg.
- RMS attitude error: 0.640 deg.
- Final bias error norm: 0.055 uT.
- Best bias error norm: 0.037 uT at 10490.0 s.
- Mean NIS: 2.980.
- Central-95% NIS fraction: 0.949.

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
