# Numerical Methods

## Floating-Point Type

The numerical code uses `float64` arrays.

This is the default precision for:

- orbit vectors,
- magnetic-field vectors,
- quaternions,
- angular velocity,
- rotation matrices.

## Attitude Integration

Attitude is integrated with:

```text
scipy.integrate.solve_ivp
method = DOP853
rtol   = 1e-10
atol   = 1e-12
```

The integrated state is:

```text
y = [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
```

The quaternion is normalized at output samples.

## Time Grid

The simulation uses a fixed output grid:

```text
t = 0, dt, 2 dt, ..., duration
```

The integrator may use internal adaptive steps, but results are evaluated at
the requested output times.

## Important Diagnostics

Use these quantities to detect numerical problems:

```text
||q||                         should be 1
||R^T R - I||_F               should be small
det(R)                        should be 1
||B_body|| - ||B_eci||         should be near 0
```

## Main Approximations

- Orbit propagation currently uses a two-body model.
- ECI vectors from `poliastro` are treated as GCRS-like for this milestone.
- The ECEF-to-ECI vector rotation is estimated locally using displaced points.
- The attitude model uses a constant body-frame torque, zero by default.
