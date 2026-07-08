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

$$
\mathbf{y}
=
\begin{bmatrix}
q_w & q_x & q_y & q_z & \omega_x & \omega_y & \omega_z
\end{bmatrix}^\mathsf{T}
$$

The quaternion is normalized at output samples:

$$
\mathbf{q}
\leftarrow
\frac{\mathbf{q}}{\lVert \mathbf{q} \rVert}
$$

## Time Grid

The simulation uses a fixed output grid:

$$
t_k = k\,\Delta t,
\qquad
k = 0, 1, 2, \ldots, N
$$

with:

$$
t_N \leq t_\mathrm{duration}
$$

The integrator may use internal adaptive steps, but results are evaluated at
the requested output times.

## Important Diagnostics

Use these quantities to detect numerical problems:

| Quantity | Expected behavior |
| --- | --- |
| $\lVert \mathbf{q} \rVert$ | should be 1 |
| $\lVert \mathbf{R}^\mathsf{T}\mathbf{R} - \mathbf{I} \rVert_F$ | should be small |
| $\det(\mathbf{R})$ | should be 1 |
| $\lVert \mathbf{B}_\mathrm{body} \rVert - \lVert \mathbf{B}_\mathrm{eci} \rVert$ | should be near 0 |

## Main Approximations

- Orbit propagation currently uses a two-body model.
- ECI vectors from `poliastro` are treated as GCRS-like for this milestone.
- The attitude model uses a constant body-frame torque, zero by default.
- The default magnetometer has zero bias and zero noise unless a configured
  `MagnetometerModel` is injected into the simulation runner.
