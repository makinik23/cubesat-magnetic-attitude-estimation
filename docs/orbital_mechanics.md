# Orbital Mechanics

## Two-Body Model

The orbit propagation starts from the classical two-body equation:

$$
\ddot{\mathbf{r}}
= -\mu\,\frac{\mathbf{r}}{\lVert \mathbf{r} \rVert^3}
$$

where:

- $\mathbf{r}$ is the spacecraft position vector in an inertial frame,
- $\mu$ is Earth's gravitational parameter,
- perturbations are neglected at this stage.

The code delegates the propagation to `poliastro`, but the state still follows
this model conceptually.

## Classical Orbital Elements

The initial orbit is described by:

| Symbol | Meaning |
| --- | --- |
| $a$ | semi-major axis |
| $e$ | eccentricity |
| $i$ | inclination |
| $\Omega$ | right ascension of ascending node |
| $\omega$ | argument of perigee |
| $\nu$ | true anomaly |
| $t_0$ | epoch |

In the default configuration:

- altitude is about $500\,\mathrm{km}$,
- eccentricity is small: $e = 0.001$,
- inclination is $97^\circ$,
- epoch is `2026-01-01T12:00:00 UTC`.

## State Vector

For each simulation time $t$, the propagated orbit gives:

$$
\mathbf{r}_\mathrm{eci}(t)
=
\begin{bmatrix}
x_\mathrm{eci} & y_\mathrm{eci} & z_\mathrm{eci}
\end{bmatrix}^\mathsf{T}
$$

$$
\mathbf{v}_\mathrm{eci}(t)
=
\begin{bmatrix}
v_{x,\mathrm{eci}} & v_{y,\mathrm{eci}} & v_{z,\mathrm{eci}}
\end{bmatrix}^\mathsf{T}
$$

The CSV also stores:

$$
\lVert \mathbf{r}_\mathrm{eci} \rVert
= \sqrt{x_\mathrm{eci}^2 + y_\mathrm{eci}^2 + z_\mathrm{eci}^2}
$$

$$
\lVert \mathbf{v}_\mathrm{eci} \rVert
= \sqrt{v_{x,\mathrm{eci}}^2 + v_{y,\mathrm{eci}}^2 + v_{z,\mathrm{eci}}^2}
$$

## Angular Momentum Direction

The orbital angular momentum direction is:

$$
\mathbf{h}
= \mathbf{r}_\mathrm{eci} \times \mathbf{v}_\mathrm{eci}
$$

$$
\hat{\mathbf{h}}
= \frac{\mathbf{h}}{\lVert \mathbf{h} \rVert}
$$

This is used only to define the initial body-frame orientation.
