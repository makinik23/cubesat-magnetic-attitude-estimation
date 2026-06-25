# Orbital Mechanics

## Two-Body Model

The orbit propagation starts from the classical two-body equation:

```text
r_ddot = -mu * r / ||r||^3
```

where:

- `r` is the spacecraft position vector in an inertial frame,
- `mu` is Earth's gravitational parameter,
- perturbations are neglected at this stage.

The code delegates the propagation to `poliastro`, but the state still follows
this model conceptually.

## Classical Orbital Elements

The initial orbit is described by:

```text
a      semi-major axis
e      eccentricity
i      inclination
Omega  right ascension of ascending node
omega  argument of perigee
nu     true anomaly
t0     epoch
```

In the default configuration:

- altitude is about `500 km`,
- eccentricity is small: `e = 0.001`,
- inclination is `97 deg`,
- epoch is `2026-01-01T12:00:00 UTC`.

## State Vector

For each simulation time `t`, the propagated orbit gives:

```text
r_eci(t) = [x_eci, y_eci, z_eci]
v_eci(t) = [vx_eci, vy_eci, vz_eci]
```

The CSV also stores:

```text
||r_eci|| = sqrt(x_eci^2 + y_eci^2 + z_eci^2)
||v_eci|| = sqrt(vx_eci^2 + vy_eci^2 + vz_eci^2)
```

## Angular Momentum Direction

The orbital angular momentum direction is:

```text
h = r_eci x v_eci
h_hat = h / ||h||
```

This is used only to define the initial body-frame orientation.
