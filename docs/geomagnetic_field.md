# Geomagnetic Field

## IGRF Model

The geomagnetic field is computed with IGRF through `ppigrf`.

Inputs:

```text
latitude  [deg]
longitude [deg]
altitude  [km]
date      [UTC]
```

IGRF models the main internal magnetic field of Earth. It represents the field
as the gradient of a scalar magnetic potential expanded in spherical harmonics.

## Spherical Harmonic Potential

Outside the source region, the magnetic field is written as:

```text
B = -grad(V)
```

The IGRF internal scalar potential is:

```text
V(r, theta, lambda, t)
  = a * sum_{n=1}^{N} sum_{m=0}^{n}
      (a / r)^(n+1)
      [g_n^m(t) cos(m lambda) + h_n^m(t) sin(m lambda)]
      P_n^m(cos theta)
```

where:

- `r` is geocentric radius,
- `theta` is geocentric colatitude,
- `lambda` is longitude,
- `a` is the IGRF reference Earth radius,
- `N` is the maximum model degree,
- `P_n^m` are Schmidt semi-normalized associated Legendre functions,
- `g_n^m(t)`, `h_n^m(t)` are Gauss coefficients.

For IGRF-14, the maximum degree is:

```text
N = 13
```

## Gauss Coefficients

The coefficients:

```text
g_n^m(t), h_n^m(t)
```

describe the Earth's main magnetic field at a given epoch.

Between tabulated epochs, coefficients are time-interpolated. For near-future
dates, secular variation coefficients are used:

```text
g_n^m(t) ~= g_n^m(t0) + (t - t0) * gdot_n^m
h_n^m(t) ~= h_n^m(t0) + (t - t0) * hdot_n^m
```

## Magnetic Field Components

Using spherical coordinates, the components follow from the potential:

```text
B_r      = -dV/dr
B_theta  = -(1/r) dV/dtheta
B_lambda = -(1/(r sin theta)) dV/dlambda
```

These are then converted to local geodetic components.

In local navigation notation:

```text
B_N  north component
B_E  east component
B_D  down component
```

The relation to spherical components is convention-dependent because IGRF
starts from geocentric coordinates, while the code requests geodetic output
from `ppigrf`.

The useful project-level convention is simply:

```text
B_ned = [B_N, B_E, B_D]
```

## ppigrf Output Convention

`ppigrf.igrf` returns:

```text
B_east, B_north, B_up
```

in nanotesla.

The simulation stores local NED components:

```text
B_N =  B_north
B_E =  B_east
B_D = -B_up
```

So:

```text
B_ned = [B_N, B_E, B_D]
```

## Transformations

The full pipeline is:

```text
lat, lon, h, t -> IGRF -> B_ned
B_ned -> B_ecef
B_ecef -> B_eci
B_eci -> B_body
```

The body-frame field is:

```text
B_body = R_eci_from_body^T * B_eci
```

The norm is invariant under a valid rotation:

```text
||B_body|| = ||B_eci||
```

Small differences can appear only from floating-point roundoff.
