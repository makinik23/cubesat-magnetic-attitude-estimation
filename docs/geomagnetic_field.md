# Geomagnetic Field

## IGRF Model

The geomagnetic field is computed with IGRF through `ppigrf`.

Inputs:

| Quantity | Unit |
| --- | --- |
| latitude | deg |
| longitude | deg |
| altitude | km |
| date | UTC |

IGRF models the main internal magnetic field of Earth. It represents the field
as the gradient of a scalar magnetic potential expanded in spherical harmonics.

## Spherical Harmonic Potential

Outside the source region, the magnetic field is written as:

$$
\mathbf{B} = -\nabla V
$$

The IGRF internal scalar potential is:

$$
V(r,\theta,\lambda,t)
= a
  \sum_{n=1}^{N}
  \sum_{m=0}^{n}
  \left(\frac{a}{r}\right)^{n+1}
  \left[
    g_n^m(t)\cos(m\lambda)
    + h_n^m(t)\sin(m\lambda)
  \right]
  P_n^m(\cos\theta)
$$

where:

- $r$ is geocentric radius,
- $\theta$ is geocentric colatitude,
- $\lambda$ is longitude,
- $a$ is the IGRF reference Earth radius,
- $N$ is the maximum model degree,
- $P_n^m$ are Schmidt semi-normalized associated Legendre functions,
- $g_n^m(t)$ and $h_n^m(t)$ are Gauss coefficients.

For IGRF-14, the maximum degree is:

$$
N = 13
$$

## Gauss Coefficients

The coefficients $g_n^m(t)$ and $h_n^m(t)$ describe the Earth's main magnetic
field at a given epoch.

Between tabulated epochs, coefficients are time-interpolated. For near-future
dates, secular variation coefficients are used:

$$
g_n^m(t)
\approx g_n^m(t_0) + (t - t_0)\dot{g}_n^m
$$

$$
h_n^m(t)
\approx h_n^m(t_0) + (t - t_0)\dot{h}_n^m
$$

## Magnetic Field Components

Using spherical coordinates, the components follow from the potential:

$$
B_r = -\frac{\partial V}{\partial r}
$$

$$
B_\theta = -\frac{1}{r}\frac{\partial V}{\partial \theta}
$$

$$
B_\lambda
= -\frac{1}{r\sin\theta}\frac{\partial V}{\partial \lambda}
$$

These are then converted to local geodetic components.

In local navigation notation:

| Symbol | Meaning |
| --- | --- |
| $B_N$ | north component |
| $B_E$ | east component |
| $B_D$ | down component |

The relation to spherical components is convention-dependent because IGRF
starts from geocentric coordinates, while the code requests geodetic output
from `ppigrf`.

The useful project-level convention is:

$$
\mathbf{B}_\mathrm{ned}
=
\begin{bmatrix}
B_N & B_E & B_D
\end{bmatrix}^\mathsf{T}
$$

## ppigrf Output Convention

`ppigrf.igrf` returns:

$$
(B_\mathrm{east}, B_\mathrm{north}, B_\mathrm{up})
$$

in nanotesla.

The simulation stores local NED components:

$$
B_N = B_\mathrm{north}
$$

$$
B_E = B_\mathrm{east}
$$

$$
B_D = -B_\mathrm{up}
$$

So:

$$
\mathbf{B}_\mathrm{ned}
=
\begin{bmatrix}
B_N & B_E & B_D
\end{bmatrix}^\mathsf{T}
$$

## Transformations

The full pipeline is:

```text
lat, lon, h, t -> IGRF -> B_ned
B_ned -> B_ecef
B_ecef -> B_eci
B_eci -> B_body
B_body -> B_magnetometer
```

The body-frame field is:

$$
\mathbf{B}_\mathrm{body}
= \mathbf{R}_{\mathrm{eci}\leftarrow\mathrm{body}}^\mathsf{T}\,
  \mathbf{B}_\mathrm{eci}
$$

The norm is invariant under a valid rotation:

$$
\lVert \mathbf{B}_\mathrm{body} \rVert
=
\lVert \mathbf{B}_\mathrm{eci} \rVert
$$

Small differences can appear only from floating-point roundoff.

The magnetometer stage is not another geomagnetic-field model. It uses
`B_body` as the ideal body-frame field and applies the configured sensor bias
and Gaussian noise.
