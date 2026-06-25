# Reference Frames

## ECI

ECI is the inertial frame used by the orbit propagation.

The pipeline assumes that the vectors from `poliastro` can be treated as
GCRS-like ECI vectors for the current engineering model.

## ECEF

ECEF is the Earth-fixed frame.

The position transformation is:

```text
r_ecef(t) = R_ecef_from_eci(t) * r_eci(t)
```

In code this is performed with Astropy frame transformations.

## Geodetic Coordinates

ECEF position is converted to WGS84 geodetic coordinates:

```text
r_ecef -> latitude, longitude, altitude
```

The project stores:

- `lat_deg`, `lon_deg`,
- `lat_rad`, `lon_rad`,
- `alt_m`, `alt_km`.

## NED Basis

At geodetic latitude `phi` and longitude `lambda`, the local NED basis in ECEF
coordinates is:

```text
north = [-sin(phi) cos(lambda), -sin(phi) sin(lambda),  cos(phi)]
east  = [-sin(lambda),            cos(lambda),           0       ]
down  = [-cos(phi) cos(lambda), -cos(phi) sin(lambda), -sin(phi)]
```

For a vector `b_ned = [B_N, B_E, B_D]`:

```text
b_ecef = B_N * north + B_E * east + B_D * down
```

## Vector Transformation Back to ECI

The current code estimates `R_eci_from_ecef` locally by transforming a base ECEF
point and three one-meter displaced points.

This is explicit and robust for the current milestone, but not the fastest
possible method.
