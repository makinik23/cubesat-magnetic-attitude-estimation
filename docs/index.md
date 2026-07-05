# Mathematical Notes

These notes summarize the mechanics used in the simulation.

Read in this order:

1. [Orbital mechanics](orbital_mechanics.md)
2. [Reference frames](reference_frames.md)
3. [Geomagnetic field](geomagnetic_field.md)
4. [Attitude dynamics](attitude_dynamics.md)
5. [Magnetometer model](magnetometer_model.md)
6. [Numerical methods](numerical_methods.md)

Main conventions:

- Vectors are column vectors.
- Rotation matrices are named `R_target_from_source`.
- Quaternions are scalar-first: `[w, x, y, z]`.
- SI units are used internally unless a column name says otherwise.
- Magnetic field is stored in tesla for vectors and nT for IGRF NED components.
- Magnetometer output is the body-frame field plus configured bias and
  Gaussian noise.
