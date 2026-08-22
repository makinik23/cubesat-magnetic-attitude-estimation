# Magnetometer Model

## Purpose

The geomagnetic and attitude pipeline produces the ideal magnetic-field vector
in spacecraft body coordinates:

$$
\mathbf{B}_\mathrm{body}
= \mathbf{R}_{\mathrm{eci}\leftarrow\mathrm{body}}^\mathsf{T}\,
  \mathbf{B}_\mathrm{eci}
$$

The magnetometer model turns this ideal body-frame vector into a sensor-frame
measurement. It does not recompute the magnetic field; it applies the configured
sensor axes, stores the configured sensor positions, and adds a simple sensor
error model.

## Measurement Equation

For each output sample $k$:

$$
\mathbf{B}_{\mathrm{sensor},k}
= \mathbf{C}_{\mathrm{sensor}\leftarrow\mathrm{body}}\,
  \mathbf{B}_{\mathrm{body},k}
  + \mathbf{b}_\mathrm{sensor}
  + \boldsymbol{\eta}_k
$$

where:

- $\mathbf{B}_{\mathrm{body},k}$ is the ideal body-frame magnetic field in tesla,
- $\mathbf{C}_{\mathrm{sensor}\leftarrow\mathrm{body}}$ maps body components to
  the sensor channel axes,
- $\mathbf{b}_\mathrm{sensor}$ is a constant 3-axis bias in the sensor frame,
- $\boldsymbol{\eta}_k$ is zero-mean Gaussian noise in tesla.

The noise model is:

$$
\boldsymbol{\eta}_k \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})
$$

For per-axis noise:

$$
\boldsymbol{\Sigma}
=
\operatorname{diag}(\sigma_x^2,\sigma_y^2,\sigma_z^2)
$$

For scalar `noise_std_t = sigma`:

$$
\boldsymbol{\Sigma} = \sigma^2\mathbf{I}
$$

## Implementation

The default implementation is `MagnetometerModel`.

Constructor parameters:

```text
bias_sensor_t              shape (3,), units T
noise_std_t                scalar or shape (3,), units T
seed                       optional RNG seed
rotation_sensor_from_body  shape (3, 3), dimensionless
positions_body_m           shape (3, 3), units m
```

The model implements the `Magnetometer` interface through:

```text
measure(b_body_t) -> b_magnetometer_t
```

`SimulationRunner` owns the magnetometer dependency:

```text
magnetometer_model: Magnetometer
```

This keeps the runner independent of the concrete sensor implementation.

## Current Geometry

The current default configuration uses three orthogonal sensor axes parallel to
the spacecraft body axes:

```text
rotation_sensor_from_body = I
```

The three physical sensor positions are offset from the body-frame origin:

```text
X-channel sensor: [ 0.02,  0.03,  0.00] m
Y-channel sensor: [ 0.00, -0.02,  0.03] m
Z-channel sensor: [-0.03,  0.00, -0.02] m
```

Thus the channels are orthogonal and body-parallel, but the sensors are not
located on the body-axis centerlines. The positions are stored and validated.
They do not yet alter the measurement because the current magnetic-field model
provides one uniform body-frame field vector at the spacecraft center. A
field-gradient model or a local magnetic-disturbance model would be needed for
the translations to change the measured values.

## Configuring Visible Noise

A realistic magnetometer noise level can be small compared with the orbital
magnetic-field variation. For example, tens of nanotesla are only hundredths of
a microtesla, while the plot range is often tens of microtesla.

For a visibly noisy demonstration, inject a model with a larger standard
deviation:

```python
import numpy as np

from simulation.pipeline import SimulationRunner
from simulation.sensors import MagnetometerModel

runner = SimulationRunner(
    magnetometer_model=MagnetometerModel(
        bias_sensor_t=np.array([0.3e-6, -0.2e-6, 0.1e-6]),
        noise_std_t=1.0e-6,
        seed=42,
        rotation_sensor_from_body=np.eye(3),
        positions_body_m=np.array([
            [0.02, 0.03, 0.0],
            [0.0, -0.02, 0.03],
            [-0.03, 0.0, -0.02],
        ]),
    )
)
```

All values above are in tesla. The corresponding plot is shown in microtesla.

## Outputs

The results table includes the measured magnetometer components:

```text
Bx_magnetometer_T
By_magnetometer_T
Bz_magnetometer_T
B_magnetometer_norm_T
```

The pipeline also writes:

```text
magnetometer_measurement.png
```

This plot shows the three measured sensor-frame components over time.
