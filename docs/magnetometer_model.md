# Magnetometer Model

## Purpose

The geomagnetic and attitude pipeline produces the ideal magnetic-field vector
in spacecraft body coordinates:

$$
\mathbf{B}_\mathrm{body}
= \mathbf{R}_{\mathrm{eci}\leftarrow\mathrm{body}}^\mathsf{T}\,
  \mathbf{B}_\mathrm{eci}
$$

The magnetometer model turns this ideal body-frame vector into a sensor
measurement. It does not recompute the magnetic field; it only adds a simple
sensor error model.

## Measurement Equation

For each output sample $k$:

$$
\mathbf{B}_{\mathrm{mag},k}
= \mathbf{B}_{\mathrm{body},k}
  + \mathbf{b}_\mathrm{body}
  + \boldsymbol{\eta}_k
$$

where:

- $\mathbf{B}_{\mathrm{body},k}$ is the ideal body-frame magnetic field in tesla,
- $\mathbf{b}_\mathrm{body}$ is a constant 3-axis bias in tesla,
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
bias_body_t  shape (3,), units T
noise_std_t  scalar or shape (3,), units T
seed         optional RNG seed
```

The model implements the `Magnetometer` interface through:

```text
measure(b_body_t) -> b_magnetometer_t
```

`SimulationRunner` owns the magnetometer dependency:

```text
magnetometer_model: Magnetometer
```

This keeps the runner independent of the concrete sensor implementation. The
default runner uses `MagnetometerModel()` with zero bias and zero noise.

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
        bias_body_t=np.array([0.3e-6, -0.2e-6, 0.1e-6]),
        noise_std_t=1.0e-6,
        seed=42,
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

This plot shows the three measured body-frame components over time.
