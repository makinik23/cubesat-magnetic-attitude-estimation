# CesiumJS visualisation

This folder contains the static CesiumJS viewer and the Python exporter that
turns the simulation DataFrame into CZML.

Running the main pipeline creates:

- `outputs/visualisation/index.html`
- `outputs/visualisation/orbit.czml`

Serve the generated directory over HTTP before opening the viewer:

```bash
cd outputs/visualisation
python -m http.server 8000
```

Then open `http://localhost:8000`.

The viewer uses CesiumJS from the official CDN and the generated CZML file from
the same directory.

When attitude columns are present, the exporter also writes sampled
`orientation.unitQuaternion` values. The simulation quaternions are converted
from `ECI_from_body` into Cesium's Earth-fixed frame before export.
