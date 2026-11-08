"""A simulated Sentinel-2 scene, to run and demo the pipeline without downloading images.

This is NOT a measurement of the parcel. It only produces plausible reflectances with a
spatial structure (a vigorous patch, a dry corner, a gradient along the field), so that the
indices, zones and map have something realistic to show. Use `myfarm.sentinel` for real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from myfarm.parcel import Parcel


def simulate_scene(parcel: Parcel, points: np.ndarray, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, y = parcel.to_metres(points[:, 0], points[:, 1])
    span = max(np.ptp(x), np.ptp(y), 1.0)

    def blob(cx, cy, radius):
        return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * (radius * span) ** 2)))

    # vigour in [0, 1]: a gradient along the field, one healthy patch, one weak patch
    corners = [(np.percentile(x, p), np.percentile(y, q)) for p, q in ((70, 75), (30, 20))]
    vigour = 0.55 + 0.15 * (y - y.mean()) / span
    vigour += 0.30 * blob(*corners[0], 0.18) - 0.35 * blob(*corners[1], 0.15)
    vigour = np.clip(vigour + rng.normal(0, 0.04, len(x)), 0.02, 0.98)
    # leaf water follows vigour, with a drier band along one edge (e.g. a broken drip line)
    moisture = np.clip(vigour - 0.25 * blob(np.percentile(x, 90), y.mean(), 0.25), 0, 1)

    def noise():
        return rng.normal(0, 0.008, len(x))

    bands = pd.DataFrame({
        "blue": 0.06 - 0.02 * vigour + noise(),
        "green": 0.09 - 0.01 * vigour + noise(),
        "red": 0.14 - 0.10 * vigour + noise(),
        "rededge": 0.17 + 0.08 * vigour + noise(),
        "nir": 0.22 + 0.30 * vigour + noise(),
        "swir16": 0.32 - 0.14 * moisture + noise(),
    }).clip(lower=0.001)  # reflectance is never negative
    bands.insert(0, "lat", points[:, 1])
    bands.insert(0, "lon", points[:, 0])
    return bands
