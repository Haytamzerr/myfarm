"""Vegetation indices from Sentinel-2 surface reflectance (values between 0 and 1).

| Index | Formula                                  | Sentinel-2 bands | What it shows                     |
|-------|------------------------------------------|------------------|-----------------------------------|
| NDVI  | (NIR - Red) / (NIR + Red)                | B8, B4           | how much green vegetation          |
| GNDVI | (NIR - Green) / (NIR + Green)            | B8, B3           | chlorophyll, more sensitive late   |
| NDRE  | (NIR - RedEdge) / (NIR + RedEdge)        | B8, B5           | chlorophyll / nitrogen, dense crops|
| EVI   | 2.5 (NIR - Red) / (NIR + 6Red - 7.5Blue + 1) | B8, B4, B2   | vegetation, corrects soil and haze |
| NDMI  | (NIR - SWIR) / (NIR + SWIR)              | B8, B11          | water in the leaves (water stress) |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = ["blue", "green", "red", "rededge", "nir", "swir16"]
INDICES = ["NDVI", "GNDVI", "NDRE", "EVI", "NDMI"]


def _normalized_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    return np.where(np.isfinite(out), out, np.nan)


def ndvi(nir, red):
    return _normalized_difference(nir, red)


def gndvi(nir, green):
    return _normalized_difference(nir, green)


def ndre(nir, rededge):
    return _normalized_difference(nir, rededge)


def ndmi(nir, swir16):
    return _normalized_difference(nir, swir16)


def evi(nir, red, blue):
    nir, red, blue = (np.asarray(v, dtype=float) for v in (nir, red, blue))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    return np.where(np.isfinite(out), out, np.nan)


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add every index as a column to a table with the band columns in BANDS."""
    missing = [b for b in BANDS if b not in df]
    if missing:
        raise ValueError(f"missing bands: {missing}")
    out = df.copy()
    out["NDVI"] = ndvi(df.nir, df.red)
    out["GNDVI"] = gndvi(df.nir, df.green)
    out["NDRE"] = ndre(df.nir, df.rededge)
    out["EVI"] = evi(df.nir, df.red, df.blue)
    out["NDMI"] = ndmi(df.nir, df.swir16)
    return out
