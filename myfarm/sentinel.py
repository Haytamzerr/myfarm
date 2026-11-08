"""Real Sentinel-2 data for the parcel, free and without an account.

Sentinel-2 (ESA Copernicus) photographs every field on Earth every ~5 days at 10 m resolution.
The images are published as Cloud-Optimised GeoTIFFs on AWS and indexed in a STAC catalogue
(Element 84 Earth Search), so we can:

1. search the catalogue for images of the parcel's bounding box in a date range;
2. keep the least cloudy one;
3. read only the pixels under our grid points over HTTP (no full image download);
4. drop pixels the scene classification marks as cloud, shadow or snow.

Needs the optional dependencies: pip install pystac-client rasterio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from myfarm.parcel import Parcel

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
# Earth Search asset name -> our band name
ASSETS = {"blue": "blue", "green": "green", "red": "red", "rededge1": "rededge",
          "nir": "nir", "swir16": "swir16"}  # fmt: skip
# Scene classification (SCL) classes that are not usable ground pixels
BAD_SCL = {0, 1, 3, 8, 9, 10, 11}  # no data, saturated, cloud shadow, clouds, cirrus, snow


def find_scene(parcel: Parcel, start: str, end: str, max_cloud: float = 20.0):
    """The least cloudy Sentinel-2 scene over the parcel between two dates (YYYY-MM-DD)."""
    from pystac_client import Client

    search = Client.open(STAC_URL).search(
        collections=[COLLECTION],
        bbox=parcel.bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=50,
    )
    items = list(search.items())
    if not items:
        raise LookupError(f"no scene under {max_cloud}% cloud between {start} and {end}")
    return min(items, key=lambda item: item.properties.get("eo:cloud_cover", 100))


def _scale_offset(asset) -> tuple[float, float]:
    """Digital numbers -> reflectance. Recent scenes store an offset of -0.1 (baseline 04.00)."""
    bands = asset.extra_fields.get("raster:bands") or [{}]
    return float(bands[0].get("scale", 0.0001)), float(bands[0].get("offset", 0.0))


def _sample(href: str, points: np.ndarray) -> np.ndarray:
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(href) as src:
        xs, ys = transform("EPSG:4326", src.crs, points[:, 0].tolist(), points[:, 1].tolist())
        return np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)


def read_scene(item, points: np.ndarray) -> pd.DataFrame:
    """Reflectance of every band at each grid point, cloudy pixels set to NaN."""
    import rasterio

    table = {"lon": points[:, 0], "lat": points[:, 1]}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        for asset_name, band in ASSETS.items():
            asset = item.assets[asset_name]
            raw = _sample(asset.href, points)
            scale, offset = _scale_offset(asset)
            table[band] = np.where(raw == 0, np.nan, raw * scale + offset)
        if "scl" in item.assets:
            scl = _sample(item.assets["scl"].href, points)
            bad = np.isin(scl, list(BAD_SCL))
            for band in ASSETS.values():
                table[band] = np.where(bad, np.nan, table[band])
    return pd.DataFrame(table)


def fetch(parcel: Parcel, points: np.ndarray, start: str, end: str, max_cloud: float = 20.0):
    """(table of bands, scene metadata) for the best scene in the date range."""
    item = find_scene(parcel, start, end, max_cloud)
    meta = {
        "source": "sentinel-2",
        "scene": item.id,
        "date": str(item.datetime.date()) if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
    }
    return read_scene(item, points), meta
