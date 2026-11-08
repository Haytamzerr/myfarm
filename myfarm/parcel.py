"""The farm parcel: read from a KML file (drawn in Google Earth), measured and sampled.

Coordinates are kept as (longitude, latitude), the order used by KML, GeoJSON and shapely.
For areas and distances the parcel is projected onto a flat plane centred on it
(equirectangular projection): at the scale of a field (a few hundred metres) the error is
far below a GPS error.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

EARTH_RADIUS_M = 6_371_008.8
KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


@dataclass
class Parcel:
    name: str
    coords: list[tuple[float, float]]  # (lon, lat), closed ring

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.coords)

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.polygon.centroid
        return (c.x, c.y)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)"""
        return self.polygon.bounds

    def to_metres(self, lon, lat):
        """Project lon/lat (degrees) to x/y metres around the parcel centroid."""
        lon0, lat0 = self.centroid
        x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
        return x, y

    def to_degrees(self, x, y):
        lon0, lat0 = self.centroid
        lon = lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
        lat = lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS_M)
        return lon, lat

    @property
    def area_m2(self) -> float:
        lon, lat = zip(*self.coords)
        x, y = self.to_metres(lon, lat)
        return float(Polygon(zip(x, y)).area)

    @property
    def area_ha(self) -> float:
        return self.area_m2 / 10_000

    def grid(self, spacing_m: float = 10.0) -> np.ndarray:
        """Points every `spacing_m` metres inside the parcel, as an (n, 2) array of lon, lat.

        10 m is the resolution of Sentinel-2's visible and near-infrared bands, so one point
        stands for one satellite pixel.
        """
        lon, lat = zip(*self.coords)
        x, y = self.to_metres(lon, lat)
        shape = Polygon(zip(x, y))
        min_x, min_y, max_x, max_y = shape.bounds
        xs = np.arange(min_x + spacing_m / 2, max_x, spacing_m)
        ys = np.arange(min_y + spacing_m / 2, max_y, spacing_m)
        gx, gy = np.meshgrid(xs, ys)
        inside = np.array([shape.contains(Point(px, py)) for px, py in zip(gx.ravel(), gy.ravel())])
        glon, glat = self.to_degrees(gx.ravel()[inside], gy.ravel()[inside])
        return np.column_stack([glon, glat])

    def to_geojson(self) -> dict:
        return {
            "type": "Feature",
            "properties": {"name": self.name, "area_ha": round(self.area_ha, 2)},
            "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in self.coords]]},
        }


def load_kml(path: str | Path) -> Parcel:
    """The first polygon of a KML file (Google Earth / Google My Maps export)."""
    root = ET.parse(path).getroot()
    placemark_name = root.find(".//kml:Document/kml:name", KML_NS)
    ring = root.find(".//kml:Polygon//kml:coordinates", KML_NS)
    if ring is None or not ring.text:
        raise ValueError(f"no polygon found in {path}")
    coords = []
    for chunk in ring.text.split():
        lon, lat, *_ = (float(v) for v in chunk.split(","))
        coords.append((lon, lat))
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    if len(coords) < 4:
        raise ValueError("a polygon needs at least 3 distinct points")
    name = placemark_name.text.strip() if placemark_name is not None and placemark_name.text else "parcel"
    return Parcel(name=name, coords=coords)
