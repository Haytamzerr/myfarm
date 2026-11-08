"""MyFarm web API and map page.

    GET /                    the map dashboard
    GET /api/parcel          the parcel as GeoJSON, with its area
    GET /api/scan            indices and management zones for every 10 m pixel
                             ?source=demo (simulated) | sentinel (real, needs internet)
                             &start=YYYY-MM-DD&end=YYYY-MM-DD&k=3
    GET /api/weather         daily agronomic indicators and irrigation need  ?kc=1.0&base=10
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from myfarm.indices import INDICES, add_indices
from myfarm.parcel import load_kml
from myfarm.synthetic import simulate_scene
from myfarm.weather import agro_summary, irrigation_need, load_open_meteo
from myfarm.zones import management_zones

ROOT = Path(__file__).resolve().parent.parent
PARCEL_FILE = Path(os.environ.get("MYFARM_PARCEL", ROOT / "data" / "firmatech.kml"))
WEATHER_FILE = Path(os.environ.get("MYFARM_WEATHER", ROOT / "data" / "weather_open-meteo.csv"))

app = FastAPI(title="MyFarm", description="Precision agriculture for one parcel", version="1.0")


@lru_cache
def parcel():
    return load_kml(PARCEL_FILE)


@lru_cache
def grid():
    return parcel().grid(10.0)


@lru_cache
def scan(source: str, start: str, end: str):
    if source == "demo":
        return simulate_scene(parcel(), grid()), {"source": "demo", "simulated": True}
    from myfarm import sentinel

    try:
        return sentinel.fetch(parcel(), grid(), start, end)
    except ImportError as error:
        raise HTTPException(501, "install pystac-client and rasterio for real data") from error
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/parcel")
def get_parcel():
    p = parcel()
    return {**p.to_geojson(), "centroid": p.centroid, "pixels": len(grid())}


@app.get("/api/scan")
def get_scan(
    source: str = Query("demo", pattern="^(demo|sentinel)$"),
    start: str = Query("2025-06-01"),
    end: str = Query("2025-07-15"),
    k: int = Query(3, ge=2, le=4),
):
    bands, meta = scan(source, start, end)
    table = add_indices(bands)
    zoned, summary = management_zones(table, k)
    columns = ["lon", "lat", *INDICES, "zone", "zone_name", "low_vegetation", "water_stress"]
    return {
        "meta": {**meta, "valid_pixels": len(zoned), "total_pixels": len(table)},
        "summary": summary,
        "pixels": zoned[columns].round({"lon": 6, "lat": 6, **dict.fromkeys(INDICES, 4)})
        .to_dict(orient="records"),
    }


@app.get("/api/weather")
def get_weather(kc: float = Query(1.0, gt=0, le=1.5), base: float = Query(10.0, ge=0, le=20)):
    if not WEATHER_FILE.exists():
        raise HTTPException(404, "no weather file")
    location, _, daily = load_open_meteo(WEATHER_FILE)
    summary = agro_summary(daily, base_temp=base, kc=kc)
    return {
        "location": location,
        "totals": irrigation_need(summary, parcel().area_ha),
        "days": summary.astype({"date": str}).to_dict(orient="records"),
    }
