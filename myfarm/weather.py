"""Weather and water balance for the parcel, from an Open-Meteo CSV export.

Open-Meteo (open-meteo.com) exports one file with a header line, then an hourly table, then
a daily table, separated by blank lines. From the daily table we compute what a grower acts on:

- growing degree days (GDD): heat available to the crop, max(0, (Tmax + Tmin) / 2 - base);
- heat-stress days: Tmax at or above 35 °C;
- crop water need: ET0 x Kc (FAO-56 reference evapotranspiration x crop coefficient);
- water balance: rain - crop water need, i.e. what irrigation must make up.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

HEAT_STRESS_C = 35.0


def _clean(column: str) -> str:
    """'temperature_2m_max (°C)' -> 'temperature_2m_max'"""
    return re.sub(r"\s*\(.*\)$", "", column).strip()


def load_open_meteo(path: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """(location, hourly table, daily table)."""
    blocks = [b for b in Path(path).read_text(encoding="utf-8").split("\n\n") if b.strip()]
    if len(blocks) < 2:
        raise ValueError("not an Open-Meteo CSV export")
    location = pd.read_csv(io.StringIO(blocks[0])).iloc[0].to_dict()
    tables = []
    for block in blocks[1:]:
        table = pd.read_csv(io.StringIO(block))
        table.columns = [_clean(c) for c in table.columns]
        table["time"] = pd.to_datetime(table["time"])
        tables.append(table)
    hourly = next((t for t in tables if t["time"].dt.hour.nunique() > 1), pd.DataFrame())
    daily = next((t for t in tables if t["time"].dt.hour.nunique() == 1), pd.DataFrame())
    if daily.empty and not hourly.empty:  # only hourly data: build the daily table ourselves
        daily = _daily_from_hourly(hourly)
    return location, hourly, daily


def _daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    g = hourly.set_index("time").resample("D")
    daily = pd.DataFrame({
        "temperature_2m_max": g["temperature_2m"].max(),
        "temperature_2m_min": g["temperature_2m"].min(),
    })
    if "precipitation" in hourly:
        daily["precipitation_sum"] = g["precipitation"].sum()
    if "et0_fao_evapotranspiration" in hourly:
        daily["et0_fao_evapotranspiration"] = g["et0_fao_evapotranspiration"].sum()
    return daily.reset_index()


def agro_summary(daily: pd.DataFrame, base_temp: float = 10.0, kc: float = 1.0) -> pd.DataFrame:
    """Daily agronomic indicators. `kc` is the crop coefficient (FAO-56: ~0.4 young crop,
    ~1.0-1.2 full canopy); `base_temp` the crop's base temperature (10 °C for maize, olive…)."""
    out = pd.DataFrame({"date": daily["time"].dt.date})
    tmax, tmin = daily["temperature_2m_max"], daily["temperature_2m_min"]
    out["tmax"], out["tmin"] = tmax, tmin
    out["gdd"] = ((tmax + tmin) / 2 - base_temp).clip(lower=0)
    out["gdd_cumulative"] = out["gdd"].cumsum()
    out["heat_stress"] = tmax >= HEAT_STRESS_C
    rain = daily.get("precipitation_sum", daily.get("rain_sum", pd.Series(0.0, index=daily.index)))
    out["rain_mm"] = rain.fillna(0)
    if "et0_fao_evapotranspiration" in daily:
        out["et0_mm"] = daily["et0_fao_evapotranspiration"]
        out["crop_need_mm"] = out["et0_mm"] * kc
        out["balance_mm"] = out["rain_mm"] - out["crop_need_mm"]
        out["balance_cumulative_mm"] = out["balance_mm"].cumsum()
    return out.round(2)


def irrigation_need(summary: pd.DataFrame, area_ha: float) -> dict:
    """Total water deficit over the period, in mm and in cubic metres for the parcel
    (1 mm over 1 ha = 10 m³)."""
    if "balance_mm" not in summary:
        return {}
    deficit_mm = float(-summary["balance_mm"].clip(upper=0).sum())
    return {
        "days": len(summary),
        "deficit_mm": round(deficit_mm, 1),
        "deficit_m3": round(deficit_mm * 10 * area_ha, 1),
        "heat_stress_days": int(summary["heat_stress"].sum()),
        "gdd_total": round(float(summary["gdd"].sum()), 1),
    }
