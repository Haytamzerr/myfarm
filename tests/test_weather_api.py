import pytest
from fastapi.testclient import TestClient

from myfarm.api import app
from myfarm.weather import agro_summary, irrigation_need, load_open_meteo

WEATHER = "data/weather_open-meteo.csv"
client = TestClient(app)


def test_open_meteo_blocks_are_split():
    location, hourly, daily = load_open_meteo(WEATHER)
    assert location["latitude"] == pytest.approx(33.9375)
    assert len(hourly) == 24 * len(daily)
    assert "et0_fao_evapotranspiration" in daily


def test_growing_degree_days():
    _, _, daily = load_open_meteo(WEATHER)
    s = agro_summary(daily, base_temp=10)
    first = daily.iloc[0]
    expected = (first.temperature_2m_max + first.temperature_2m_min) / 2 - 10
    assert s.loc[0, "gdd"] == pytest.approx(expected, abs=0.01)
    assert (s["gdd"] >= 0).all()


def test_irrigation_need_in_cubic_metres():
    _, _, daily = load_open_meteo(WEATHER)
    totals = irrigation_need(agro_summary(daily, kc=1.0), area_ha=2.0)
    assert totals["deficit_m3"] == pytest.approx(totals["deficit_mm"] * 10 * 2.0, rel=0.01)


def test_api_parcel_and_page():
    assert client.get("/").status_code == 200
    p = client.get("/api/parcel").json()
    assert p["geometry"]["type"] == "Polygon" and p["pixels"] > 100


def test_api_demo_scan():
    data = client.get("/api/scan", params={"source": "demo", "k": 4}).json()
    assert data["meta"]["simulated"] is True
    assert data["summary"]["k"] == 4
    assert data["pixels"][0]["lon"] < 0


def test_api_rejects_unknown_source():
    assert client.get("/api/scan", params={"source": "drone"}).status_code == 422


def test_api_weather_responds_to_kc():
    low = client.get("/api/weather", params={"kc": 0.5}).json()["totals"]["deficit_mm"]
    high = client.get("/api/weather", params={"kc": 1.2}).json()["totals"]["deficit_mm"]
    assert high > low
