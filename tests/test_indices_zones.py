import numpy as np
import pandas as pd
import pytest

from myfarm.indices import add_indices, evi, ndmi, ndvi
from myfarm.parcel import load_kml
from myfarm.synthetic import simulate_scene
from myfarm.zones import best_k, leakage_demo, management_zones


@pytest.fixture(scope="module")
def scene():
    parcel = load_kml("data/firmatech.kml")
    return add_indices(simulate_scene(parcel, parcel.grid(10)))


def test_ndvi_known_values():
    assert ndvi(0.5, 0.1) == pytest.approx(0.6667, abs=1e-4)  # healthy vegetation
    assert ndvi(0.2, 0.2) == 0  # bare soil-like
    assert np.isnan(ndvi(0.0, 0.0))  # no division by zero


def test_ndmi_and_evi():
    assert ndmi(0.4, 0.2) == pytest.approx(1 / 3)
    assert evi(0.5, 0.1, 0.05) == pytest.approx(2.5 * 0.4 / (0.5 + 0.6 - 0.375 + 1))


def test_missing_band_is_reported():
    with pytest.raises(ValueError, match="missing bands"):
        add_indices(pd.DataFrame({"nir": [0.5], "red": [0.1]}))


def test_simulated_scene_keeps_coordinates_and_plausible_values(scene):
    assert (scene["lon"] < 0).all()  # west of Greenwich: coordinates must not be clipped
    assert scene["NDVI"].between(-1, 1).all()
    assert 0.3 < scene["NDVI"].mean() < 0.9


def test_zones_are_ordered_by_vigour(scene):
    zoned, summary = management_zones(scene, k=3)
    means = [z["mean_NDVI"] for z in summary["zones"]]
    assert means == sorted(means)
    assert [z["name"] for z in summary["zones"]] == ["low", "medium", "high"]
    assert sum(z["share_pct"] for z in summary["zones"]) == pytest.approx(100, abs=0.2)
    assert set(zoned["zone"]) == {0, 1, 2}


def test_zones_are_reproducible(scene):
    assert management_zones(scene, 3, seed=1)[1] == management_zones(scene, 3, seed=1)[1]


def test_best_k_is_a_candidate(scene):
    assert best_k(scene) in (2, 3, 4)


def test_cloudy_pixels_are_ignored(scene):
    cloudy = scene.copy()
    cloudy.loc[:19, ["NDVI", "NDRE", "NDMI"]] = np.nan
    zoned, _ = management_zones(cloudy, 3)
    assert len(zoned) == len(scene) - 20


def test_leakage_demo_scores_perfectly_on_pure_noise():
    """The trap of the first version: on random numbers, 'predicting' NDVI > 0.4 from NDVI
    still scores ~100 %, so such a score says nothing about the field."""
    noise = pd.DataFrame({"NDVI": np.random.default_rng(0).uniform(-1, 1, 300)})
    assert leakage_demo(noise) > 0.97
