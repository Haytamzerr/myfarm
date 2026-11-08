"""Sentinel-2 reading logic, without the network: the catalogue item and the pixel reads are
faked, so only our own code (scaling, offset, cloud masking) is tested."""

import sys
import types

import numpy as np
import pytest

from myfarm import sentinel


class FakeAsset:
    def __init__(self, href, offset=-0.1):
        self.href = href
        self.extra_fields = {"raster:bands": [{"scale": 0.0001, "offset": offset}]}


class FakeItem:
    def __init__(self):
        self.assets = {name: FakeAsset(name) for name in sentinel.ASSETS}
        self.assets["scl"] = FakeAsset("scl")


@pytest.fixture
def fake_reads(monkeypatch):
    values = {"blue": 1500, "green": 1800, "red": 1600, "rededge1": 2900, "nir": 5000, "swir16": 3000}
    scl = np.array([4, 9, 4])  # vegetation, cloud, vegetation

    def sample(href, points):
        return scl if href == "scl" else np.array([values[href], values[href], 0.0])

    monkeypatch.setattr(sentinel, "_sample", sample)
    # rasterio.Env is only used as a context manager: a stand-in keeps the test light
    fake_rasterio = types.SimpleNamespace(Env=lambda **_: __import__("contextlib").nullcontext())
    monkeypatch.setitem(sys.modules, "rasterio", fake_rasterio)


def test_scaling_offset_and_cloud_mask(fake_reads):
    points = np.array([[-4.892, 33.915], [-4.892, 33.916], [-4.892, 33.917]])
    table = sentinel.read_scene(FakeItem(), points)
    assert table.loc[0, "nir"] == pytest.approx(5000 * 0.0001 - 0.1)  # 0.4 reflectance
    assert np.isnan(table.loc[1, "nir"])  # pixel under a cloud
    assert np.isnan(table.loc[2, "nir"])  # no data (0)
    assert list(table.columns[:2]) == ["lon", "lat"]


def test_old_scenes_without_offset():
    assert sentinel._scale_offset(FakeAsset("x", offset=0.0)) == (0.0001, 0.0)
