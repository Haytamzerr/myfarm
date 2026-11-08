import numpy as np
import pytest

from myfarm.parcel import Parcel, load_kml

KML = "data/firmatech.kml"


def square(side_m: float) -> Parcel:
    """A square parcel of known size near the farm, built in metres then turned into lon/lat."""
    ref = Parcel("ref", [(-4.89, 33.91), (-4.889, 33.91), (-4.889, 33.911), (-4.89, 33.91)])
    h = side_m / 2
    lon, lat = ref.to_degrees(np.array([-h, h, h, -h, -h]), np.array([-h, -h, h, h, -h]))
    return Parcel("square", list(zip(lon.tolist(), lat.tolist())))


def test_kml_is_read_as_lon_lat():
    p = load_kml(KML)
    assert p.name == "firmatech"
    lon, lat = p.centroid
    assert -5 < lon < -4.8 and 33.9 < lat < 34  # near Fès-Meknès, not swapped
    assert p.coords[0] == p.coords[-1]


def test_area_of_a_known_square():
    assert square(100).area_ha == pytest.approx(1.0, rel=0.01)


def test_farm_area_is_plausible():
    assert 1.5 < load_kml(KML).area_ha < 2.2


def test_grid_has_one_point_per_10m_pixel_inside():
    p = square(100)
    points = p.grid(10)
    assert len(points) == 100
    from shapely.geometry import Point
    assert all(p.polygon.contains(Point(x, y)) for x, y in points)


def test_bad_kml_is_rejected(tmp_path):
    bad = tmp_path / "bad.kml"
    bad.write_text('<kml xmlns="http://www.opengis.net/kml/2.2"><Document/></kml>')
    with pytest.raises(ValueError):
        load_kml(bad)
