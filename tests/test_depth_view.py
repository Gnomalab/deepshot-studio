"""Mapa de profundidad en color y con zonas marcadas (sin cambiar el mapa)."""

import numpy as np

from deepshot import depth_view as dv
from deepshot.depth_control import DepthControl


def ramp_depth():
    return np.tile(np.linspace(0, 1, 101, dtype=np.float32), (4, 1))


def test_colorize_maps_near_to_the_first_color_and_far_to_the_last_for_every_palette():
    depth = ramp_depth()
    original = depth.copy()
    for name, ramp in dv.PALETTES.items():
        img = dv.colorize(depth, name)
        assert img.shape == (4, 101, 3) and img.dtype == np.uint8
        first, last = ramp.sorted_stops()[0].color, ramp.sorted_stops()[-1].color
        assert np.allclose(img[0, 0] / 255.0, first, atol=0.01) and np.allclose(img[0, -1] / 255.0, last, atol=0.01)
    assert np.array_equal(depth, original)                              # no toca el mapa
    assert np.array_equal(dv.colorize(depth, "no existe"), dv.colorize(depth, "red_blue"))   # paleta desconocida: la de fábrica


def test_zones_image_is_plain_gray_without_zones_and_tints_each_zone_with_its_own_color():
    depth = ramp_depth()
    control = DepthControl(enabled=True)
    img = dv.zones_image(depth, control)
    assert img.shape == (4, 101, 3)
    near, mid, far = img[0, 5], img[0, 50], img[0, 95]
    assert near[2] > near[0] and mid[1] > mid[0] and far[0] > far[2]    # azul · verde · naranja
    none = DepthControl(enabled=True)
    for z in none.zones():
        z.enabled = False
    expect = np.round((1.0 - depth) * 255.0).astype(np.uint8)
    assert np.array_equal(dv.zones_image(depth, none)[..., 0], expect)  # sin zonas activas: el mapa en gris


def test_zones_image_includes_extra_zones_and_overlapping_manual_zones_stay_in_range():
    depth = ramp_depth()
    control = DepthControl(enabled=True)
    control.add_zone()
    control.near.auto_range = control.mid.auto_range = False
    control.near.range_min, control.near.range_max = 0.0, 0.8             # solapadas a propósito
    control.mid.range_min, control.mid.range_max = 0.2, 1.0
    img = dv.zones_image(depth, control)
    assert img.dtype == np.uint8 and img.shape == (4, 101, 3)
    weights = dict(control.zone_weights(depth))
    assert "extra0" in weights and set(weights) <= set(control.zone_names())
    assert all(w.shape == depth.shape and 0 <= w.min() and w.max() <= 1.0 for w in weights.values())
