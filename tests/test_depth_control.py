"""Control de profundidad por zonas (cerca / medio / lejos): matemática, JSON y efecto en el render."""

import json

import numpy as np
import pytest

from deepshot.depth_control import DepthControl, DepthZone
from deepshot.project import Project, render

D = np.linspace(0.0, 1.0, 1001, dtype=np.float32)       # eje de profundidad muestreado: índice = valor × 1000


def control(**zones):
    """DepthControl activo con los cambios indicados por zona, p. ej. control(mid={'target': 0.6})."""
    c = DepthControl(enabled=True)
    for name, changes in zones.items():
        for k, v in changes.items():
            setattr(getattr(c, name), k, v)
    return c


def test_off_or_without_shifts_returns_the_same_array():
    assert DepthControl().apply(D) is D                                           # apagado
    assert control().apply(D) is D                                                # activo pero destino = centro
    assert control(mid={"enabled": False, "target": 0.9}).is_identity()           # la única zona movida está apagada


def test_a_zone_moves_its_centre_to_the_target_and_leaves_far_zones_alone():
    c = control(mid={"center": 0.5, "target": 0.6, "feather": 0.04})
    out = c.apply(D)
    assert out[500] == pytest.approx(0.6, abs=1e-5)                               # el centro llega al destino
    assert out[150] == pytest.approx(0.15, abs=1e-5)                              # la zona cercana no se toca
    assert out[850] == pytest.approx(0.85, abs=1e-5)                              # ni la lejana
    assert out[300] == pytest.approx(0.35, abs=1e-4)                              # en el límite (0,3) mezcla a medias


def test_automatic_limits_split_the_axis_halfway_between_centres():
    c = control(near={"center": 0.1}, mid={"center": 0.5}, far={"center": 0.9})
    (n, n_lo, n_hi), (m, m_lo, m_hi), (f, f_lo, f_hi) = c.ranges()
    assert (n_lo, n_hi) == (0.0, pytest.approx(0.3))
    assert (m_lo, m_hi) == (pytest.approx(0.3), pytest.approx(0.7))
    assert (f_lo, f_hi) == (pytest.approx(0.7), 1.0)


def test_moving_every_zone_by_the_same_amount_shifts_the_whole_map():
    """Con reparto automático los pesos suman 1: tres zonas con el mismo desplazamiento = un desplazamiento global."""
    c = control(near={"center": 0.1, "target": 0.15}, mid={"center": 0.5, "target": 0.55},
                far={"center": 0.9, "target": 0.95})
    out = c.apply(D)
    inner = slice(0, 950)                                                         # el borde alto se recorta a 1
    assert np.allclose(out[inner], D[inner] + 0.05, atol=1e-5)


def test_feather_makes_the_transition_gradual_and_never_a_step():
    sharp = control(mid={"target": 0.6, "feather": 0.0})
    soft = control(mid={"target": 0.6, "feather": 0.2})
    jump_sharp = np.abs(np.diff(sharp.apply(D))).max()
    jump_soft = np.abs(np.diff(soft.apply(D))).max()
    assert jump_soft < jump_sharp / 5                                             # con suavizado el salto desaparece
    assert np.abs(np.diff(soft.apply(D))).max() < 0.004


def test_manual_range_moves_only_that_range():
    c = control(mid={"center": 0.5, "target": 0.6, "auto_range": False, "range_min": 0.4, "range_max": 0.6,
                     "feather": 0.02})
    out = c.apply(D)
    assert out[500] == pytest.approx(0.6, abs=1e-5)
    assert out[300] == pytest.approx(0.3, abs=1e-5) and out[800] == pytest.approx(0.8, abs=1e-5)
    # Un límite manual al revés se interpreta igual (desde > hasta).
    swapped = control(mid={"center": 0.5, "target": 0.6, "auto_range": False, "range_min": 0.6, "range_max": 0.4,
                           "feather": 0.02})
    assert np.allclose(swapped.apply(D), out)


def test_result_is_always_inside_0_1_and_input_is_not_modified():
    src = D.copy()
    c = control(far={"center": 0.9, "target": 1.0}, near={"center": 0.1, "target": 0.0},
                mid={"center": 0.5, "target": 0.0})
    out = c.apply(D)
    assert out.min() >= 0.0 and out.max() <= 1.0 and out.dtype == np.float32
    assert np.array_equal(D, src)


def test_centers_from_depth_use_percentiles_and_keep_the_image_unchanged():
    depth = np.linspace(0.2, 0.8, 1000, dtype=np.float32)
    c = DepthControl(enabled=True)
    c.set_centers_from_depth(depth)
    assert c.near.center == pytest.approx(0.26, abs=0.01) and c.mid.center == pytest.approx(0.5, abs=0.01)
    assert c.far.center == pytest.approx(0.74, abs=0.01)
    assert all(z.target == z.center for z in c.zones()) and c.is_identity()


def test_json_round_trip_and_tolerance_to_missing_or_unknown_keys():
    p = Project(depth_control=control(mid={"target": 0.7, "feather": 0.1}, far={"auto_range": False}))
    restored = Project.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored.depth_control == p.depth_control
    # Un proyecto antiguo no tiene la clave; uno raro trae claves de más y zonas incompletas.
    assert Project.from_dict({}).depth_control == DepthControl()
    odd = DepthControl.from_dict({"enabled": True, "mid": {"target": 0.7, "inventado": 1}, "extra": 3})
    assert odd.enabled and odd.mid.target == 0.7 and odd.near == DepthZone(center=0.1, target=0.1)


def test_key_changes_with_any_setting():
    a, b = control(), control()
    assert a.key() == b.key()
    b.mid.target = 0.6
    assert a.key() != b.key()


# --- En el render --------------------------------------------------------------------------------------------
def gradient_depth(w=64, h=32):
    return np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))


def test_render_without_the_control_is_unchanged_and_with_it_the_image_changes():
    depth = gradient_depth()
    base = render(Project(), depth)
    off = Project(depth_control=control())                                         # activo pero sin mover nada
    assert np.array_equal(render(off, depth), base)
    moved = Project(depth_control=control(mid={"target": 0.7, "feather": 0.05}))
    out = render(moved, depth)
    assert not np.allclose(out, base)
    assert np.array_equal(out[:, 0], base[:, 0])                                  # lo más cercano (columna 0) no cambia


def test_render_areas_follow_the_moved_depth():
    """Las áreas 3D se colocan sobre la profundidad ya movida: mover una zona cambia dónde cae el área."""
    from deepshot.areas import Area

    depth = gradient_depth()
    area = Area(center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.1), strength=1.0)
    plain = render(Project(areas=[area]), depth)
    moved = render(Project(areas=[area], depth_control=control(mid={"target": 0.8, "feather": 0.05})), depth)
    assert not np.allclose(plain, moved)


# --- Zonas extra (más de tres) ----------------------------------------------------------------------------
def test_extra_zones_are_added_in_the_widest_gap_removed_and_limited():
    from deepshot.depth_control import MAX_EXTRA

    c = DepthControl()
    first = c.add_zone()
    assert first == "extra0" and c.zone("extra0").center == pytest.approx(0.7, abs=0.01)      # entre 0,5 y 0,9
    assert c.zone("extra0").center == c.zone("extra0").target                                  # no cambia la imagen
    names = [first] + [c.add_zone() for _ in range(MAX_EXTRA - 1)]
    assert names == [f"extra{i}" for i in range(MAX_EXTRA)] and c.add_zone() is None           # tope
    assert c.zone_names() == ["near", "mid", "far", *names] and len(c.zones()) == 3 + MAX_EXTRA
    assert not c.remove_zone("mid") and not c.remove_zone("extra9") and not c.remove_zone("nope")
    assert c.remove_zone("extra0") and len(c.extra) == MAX_EXTRA - 1
    with pytest.raises(KeyError):
        c.zone("extra7")


def test_extra_zone_moves_its_own_range_and_everything_still_sums_to_the_original_when_untouched():
    depth = np.tile(np.linspace(0, 1, 201, dtype=np.float32), (3, 1))
    c = DepthControl(enabled=True)
    c.add_zone()                                                  # centro ≈ 0,7
    assert np.allclose(c.apply(depth), depth)                     # sin mover destinos no cambia nada
    c.zone("extra0").target = c.zone("extra0").center + 0.1       # solo esa zona se aleja
    out = c.apply(depth)[0]
    base = depth[0]
    mid_extra = (base > 0.66) & (base < 0.74)                     # el centro de la zona nueva
    assert np.allclose(out[mid_extra] - base[mid_extra], 0.1, atol=0.03)
    assert np.allclose(out[base < 0.2], base[base < 0.2], atol=1e-3)                          # lo cercano no se mueve
    # Las zonas se reparten el eje por el orden de sus centros, no por el orden en que se crearon.
    lows = [lo for _z, lo, _hi in c.ranges()]
    assert lows == sorted(lows) and lows[0] == 0.0


def test_extra_zones_survive_json_old_projects_and_key_changes():
    c = DepthControl(enabled=True)
    c.add_zone()
    c.zone("extra0").target = 0.8
    back = DepthControl.from_dict(json.loads(json.dumps(Project(depth_control=c).to_dict()["depth_control"])))
    assert back.extra == c.extra and back.key() == c.key()
    old = DepthControl.from_dict({"enabled": True, "mid": {"target": 0.6}})                  # proyecto antiguo, sin «extra»
    assert old.extra == [] and old.mid.target == 0.6
    junk = DepthControl.from_dict({"extra": [{"center": 0.4, "bogus": 1}, "no soy una zona"] * 5})
    assert len(junk.extra) == 2 and all(z.center == 0.4 for z in junk.extra)                  # se miran solo las 3 primeras y la basura se ignora
    before = c.key()
    c.zone("extra0").center = 0.65
    assert c.key() != before
