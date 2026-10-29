"""Fase 2: rampa, bandas, áreas y mezcla con imágenes sintéticas."""

import numpy as np
import pytest

from deepshot.areas import Area, Scene, apply_areas, rotation_matrix
from deepshot.compose import blend, blend_pixels, depth_range_mask, load_mask, save_mask
from deepshot.tonemap import Band, ColorRamp, RampStop, apply_bands, base_image, ramp_preset


def gradient(h=10, w=101):
    """Profundidad que va de 0 (izquierda, cerca) a 1 (derecha, lejos)."""
    return np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))


# --- Rampa ---------------------------------------------------------------------
def test_default_ramp_near_white_far_black():
    out = ColorRamp().apply(gradient())
    assert out.shape == (10, 101, 3)
    assert np.allclose(out[0, 0], 1.0) and np.allclose(out[0, -1], 0.0)
    assert np.allclose(out[0, 50], 0.5, atol=1e-6)


def test_ramp_interpolations():
    ramp = ColorRamp([RampStop(0.0, (0, 0, 0)), RampStop(1.0, (1, 1, 1))])
    d = np.array([[0.25]], dtype=np.float32)
    assert np.isclose(ramp.apply(d)[0, 0, 0], 0.25)
    ramp.interpolation = "constant"
    assert ramp.apply(d)[0, 0, 0] == 0.0
    ramp.interpolation = "ease"
    assert np.isclose(ramp.apply(d)[0, 0, 0], 0.25 * 0.25 * (3 - 0.5))


def test_ramp_unsorted_stops_and_outside_values():
    ramp = ColorRamp([RampStop(0.8, (0, 0, 1)), RampStop(0.2, (1, 0, 0))])
    out = ramp.apply(np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.allclose(out[0, 0], (1, 0, 0))   # antes de la primera parada: su color
    assert np.allclose(out[0, 1], (0, 0, 1))   # después de la última: su color


def test_fit_ramp_to_relief_uses_real_range():
    depth = 0.3 + 0.2 * gradient()            # profundidades reales entre 0.3 y 0.5
    ramp = ColorRamp()
    ramp.fit_to_relief(depth, 0, 100)
    out = ramp.apply(depth)
    assert np.isclose(ramp.range_min, 0.3) and np.isclose(ramp.range_max, 0.5)
    assert np.allclose(out[0, 0], 1.0) and np.allclose(out[0, -1], 0.0)   # usa toda la rampa


def test_presets_exist_and_color_is_multicolor():
    for name in ("haze", "contrast", "color", "linear"):
        assert ramp_preset(name).apply(gradient()).shape == (10, 101, 3)
    c = ramp_preset("color").apply(gradient())
    assert c[0, 0, 0] > c[0, 0, 2] and c[0, -1, 2] > c[0, -1, 0]   # cerca rojo, lejos azul
    with pytest.raises(KeyError):
        ramp_preset("no-existe")


# --- Bandas ----------------------------------------------------------------------
def test_band_at_half_on_black_whitens_only_that_strip():
    """ROADMAP: una banda en 0.5 sobre negro deja blanco solo esa franja."""
    depth = gradient()
    ramp = ColorRamp()
    img = base_image("black", ramp, depth)
    out = apply_bands(img, [Band(position=0.5, width=0.2, softness=0.0)], ramp, depth)
    row = out[0, :, 0]
    assert np.all(row[41:60] == 1.0)              # dentro de 0.4..0.6: blanco (bordes exactos: redondeo float32)
    assert np.all(row[:39] == 0.0) and np.all(row[62:] == 0.0)


def test_band_softness_and_disabled():
    depth = gradient()
    ramp = ColorRamp()
    img = base_image("black", ramp, depth)
    soft = apply_bands(img, [Band(0.5, 0.0, 0.2)], ramp, depth)[0, :, 0]
    assert soft[50] == 1.0 and 0 < soft[40] < 1 and soft[20] == 0.0   # transición suave
    off = apply_bands(img, [Band(0.5, 0.2, 0.0, enabled=False)], ramp, depth)
    assert np.array_equal(off, img)


# --- Áreas ------------------------------------------------------------------------
def test_area_center_full_outside_zero():
    depth = np.full((51, 51), 0.5, dtype=np.float32)
    pts = Scene().points(depth)
    area = Area(center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3), edge_softness=0.0, falloff=0.0)
    m = area.mask(pts)
    assert m[25, 25] == 1.0 and m[0, 0] == 0.0


def test_area_selects_by_depth_not_only_position():
    """Elipsoide 3D: mismo píxel en pantalla, pero solo afecta a lo que está a su profundidad."""
    depth = np.full((21, 21), 0.9, dtype=np.float32)
    depth[:, :10] = 0.2                                   # mitad izquierda cerca
    pts = Scene().points(depth)
    area = Area(center=(0.0, 0.0, 0.2), scale=(5.0, 5.0, 0.1), edge_softness=0.0, falloff=0.0)
    m = area.mask(pts)
    assert np.all(m[:, :10] == 1.0) and np.all(m[:, 11:] == 0.0)


def test_sphere_is_rotation_invariant_and_box_rotates():
    pts = Scene().points(np.random.default_rng(0).random((20, 30), dtype=np.float32))
    sphere = Area(scale=(0.4, 0.4, 0.4))
    rotated = Area(scale=(0.4, 0.4, 0.4), rotation=(30, 45, 60))
    assert np.allclose(sphere.mask(pts), rotated.mask(pts), atol=1e-5)
    r = rotation_matrix(10, 20, 30)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)   # rotación pura


def test_area_strength_sign_lightens_or_darkens():
    depth = np.full((11, 11), 0.5, dtype=np.float32)
    pts = Scene().points(depth)
    gray = np.full((11, 11, 3), 0.5, dtype=np.float32)
    big = dict(center=(0, 0, 0.5), scale=(3, 3, 3), edge_softness=0, falloff=0)
    assert np.allclose(apply_areas(gray, [Area(strength=1.0, **big)], pts), 1.0)
    assert np.allclose(apply_areas(gray, [Area(strength=-1.0, **big)], pts), 0.0)
    assert np.allclose(apply_areas(gray, [Area(strength=-0.5, **big)], pts), 0.25)


def test_point_at_matches_points():
    depth = np.random.default_rng(1).random((30, 40), dtype=np.float32)
    for scene in (Scene(), Scene(relief=4.0, fov_deg=50)):
        p = scene.point_at(depth, 0.3, 0.6)
        row, col = int(0.6 * 30), int(0.3 * 40)
        assert np.allclose(p, scene.points(depth)[row, col], atol=1e-5)


# --- Mezcla y máscaras --------------------------------------------------------------
def test_empty_mask_changes_nothing():
    """ROADMAP: una máscara vacía no cambia nada."""
    rng = np.random.default_rng(2)
    base, top = rng.random((8, 8, 3), dtype=np.float32), rng.random((8, 8, 3), dtype=np.float32)
    out = blend(base, top, "multiply", 1.0, np.zeros((8, 8), dtype=np.float32))
    assert np.array_equal(out, base)


# --- Bandas con pan y tilt ------------------------------------------------------------------
def test_band_pan_and_tilt_tilt_the_slice_plane_around_the_image_centre():
    from deepshot.tonemap import tilted_depth

    flat = np.full((41, 81), 0.5, np.float32)                       # profundidad constante: todo a la misma distancia
    ramp = ColorRamp([RampStop(0.0, (0, 0, 0)), RampStop(1.0, (0, 0, 0))])
    black = np.zeros(flat.shape + (3,), np.float32)
    band = Band(position=0.5, width=0.06, softness=0.0, color=(1, 1, 1), strength=1.0)
    assert apply_bands(black, [band], ramp, flat)[..., 0].min() == 1.0      # sin pan ni tilt: cubre toda la imagen
    tilted = Band(position=0.5, width=0.06, softness=0.0, color=(1, 1, 1), strength=1.0, pan=30.0)
    out = apply_bands(black, [tilted], ramp, flat, relief=1.0)[..., 0]
    assert out[20, 40] == 1.0                                       # el centro de la imagen no cambia
    assert out[:, 0].max() == 0.0 and out[:, -1].max() == 0.0       # los lados quedan fuera: el plano se inclina
    assert out[0].tolist() == out[-1].tolist()                      # pan no depende de la altura
    t2 = Band(position=0.5, width=0.06, softness=0.0, color=(1, 1, 1), strength=1.0, tilt=30.0)
    out2 = apply_bands(black, [t2], ramp, flat, relief=1.0)[..., 0]
    assert out2[20, 0] == 1.0 and out2[0, 0] == 0.0 and out2[-1, 0] == 0.0     # tilt: franja horizontal en el centro
    assert out2[:, 0].tolist() == out2[:, -1].tolist()
    d = tilted_depth(flat, 30.0, 0.0, 1.0)
    assert d[20, -1] < d[20, 0] and d[20, 40] == pytest.approx(0.5, abs=0.01)
    # Más relieve = el mismo ángulo cambia menos la profundidad (en unidades de profundidad).
    assert abs(tilted_depth(flat, 30.0, 0.0, 4.0)[20, -1] - 0.5) < abs(tilted_depth(flat, 30.0, 0.0, 1.0)[20, -1] - 0.5)
    assert np.allclose(tilted_depth(flat, 0.0, 0.0, 1.0), flat)


def test_band_pan_and_tilt_round_trip_through_the_project_json_and_old_files_still_load():
    from deepshot.project import Project

    p = Project()
    p.bands = [Band(0.4, 0.1, 0.05, (1, 1, 1), 1.0, True, 25.0, -10.0)]
    again = Project.from_dict(p.to_dict())
    assert (again.bands[0].pan, again.bands[0].tilt) == (25.0, -10.0)
    old = p.to_dict()
    for b in old["bands"]:
        b.pop("pan"), b.pop("tilt")                                  # un archivo guardado antes de existir pan y tilt
    loaded = Project.from_dict(old)
    assert (loaded.bands[0].pan, loaded.bands[0].tilt) == (0.0, 0.0)


@pytest.mark.parametrize("mode,b,s,expected", [
    ("multiply", 0.5, 0.5, 0.25),
    ("screen", 0.5, 0.5, 0.75),
    ("overlay", 0.25, 0.5, 0.25),
    ("soft_light", 0.5, 0.5, 0.5),
    ("add", 0.7, 0.6, 1.0),
    ("darken", 0.3, 0.6, 0.3),
    ("lighten", 0.3, 0.6, 0.6),
    ("difference", 0.3, 0.8, 0.5),
])
def test_blend_modes_known_values(mode, b, s, expected):
    base = np.full((1, 1, 3), b, dtype=np.float32)
    top = np.full((1, 1, 3), s, dtype=np.float32)
    assert np.allclose(blend_pixels(base, top, mode), expected, atol=1e-6)


def test_opacity_is_linear():
    base = np.zeros((1, 1, 3), dtype=np.float32)
    top = np.ones((1, 1, 3), dtype=np.float32)
    assert np.allclose(blend(base, top, "normal", 0.25), 0.25)


def test_depth_range_mask_and_painted_mask_roundtrip(tmp_path):
    m = depth_range_mask(gradient(), 0.2, 0.4, softness=0.0)[0]
    assert m[30] == 1.0 and m[10] == 0.0 and m[50] == 0.0
    painted = np.linspace(0, 1, 50, dtype=np.float32).reshape(5, 10)
    save_mask(painted, tmp_path / "máscara pintada.png")
    back = load_mask(tmp_path / "máscara pintada.png")
    assert np.abs(back - painted).max() < 1e-4
    assert load_mask(tmp_path / "máscara pintada.png", (20, 10)).shape == (10, 20)


def test_band_color_from_ramp_uses_the_ramp_color_at_its_position():
    import numpy as np

    from deepshot import tonemap
    from deepshot.tonemap import Band, ColorRamp, RampStop

    ramp = ColorRamp([RampStop(0.0, (1.0, 0.0, 0.0)), RampStop(1.0, (0.0, 0.0, 1.0))])
    assert tonemap.ramp_color_at(ramp, 0.0) == pytest.approx((1.0, 0.0, 0.0), abs=1e-3)
    assert tonemap.ramp_color_at(ramp, 0.5) == pytest.approx((0.5, 0.0, 0.5), abs=1e-2)
    band = Band(position=0.5, width=1.0, softness=0.0, color=(0.0, 1.0, 0.0), strength=1.0)
    depth = np.full((4, 4), 0.5, dtype=np.float32)
    base = np.zeros((4, 4, 3), dtype=np.float32)
    own = tonemap.apply_bands(base, [band], ramp, depth)
    assert own[0, 0] == pytest.approx((0.0, 1.0, 0.0), abs=1e-3)                  # su propio color
    band.color_from_ramp = True
    from_ramp = tonemap.apply_bands(base, [band], ramp, depth)
    assert from_ramp[0, 0] == pytest.approx((0.5, 0.0, 0.5), abs=1e-2)             # el de la rampa en 0,5
    from deepshot.project import Project

    p = Project(bands=[band])
    assert Project.from_dict(p.to_dict()).bands[0].color_from_ramp is True        # se guarda y se carga
    old = p.to_dict()
    del old["bands"][0]["color_from_ramp"]
    assert Project.from_dict(old).bands[0].color_from_ramp is False               # los archivos antiguos siguen valiendo


def test_refine_mask_feather_softens_and_snap_sticks_the_edge_to_the_photo_edge():
    import numpy as np

    from deepshot.compose import refine_mask

    h, w = 60, 120
    photo = np.zeros((h, w, 3), dtype=np.float32)
    photo[:, 70:] = 1.0                                                    # el borde real de la foto está en la columna 70
    hard = np.zeros((h, w), dtype=np.float32)
    hard[:, 66:] = 1.0                                                     # un pincel poco preciso: el borde queda en la 66
    assert np.array_equal(refine_mask(hard, photo, 0.0, 0.0), hard)        # sin retoque, igual
    soft = refine_mask(hard, None, feather=0.5)
    assert 0.0 < soft[30, 60] < 1.0 and soft[30, 66] < 1.0 and soft.min() >= 0.0 and soft.max() <= 1.0
    assert abs(soft.sum() - hard.sum()) / hard.sum() < 0.05               # difuminar no cambia lo que cubre
    snapped = refine_mask(hard, photo, snap=1.0)
    assert snapped[30, 67] < hard[30, 67] - 0.2 and snapped[30, 76] > 0.9   # el borde se corre hacia el de la foto (70)
    assert refine_mask(hard, None, snap=1.0).tolist() == hard.tolist()     # sin foto guía, no hay ajuste


def test_painted_mask_retouch_is_used_by_render_and_saved_in_the_project():
    import numpy as np

    from deepshot.project import Composite, Project, render

    depth = np.tile(np.linspace(0, 1, 40, dtype=np.float32), (30, 1))
    photo = np.zeros((30, 40, 3), dtype=np.float32)
    mask = np.zeros((30, 40), dtype=np.float32)
    mask[:, 20:] = 1.0
    base = Project(composite=Composite(mode="normal", opacity=1.0, mask="painted"))
    hard = render(base, depth, photo, painted_mask=mask)
    p = Project(composite=Composite(mode="normal", opacity=1.0, mask="painted", mask_feather=0.6))
    soft = render(p, depth, photo, painted_mask=mask)
    assert not np.array_equal(hard, soft)                                  # el suavizado cambia el resultado
    assert Project.from_dict(p.to_dict()).composite.mask_feather == 0.6   # y se guarda
    old = p.to_dict()
    del old["composite"]["mask_feather"], old["composite"]["mask_snap"]
    assert Project.from_dict(old).composite.mask_snap == 0.0              # los archivos antiguos valen
