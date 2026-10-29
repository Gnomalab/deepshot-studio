"""Desenfoque de lente por profundidad: núcleo del diafragma, círculo de confusión, halos y proyecto."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from deepshot import dof  # noqa: E402
from deepshot import presets  # noqa: E402
from deepshot.dof import LensBlur, aperture_kernel, coc_map, lens_blur  # noqa: E402
from deepshot.project import Project, render  # noqa: E402
from deepshot.tonemap import ramp_preset  # noqa: E402


# --- Núcleo del diafragma ----------------------------------------------------------------------------
def test_kernel_is_normalized_and_round_by_default():
    k = aperture_kernel(10.0)
    assert k.sum() == pytest.approx(1.0, abs=1e-5) and k.shape == (23, 23)
    assert k[11, 11] > 0 and k[0, 0] == 0                                  # el centro cuenta, la esquina no
    assert np.allclose(k, k[::-1, ::-1], atol=1e-6) and np.allclose(k, k.T, atol=1e-6)     # simétrico
    area = (k > 0.5 * k.max()).sum()
    assert area == pytest.approx(np.pi * 10 ** 2, rel=0.08)                # área de un círculo de radio 10


def test_polygon_kernel_has_corners_and_rotates():
    r = 12.0
    hexa = aperture_kernel(r, blades=6)
    circle = aperture_kernel(r)
    assert hexa.sum() == pytest.approx(1.0, abs=1e-5)
    assert (hexa > 0).sum() < (circle > 0).sum()                           # un polígono inscrito cubre menos que el círculo
    turned = aperture_kernel(r, blades=6, rotation_deg=30)
    assert not np.allclose(hexa, turned)
    assert np.allclose(aperture_kernel(r, blades=6, rotation_deg=60), hexa, atol=1e-3)   # 60° = misma forma en un hexágono
    soft = aperture_kernel(r, softness=1.0)
    hard = aperture_kernel(r, softness=0.0)
    assert ((soft > 0) & (soft < soft.max() * 0.9)).sum() > ((hard > 0) & (hard < hard.max() * 0.9)).sum()   # borde más ancho


def test_tiny_radius_gives_identity_kernel():
    assert aperture_kernel(0.2).tolist() == [[1.0]]


# --- Círculo de confusión -----------------------------------------------------------------------------
def test_coc_is_zero_in_focus_and_grows_with_distance_and_aperture():
    depth = np.tile(np.linspace(0, 1, 101, dtype=np.float32), (4, 1))
    p = LensBlur(enabled=True, focus=0.4, sharp_range=0.1, aperture=2.8)
    c = coc_map(depth, p)[0]
    assert c[30:51].max() == 0.0                                           # dentro de la zona nítida (0.3 .. 0.5)
    assert c[60] < c[80] < c[100] and c[20] < c[10] < c[0]                 # crece al alejarse del foco, por ambos lados
    wide = coc_map(depth, LensBlur(focus=0.4, sharp_range=0.1, aperture=1.4))[0]
    narrow = coc_map(depth, LensBlur(focus=0.4, sharp_range=0.1, aperture=11.0))[0]
    assert wide[100] > c[100] > narrow[100]                                # más abierta = más desenfoque
    assert wide[100] == pytest.approx(2.0 * c[100], rel=1e-5)             # f/1.4 vs f/2.8: el doble
    no_near = coc_map(depth, LensBlur(focus=0.4, sharp_range=0.1, aperture=2.8, near=0.0))[0]
    assert no_near[:30].max() == 0.0 and no_near[100] == c[100]            # «delante» a 0: solo se desenfoca el fondo


def test_coc_is_relative_to_image_size():
    d_small = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (50, 1))
    d_big = np.tile(np.linspace(0, 1, 400, dtype=np.float32), (200, 1))
    p = LensBlur(focus=0.0, sharp_range=0.0, aperture=2.8)
    assert coc_map(d_big, p).max() == pytest.approx(4.0 * coc_map(d_small, p).max(), rel=1e-4)   # misma proporción


# --- Desenfoque ----------------------------------------------------------------------------------------
def checker_scene(h=240, w=320):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.repeat((((xx // 8 + yy // 8) % 2) * 0.7 + 0.1)[..., None], 3, axis=-1).astype(np.float32)
    depth = np.full((h, w), 0.9, dtype=np.float32)
    fg = (abs(xx - w // 2) < 50) & (abs(yy - h // 2) < 50)
    img[fg] = (1.0, 0.1, 0.1)
    depth[fg] = 0.3
    return img, depth, fg


def test_in_focus_object_stays_sharp_and_background_blurs_without_halo():
    img, depth, fg = checker_scene()
    out = lens_blur(img, depth, LensBlur(enabled=True, focus=0.3, sharp_range=0.05, aperture=2.0))
    assert np.abs(out[fg] - img[fg]).max() < 1e-4                          # el primer plano no cambia
    bg = ~fg
    assert out[bg].std() < img[bg].std() * 0.6                             # el fondo (damero) pierde contraste: desenfocado
    # Sin halo: junto al objeto, el fondo desenfocado no se tiñe de rojo (los píxeles nítidos no pesan en el desenfoque).
    h, w = depth.shape
    ring = (abs(np.mgrid[0:h, 0:w][1] - w // 2) < 62) & (abs(np.mgrid[0:h, 0:w][0] - h // 2) < 62) & bg
    assert (out[ring][:, 0] - out[ring][:, 1]).max() < 0.25                # el fondo es gris: sin tinte rojo notable


def test_focus_far_blurs_the_foreground_instead():
    img, depth, fg = checker_scene()
    out = lens_blur(img, depth, LensBlur(enabled=True, focus=0.9, sharp_range=0.05, aperture=2.0))
    yy, xx = np.mgrid[0:240, 0:320]
    far = ~fg & ~((abs(xx - 160) < 75) & (abs(yy - 120) < 75))              # fondo lejos del cuadrado: nítido (en foco)
    assert np.abs(out[far] - img[far]).max() < 1e-3
    # El objeto cercano desenfocado tiene el borde suave y translúcido: deja ver el fondo por dentro y se extiende por fuera.
    inside_edge = fg & ((abs(xx - 160) > 42) | (abs(yy - 120) > 42))
    outside_edge = ~fg & (abs(xx - 160) < 60) & (abs(yy - 120) < 60)
    assert np.abs(out[inside_edge] - img[inside_edge]).max() > 0.1          # el borde interior deja de ser rojo puro
    assert (out[outside_edge][:, 0] - out[outside_edge][:, 1]).max() > 0.15  # y el rojo se extiende sobre el fondo nítido
    assert np.abs(out[fg & (abs(xx - 160) < 20) & (abs(yy - 120) < 20)] - img[fg & (abs(xx - 160) < 20) & (abs(yy - 120) < 20)]).max() < 0.02   # el centro sigue opaco


def test_nothing_changes_when_everything_is_in_focus_or_aperture_is_tiny():
    img, depth, _ = checker_scene()
    flat = np.full_like(depth, 0.5)
    assert np.array_equal(lens_blur(img, flat, LensBlur(focus=0.5, sharp_range=0.1)), img)
    closed = LensBlur(enabled=True, focus=0.3, aperture=22.0, sharp_range=0.05)
    small = np.ascontiguousarray(img[:40, :60]), np.ascontiguousarray(depth[:40, :60])
    assert np.abs(lens_blur(small[0], small[1], closed) - small[0]).max() < 0.2     # f/22 en una imagen pequeña: casi nada


def test_bokeh_discs_are_brighter_with_highlights_and_polygonal_with_blades():
    h, w = 200, 200
    img = np.full((h, w, 3), 0.05, dtype=np.float32)
    img[100, 100] = 1.0                                                     # un punto de luz muy pequeño
    depth = np.full((h, w), 0.9, dtype=np.float32)
    base = LensBlur(enabled=True, focus=0.1, sharp_range=0.0, aperture=1.4, highlights=0.0)
    lo = lens_blur(img, depth, base)
    hi = lens_blur(img, depth, LensBlur(**{**base.__dict__, "highlights": 1.0}))
    assert hi[100, 100, 0] > lo[100, 100, 0] * 2                            # con altas luces el disco brilla mucho más
    hexa = lens_blur(img, depth, LensBlur(**{**base.__dict__, "highlights": 1.0, "blades": 6}))
    ys, xs = np.nonzero(hexa[..., 0] > 0.5 * hexa[..., 0].max())
    assert 20 < len(ys) < (np.pi * dof.max_radius(base, 200) ** 2) * 1.05   # disco claro de tamaño razonable
    assert not np.allclose(hexa, hi, atol=1e-3)                             # el hexágono no es el disco redondo


def test_large_images_are_computed_small_and_scaled_back():
    img, depth, _ = checker_scene(h=300, w=420)
    big = np.kron(img, np.ones((6, 6, 1), dtype=np.float32))               # 1800×2520: más que WORK_LONG_SIDE
    bigd = np.kron(depth, np.ones((6, 6), dtype=np.float32))
    out = lens_blur(big, bigd, LensBlur(enabled=True, focus=0.3, sharp_range=0.05, aperture=2.8))
    assert out.shape == big.shape and out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0


def test_invalid_target_is_rejected():
    with pytest.raises(ValueError):
        LensBlur(target="otro").validate()


# --- Proyecto y estilos ---------------------------------------------------------------------------------
def project_with_lens(**kw):
    p = Project(ramp=ramp_preset("linear"))
    p.lens = LensBlur(enabled=True, focus=0.3, sharp_range=0.05, aperture=2.0, **kw)
    return p


def test_lens_roundtrips_in_json_and_old_projects_load_without_it():
    p = project_with_lens(blades=7, rotation=12.0)
    q = Project.from_dict(p.to_dict())
    assert q.lens == p.lens
    legacy = p.to_dict()
    del legacy["lens"]
    assert Project.from_dict(legacy).lens == LensBlur()                    # sin la clave: desactivado y por defecto


def test_render_applies_lens_to_the_result_or_only_to_the_photo():
    img, depth, _ = checker_scene(h=120, w=160)
    photo = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    plain = Project(ramp=ramp_preset("linear"))
    plain.composite.opacity = 1.0
    sharp = render(plain, depth, photo)
    on_result = project_with_lens()
    on_result.composite.opacity = 1.0
    blurred_all = render(on_result, depth, photo)
    assert not np.allclose(blurred_all, sharp, atol=1e-3)
    on_photo = project_with_lens(target="photo")
    on_photo.composite.opacity = 1.0
    blurred_photo = render(on_photo, depth, photo)
    assert not np.allclose(blurred_photo, sharp, atol=1e-3)
    no_photo = project_with_lens(target="photo")                            # sin mezcla con la foto no hay nada que desenfocar
    assert np.allclose(render(no_photo, depth, photo), render(plain.__class__(ramp=ramp_preset("linear")), depth, photo))
    off = project_with_lens()
    off.lens.enabled = False
    assert np.array_equal(render(off, depth, photo), render(Project(ramp=ramp_preset("linear")), depth, photo))


def test_style_presets_carry_the_lens_and_builtins_switch_it_off():
    p = project_with_lens(blades=5)
    preset = presets.from_project("style", p)
    assert preset["lens"]["blades"] == 5 and preset["lens"]["enabled"]
    out = presets.apply_to_project_dict("style", Project(ramp=ramp_preset("linear")).to_dict(), preset)
    assert Project.from_dict(out).lens.blades == 5
    again = presets.apply_to_project_dict("style", out, presets.builtin("style", presets.builtin_names("style")[0][0]))
    assert Project.from_dict(again).lens.enabled is False                  # un estilo sin desenfoque lo apaga


# --- Mapa de zona en foco ---------------------------------------------------------------------------------
def test_focus_map_is_bright_in_focus_dark_far_away_and_follows_the_aperture():
    depth = np.tile(np.linspace(0, 1, 201, dtype=np.float32), (20, 1))
    p = LensBlur(focus=0.5, sharp_range=0.1, aperture=2.8)
    m = dof.focus_map(depth, p)
    assert m.dtype == np.uint8 and m.shape == depth.shape
    assert (m[:, 90:111] == 255).all()                                     # zona nítida: verde completo
    assert m[:, 0].max() == 0 and m[:, -1].max() == 0                      # lo más lejano del foco: sin verde
    assert (np.diff(m[0, 110:].astype(int)) <= 0).all()                    # baja al alejarse del foco
    near_off = dof.focus_map(depth, LensBlur(focus=0.5, sharp_range=0.1, aperture=2.8, near=0.0))
    assert near_off[:, :90].min() == 255                                   # sin desenfoque delante: todo nítido por ese lado
    big = np.kron(depth, np.ones((8, 8), dtype=np.float32))                # una foto grande: se calcula pequeña
    assert max(dof.focus_map(big, p).shape) <= 512


# --- Ojo de gato -------------------------------------------------------------------------------------------
def test_squeezed_kernel_is_a_lens_narrower_on_the_chosen_axis_and_normalized():
    round_k = dof.aperture_kernel(12, 0, 0.0, 0.1)
    kx = dof.aperture_kernel(12, 0, 0.0, 0.1, squeeze=0.9, axis="x")
    ky = dof.aperture_kernel(12, 0, 0.0, 0.1, squeeze=0.9, axis="y")
    for k in (round_k, kx, ky):
        assert k.sum() == pytest.approx(1.0, abs=1e-5)

    def extent(k, axis):
        support = np.flatnonzero((k > 0.5 * k.max()).any(axis=axis))
        return support.max() - support.min() + 1

    # Recortado por los lados: más estrecho en x que en y; el otro eje, al revés; el redondo, igual en los dos.
    assert extent(kx, 0) < 0.7 * extent(kx, 1)
    assert extent(ky, 1) < 0.7 * extent(ky, 0)
    assert abs(extent(round_k, 0) - extent(round_k, 1)) <= 1
    assert np.allclose(dof.aperture_kernel(12, 0, 0.0, 0.1, squeeze=0.0), round_k)      # sin recorte = el redondo
    assert np.allclose(kx, kx[::-1, ::-1], atol=1e-6)                                       # simétrico (almendra)


def test_edge_weights_sum_to_one_are_round_in_the_center_and_squeezed_toward_the_edges():
    w_round, w_x, w_y = dof.edge_weights((60, 100), 1.0)
    assert np.allclose(w_round + w_x + w_y, 1.0, atol=1e-5)
    assert w_round[30, 50] > 0.99                                       # centro: todo redondo
    assert w_x[30, 0] > 0.7 and w_y[30, 0] < 0.05                       # borde izquierdo: recortado por los lados
    assert w_y[0, 50] > 0.7 and w_x[0, 50] < 0.05                       # borde superior: recortado arriba y abajo
    assert w_x[0, 0] == pytest.approx(w_y[0, 0], abs=5e-3)              # esquina: se reparte
    half = dof.edge_weights((60, 100), 0.5)
    assert half[0].min() > w_round.min() and half[1].max() < w_x.max()  # con menos fuerza, menos recorte
    zero = dof.edge_weights((60, 100), 0.0)
    assert np.allclose(zero[0], 1.0) and not zero[1].any() and not zero[2].any()


def bokeh_scene(point_xy):
    """Imagen oscura con un punto de luz en primer plano desenfocado (profundidad lejos del foco)."""
    h, w = 120, 160
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[point_xy[1], point_xy[0]] = 1.0
    depth = np.full((h, w), 0.95, dtype=np.float32)               # todo lejos: el foco queda en 0,1
    return image, depth


def blob_size(out):
    ys, xs = np.nonzero(out[..., 0] > 0.3 * out[..., 0].max())
    return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


def test_cat_eye_changes_edge_bokeh_but_not_the_center_and_zero_is_identical():
    params = dof.LensBlur(enabled=True, focus=0.1, sharp_range=0.0, aperture=1.0, highlights=1.0, softness=0.1)
    centre_img, depth = bokeh_scene((80, 60))
    edge_img, _ = bokeh_scene((14, 60))                               # cerca del borde izquierdo
    base_centre = dof.lens_blur(centre_img, depth, params)
    base_edge = dof.lens_blur(edge_img, depth, params)
    cw, ch = blob_size(base_centre)
    assert abs(cw - ch) <= 2                                           # disco redondo
    zero = dof.LensBlur(**{**params.__dict__, "cat_eye": 0.0})
    assert np.array_equal(dof.lens_blur(edge_img, depth, zero), base_edge)       # 0 = idéntico a antes
    cat = dof.LensBlur(**{**params.__dict__, "cat_eye": 1.0})
    cat_centre = dof.lens_blur(centre_img, depth, cat)
    cat_edge = dof.lens_blur(edge_img, depth, cat)
    assert blob_size(cat_centre) == blob_size(base_centre)             # el centro no cambia
    ew, eh = blob_size(cat_edge)
    assert ew < 0.85 * eh and ew < blob_size(base_edge)[0]             # en el borde izquierdo: almendra alta y estrecha
    assert 0.0 <= cat_edge.min() and cat_edge.max() <= 1.0 and np.isfinite(cat_edge).all()
