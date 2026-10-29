"""Look: pila de capas de ajuste (niveles, contraste, saturación), su orden y su paso por el proyecto."""

import numpy as np

from deepshot.look import KINDS, LookLayer, apply_look, brightness_contrast, layers_from_list, levels, saturation
from deepshot.project import Project, render


def _image():
    rng = np.random.default_rng(1)
    return rng.random((24, 32, 3)).astype(np.float32)


def test_capas_en_su_valor_inicial_no_cambian_nada():
    img = _image()
    assert np.allclose(levels(img), img, atol=1e-6)
    assert np.allclose(brightness_contrast(img), img, atol=1e-6)
    assert np.allclose(saturation(img), img, atol=1e-6)
    for kind in KINDS:
        assert LookLayer.new(kind).is_identity()
        assert np.allclose(apply_look(img, [LookLayer.new(kind)]), img)


def test_niveles_recortan_y_gamma_aclara():
    img = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
    out = levels(img, black=0.2, white=0.8)
    assert out[0, 0, 0] == 0.0 and out[0, 0, 2] == 1.0 and np.isclose(out[0, 0, 1], 0.5)
    assert levels(img, gamma=2.0)[0, 0, 1] > 0.5          # gamma > 1 aclara los medios tonos


def test_saturacion_menos_uno_da_gris():
    out = saturation(_image(), -1.0)
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-5) and np.allclose(out[..., 1], out[..., 2], atol=1e-5)


def test_contraste_separa_los_tonos():
    img = np.array([[[0.4, 0.5, 0.6]]], dtype=np.float32)
    out = brightness_contrast(img, 0.0, 0.5)
    assert out[0, 0, 0] < 0.4 and out[0, 0, 2] > 0.6 and np.isclose(out[0, 0, 1], 0.5)


def test_el_orden_de_las_capas_importa():
    img = _image()
    sat = LookLayer.new("saturation")
    sat.params["amount"] = 1.0
    lev = LookLayer.new("levels")
    lev.params["black"] = 0.3
    a = apply_look(img, [lev, sat])
    b = apply_look(img, [sat, lev])
    assert not np.allclose(a, b, atol=1e-3)


def test_ojo_apagado_y_opacidad_cero_no_aplican():
    img = _image()
    layer = LookLayer.new("saturation")
    layer.params["amount"] = -1.0
    layer.enabled = False
    assert np.allclose(apply_look(img, [layer]), img)
    layer.enabled, layer.opacity = True, 0.0
    assert np.allclose(apply_look(img, [layer]), img)
    layer.opacity = 0.5
    full = saturation(img, -1.0)
    assert np.allclose(apply_look(img, [layer]), img * 0.5 + full * 0.5, atol=1e-5)


def test_limitar_a_un_rango_de_profundidad():
    img = np.full((4, 8, 3), 0.5, dtype=np.float32)
    depth = np.tile(np.linspace(0, 1, 8, dtype=np.float32), (4, 1))
    layer = LookLayer.new("contrast")
    layer.params["brightness"] = 1.0                      # sube +0.5: 0.5 -> 1.0
    layer.limit, layer.depth_near, layer.depth_far, layer.depth_softness = "depth", 0.0, 0.3, 0.0
    out = apply_look(img, [layer], depth)
    assert out[0, 0, 0] == 1.0 and out[0, -1, 0] == 0.5   # cambia lo cercano, no lo lejano


def test_proyecto_guarda_y_carga_la_pila(tmp_path):
    p = Project()
    layer = LookLayer.new("levels")
    layer.params["gamma"] = 1.4
    layer.blend = "multiply"
    p.look = [layer, LookLayer.new("saturation")]
    path = tmp_path / "p.json"
    p.save(path)
    q = Project.load(path)
    assert [l.kind for l in q.look] == ["levels", "saturation"]
    assert q.look[0].params["gamma"] == 1.4 and q.look[0].blend == "multiply"
    assert q.look[0].params == {**KINDS["levels"].defaults, "gamma": 1.4}


def test_proyecto_antiguo_sin_look_y_tipos_desconocidos():
    assert Project.from_dict({}).look == []
    assert layers_from_list([{"kind": "no_existe"}, {"kind": "levels", "extra": 1}])[1].kind == "levels"
    kept = layers_from_list([{"kind": "module.no_existe", "params": {"x": 1}}])
    assert len(kept) == 1 and kept[0].missing and kept[0].is_identity() and kept[0].params == {"x": 1}   # se conserva


def test_render_aplica_el_look_al_final():
    depth = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (8, 1))
    p = Project()
    base = render(p, depth)
    layer = LookLayer.new("levels")
    layer.params["black"] = 0.5
    p.look = [layer]
    out = render(p, depth)
    assert out.mean() < base.mean()
