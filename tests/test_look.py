"""Look: pila de capas de ajuste (niveles, contraste, saturación), su orden y su paso por el proyecto."""

import numpy as np
import pytest

from deepshot.look import KINDS, LookLayer, apply_look, brightness_contrast, layers_from_list, levels, saturation
from deepshot.project import Project, render


def _image(h=24, w=32):
    rng = np.random.default_rng(1)
    return rng.random((h, w, 3)).astype(np.float32)


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


def test_la_pila_se_aplica_de_abajo_arriba_como_photoshop():
    img = _image()
    sat = LookLayer.new("saturation")
    sat.params["amount"] = -1.0
    lev = LookLayer.new("levels")
    lev.params["white"] = 0.5
    # [sat, lev]: «lev» está abajo y recibe primero la imagen; «sat» está arriba y va la última.
    expected = saturation(levels(img, 0.0, 0.5, 1.0), -1.0)
    assert np.allclose(apply_look(img, [sat, lev]), expected, atol=1e-6)
    assert not np.allclose(apply_look(img, [lev, sat]), expected, atol=1e-3)


def test_grano_nitidez_y_vineta_son_efectos_de_serie():
    img = _image(40, 60)
    for kind, change in (("grain", {"amount": 0.8}), ("sharpen", {"amount": 1.5}), ("vignette", {"amount": 0.9})):
        assert kind in KINDS and not KINDS[kind].module_id
        layer = LookLayer.new(kind)
        assert layer.is_identity() and np.allclose(apply_look(img, [layer]), img)
        layer.params.update(change)
        out = apply_look(img, [layer])
        assert out.shape == img.shape and out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0
        assert not np.allclose(out, img, atol=1e-3)


def test_el_grano_es_siempre_el_mismo_y_la_vineta_oscurece_los_bordes():
    img = np.full((40, 60, 3), 0.5, dtype=np.float32)
    grain = LookLayer.new("grain")
    grain.params["amount"] = 1.0
    assert np.array_equal(apply_look(img, [grain]), apply_look(img, [grain]))        # reproducible
    vignette = LookLayer.new("vignette")
    vignette.params["amount"] = 1.0
    out = apply_look(img, [vignette])
    assert out[20, 30, 0] == pytest.approx(0.5) and out[0, 0, 0] < 0.3
    vignette.params["amount"] = -1.0                                                  # negativa: aclara los bordes
    assert apply_look(img, [vignette])[0, 0, 0] > 0.7


def test_la_nitidez_y_el_grano_se_ven_igual_a_otra_resolucion():
    yy, xx = np.mgrid[0:200, 0:300]
    plane = 0.5 + 0.35 * np.sin(xx / 25.0) * np.cos(yy / 30.0) + 0.1 * (xx > 150)       # suave y con un borde
    big = np.repeat(plane[..., None], 3, axis=-1).astype(np.float32)
    small = np.ascontiguousarray(big[::2, ::2])                                           # la misma imagen a mitad de tamaño
    layer = LookLayer.new("sharpen")
    layer.params.update({"amount": 2.0, "radius": 0.6})
    assert float(np.abs(apply_look(big, [layer])[::2, ::2] - apply_look(small, [layer])).mean()) < 0.02
    grain = LookLayer.new("grain")
    grain.params["amount"] = 1.0
    flat_big, flat_small = np.full((200, 300, 3), 0.5, np.float32), np.full((100, 150, 3), 0.5, np.float32)
    assert float((apply_look(flat_big, [grain]) - flat_big).std()) == pytest.approx(
        float((apply_look(flat_small, [grain]) - flat_small).std()), rel=0.15)           # la misma fuerza de grano


def test_los_proyectos_con_los_antiguos_modulos_pasan_a_los_efectos_de_serie():
    layers = layers_from_list([{"kind": "module.film_grain", "params": {"amount": 0.4, "size": 2.0}},
                               {"kind": "module.sharpen"}, {"kind": "module.vignette", "enabled": False}])
    assert [l.kind for l in layers] == ["grain", "sharpen", "vignette"] and not layers[0].missing
    assert layers[0].params == {"amount": 0.4, "size": 2.0} and layers[2].enabled is False


def test_el_nombre_propio_de_la_capa_se_guarda(tmp_path):
    p = Project()
    layer = LookLayer.new("levels")
    layer.name = "Contraste de la piel"
    p.look = [layer]
    p.save(tmp_path / "p.json")
    assert Project.load(tmp_path / "p.json").look[0].name == "Contraste de la piel"
    assert Project.from_dict({"look": [{"kind": "levels"}]}).look[0].name == ""      # proyecto sin nombres



def test_new_builtin_effects_are_neutral_by_default_and_act_when_raised():
    import numpy as np

    from deepshot import look

    rng = np.random.default_rng(3)
    img = rng.random((60, 80, 3)).astype(np.float32)
    depth = np.tile(np.linspace(0, 1, 80, dtype=np.float32), (60, 1))
    for key, param, value in (("clarity", "amount", 1.0), ("glow", "amount", 1.0), ("chromatic", "amount", 1.0),
                              ("haze", "amount", 0.8), ("temperature", "temp", 0.8)):
        kind = look.KINDS[key]
        neutral = kind.apply(img, depth, kind.defaults)
        assert np.allclose(neutral, img, atol=1e-6), key                    # con los valores de partida no cambia nada
        raised = kind.apply(img, depth, {**kind.defaults, param: value})
        assert raised.shape == img.shape and np.isfinite(raised).all()
        assert float(np.abs(raised - img).mean()) > 1e-3, key


def test_haze_only_touches_the_far_zones_and_needs_a_depth_map():
    import numpy as np

    from deepshot import look

    img = np.zeros((10, 10, 3), np.float32)
    depth = np.zeros((10, 10), np.float32)
    depth[:, 5:] = 1.0                                                      # mitad cerca, mitad lejos
    out = look.depth_haze(img, depth, amount=1.0, start=0.3, tone=1.0)
    assert out[:, :5].max() == 0.0 and out[:, 5:].min() == 1.0
    assert look.depth_haze(img, None, amount=1.0) is img                    # sin profundidad: no hace nada


def test_a_layer_limited_to_a_map_only_acts_there_and_waits_for_the_map():
    import numpy as np

    from deepshot import look

    img = np.full((8, 8, 3), 0.2, np.float32)
    layer = look.LookLayer.new("temperature")
    layer.params["temp"] = 1.0
    layer.limit, layer.mask_id = "mask", "subject"
    mask = np.zeros((8, 8), np.float32)
    mask[:, :4] = 1.0
    out = look.apply_look(img, [layer], None, {"subject": mask})
    assert not np.allclose(out[:, :4], img[:, :4]) and np.allclose(out[:, 4:], img[:, 4:])      # solo el sujeto
    layer.mask_invert = True
    out = look.apply_look(img, [layer], None, {"subject": mask})
    assert np.allclose(out[:, :4], img[:, :4]) and not np.allclose(out[:, 4:], img[:, 4:])      # solo el fondo
    assert np.allclose(look.apply_look(img, [layer], None, None), img)                          # sin mapa: no actúa


def test_pre_fx_retouch_only_the_photo_before_it_is_blended_and_survive_json():
    import numpy as np

    from deepshot import look
    from deepshot.project import Project, render

    depth = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (12, 1))
    photo = np.full((12, 16, 3), 0.3, np.float32)
    project = Project()
    project.composite.opacity = 1.0                                         # solo se ve la foto
    plain = render(project, depth, photo)
    layer = look.LookLayer.new("temperature")
    layer.params["temp"] = 1.0
    project.pre = [layer]
    warm = render(project, depth, photo)
    assert warm[..., 0].mean() > plain[..., 0].mean()                       # el Pre FX cambia la foto mezclada
    again = Project.from_dict(project.to_dict())
    assert again.pre[0].kind == "temperature" and again.pre[0].params["temp"] == 1.0
    assert np.allclose(render(again, depth, photo), warm)
    project.composite.opacity = 0.0                                         # sin foto en la mezcla, el Pre FX no pinta nada
    assert np.allclose(render(project, depth, photo), render(Project(), depth, photo))
