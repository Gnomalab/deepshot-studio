"""Mapas auxiliares (deepshot/maps.py): caché en disco, productor «subject» con el modelo parcheado y errores."""

import numpy as np
import pytest
from PIL import Image

from deepshot import maps
from deepshot.look import LookLayer, apply_look


@pytest.fixture(autouse=True)
def _own_data_dir(tmp_path, monkeypatch):
    """Cada prueba con su caché de mapas vacía (la carpeta de datos de las pruebas se comparte)."""
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "datos"))


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "foto.png"
    Image.new("RGB", (80, 60), (120, 90, 60)).save(path)
    return path


@pytest.fixture
def depth():
    d = np.ones((60, 80), dtype=np.float32)
    d[20:40, 50:70] = 0.0                                   # lo más cercano, a la derecha
    return d


@pytest.fixture
def fake_sam(monkeypatch):
    """Sin pesos reales: el modelo está «instalado» y `segment_at` devuelve dos lecturas (parte y conjunto)."""
    calls = []
    small, big = np.zeros((60, 80), np.float32), np.zeros((60, 80), np.float32)
    small[28:32, 58:62], big[15:45, 45:75] = 1.0, 1.0

    def fake_segment(image, point):
        calls.append((image.shape, point))
        return np.stack([small, big])

    monkeypatch.setattr(maps, "sam_installed", lambda: True)
    monkeypatch.setattr(maps, "segment_at", fake_segment)
    return calls


def test_el_sujeto_se_calcula_se_guarda_en_cache_y_no_se_repite(photo, depth, fake_sam):
    steps = []
    mask = maps.compute(maps.SUBJECT, photo, depth, progress=lambda m, f: steps.append(f))
    assert mask.shape == (60, 80) and mask.dtype == np.float32
    assert mask[30, 60] == 1.0 and mask[0, 0] == 0.0           # el conjunto, no la parte
    assert steps[-1] == 1.0 and len(fake_sam) == 1
    x, y = fake_sam[0][1]
    assert 0.6 < x < 0.8 and 0.4 < y < 0.6                      # punto automático: lo más cercano del mapa
    again = maps.compute(maps.SUBJECT, photo, depth)             # segunda vez: de la caché, sin llamar al modelo
    assert len(fake_sam) == 1 and np.abs(again - mask).max() < 2e-5
    assert maps.cached(maps.SUBJECT, photo, depth) is not None


def test_la_cache_depende_de_la_foto_el_reencuadre_y_la_profundidad(photo, depth, fake_sam, tmp_path):
    base = maps.cache_path(maps.SUBJECT, photo, depth)
    assert maps.cache_path(maps.SUBJECT, photo, depth, crop=(0.0, 0.0, 0.5, 0.5)) != base
    other_depth = depth[:, ::-1].copy()
    assert maps.cache_path(maps.SUBJECT, photo, other_depth) != base
    copy = tmp_path / "copia.png"
    copy.write_bytes(photo.read_bytes())
    assert maps.cache_path(maps.SUBJECT, copy, depth) == base    # el contenido manda, no el nombre
    Image.new("RGB", (80, 60), (1, 2, 3)).save(copy)
    assert maps.cache_path(maps.SUBJECT, copy, depth) != base


def test_sin_el_modelo_avisa_con_un_error_claro_y_no_guarda_nada(photo, depth, monkeypatch):
    monkeypatch.setattr(maps, "sam_installed", lambda: False)
    with pytest.raises(maps.MapError):
        maps.compute(maps.SUBJECT, photo, depth)
    assert maps.cached(maps.SUBJECT, photo, depth) is None
    assert not maps.is_available(maps.SUBJECT)
    with pytest.raises(maps.MapError):
        maps.compute("no_existe", photo, depth)


def test_borrar_la_cache_y_reescalar(photo, depth, fake_sam):
    maps.compute(maps.SUBJECT, photo, depth)
    files, size = maps.clear_cache()
    assert files >= 1 and size > 0 and maps.cached(maps.SUBJECT, photo, depth) is None
    m = np.zeros((10, 20), np.float32)
    m[:, 10:] = 1.0
    r = maps.resize(m, (40, 20))
    assert r.shape == (20, 40) and r[0, 0] == 0.0 and r[0, -1] == 1.0
    assert maps.resize(m, (20, 10)) is m


def test_una_capa_limitada_al_sujeto_solo_actua_en_el_sujeto_y_se_puede_invertir():
    image = np.full((60, 80, 3), 0.5, np.float32)
    subject = np.zeros((60, 80), np.float32)
    subject[15:45, 45:75] = 1.0
    layer = LookLayer.new("levels")
    layer.params["black"] = 0.9                                 # lo pone todo casi negro
    layer.limit, layer.mask_id = "mask", maps.SUBJECT
    out = apply_look(image, [layer], None, {maps.SUBJECT: subject})
    assert out[30, 60, 0] < 0.1 and np.allclose(out[0, 0], 0.5)
    layer.mask_invert = True
    out = apply_look(image, [layer], None, {maps.SUBJECT: subject})
    assert np.allclose(out[30, 60], 0.5) and out[0, 0, 0] < 0.1
    assert np.allclose(apply_look(image, [layer], None, {}), 0.5)      # sin mapa calculado, la capa no actúa
