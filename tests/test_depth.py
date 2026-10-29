"""Fase 1: utilidades de profundidad (sin cargar el modelo)."""

import numpy as np
import pytest
from PIL import Image

from deepshot import depth as d


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Caché y modelos en una carpeta temporal, no en la de la app real."""
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))


def _photo(path, color=(120, 80, 40)):
    Image.new("RGB", (32, 24), color).save(path)
    return path


def test_png16_roundtrip_keeps_precision(tmp_path):
    depth = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(48, 64)
    path = tmp_path / "mapa 16.png"
    d.save_depth_png16(depth, path)
    with Image.open(path) as img:
        assert img.mode.startswith("I")  # 16 bits, no 8
    back = d.load_depth_png(path)
    assert back.shape == depth.shape
    assert np.abs(back - depth).max() <= 1 / 65535 + 1e-7


def test_external_map_is_normalized_and_inverted(tmp_path):
    # Mapa "cerca = claro" de 8 bits que no ocupa todo el rango.
    data = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    path = tmp_path / "externo.png"
    Image.fromarray(data).save(path)

    near_bright = d.load_external_depth(path, near_is_bright=True)
    assert near_bright.min() == 0.0 and near_bright.max() == 1.0
    assert near_bright[1, 1] == 0.0  # el píxel más claro es el más cercano -> 0

    near_dark = d.load_external_depth(path, near_is_bright=False)
    assert near_dark[1, 1] == 1.0


def test_cache_key_depends_on_content_and_settings(tmp_path):
    a = _photo(tmp_path / "a.png")
    b = _photo(tmp_path / "b.png", color=(1, 2, 3))
    s = d.DepthSettings()
    assert d.cache_path(a, s) != d.cache_path(b, s)
    assert d.cache_path(a, s) != d.cache_path(a, d.DepthSettings(ensemble_size=5))

    # Renombrar la foto no invalida la caché: la clave es el contenido.
    renamed = tmp_path / "otra carpeta" / "a renombrada.png"
    renamed.parent.mkdir()
    renamed.write_bytes(a.read_bytes())
    assert d.cache_path(renamed, s) == d.cache_path(a, s)


def test_estimate_uses_cache_without_loading_model(tmp_path):
    photo = _photo(tmp_path / "foto.png")
    expected = np.full((24, 32), 0.25, dtype=np.float32)
    d.save_depth_png16(expected, d.cache_path(photo, d.DepthSettings()))

    est = d.DepthEstimator()
    est.load = lambda *a, **k: pytest.fail("no debería cargar el modelo")
    result = est.estimate(photo)
    assert np.allclose(result, 0.25, atol=1e-4)


def test_refine_reuses_raw_model_output(tmp_path):
    """Cambiar solo el refinado no vuelve a ejecutar el modelo: parte del mapa en caché."""
    photo = _photo(tmp_path / "foto.png")
    raw = np.full((24, 32), 0.5, dtype=np.float32)
    plain = d.settings_for(quality="high")
    d.save_depth_png16(raw, d.cache_path(photo, plain, raw=True))

    est = d.DepthEstimator()
    est._infer = lambda *a, **k: pytest.fail("no debería ejecutar el modelo")
    refined = d.settings_for(quality="high", refine=1.0)
    out = est.estimate(photo, refined)
    assert out.shape == raw.shape
    assert d.cache_path(photo, refined).is_file()          # el refinado se guarda aparte
    assert d.cache_path(photo, refined) != d.cache_path(photo, plain)


def test_quality_presets_map_to_settings():
    s = d.settings_for(quality="max")
    assert (s.processing_resolution, s.ensemble_size) == (1280, 3)
    assert d.settings_for(quality="desconocida").processing_resolution == 768  # vuelve a Rápida


def test_a_manual_resolution_overrides_the_preset_and_changes_the_cache_key():
    base = d.settings_for(quality="high")
    custom = d.settings_for(quality="high", resolution=512)
    assert base.processing_resolution == 1280 and custom.processing_resolution == 512
    assert custom.ensemble_size == base.ensemble_size and custom.key() != base.key()
    assert d.settings_for(quality="high", resolution=None) == base


def test_cache_size_and_clear_only_remove_cache_files(tmp_path, monkeypatch):
    from deepshot import depth as d

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    folder = d.cache_dir()
    (folder / "a.png").write_bytes(b"x" * 100)
    (folder / "a.json").write_text("{}")
    (folder / "otra.txt").write_text("no es de la caché")
    outside = tmp_path / "data" / "fuera.png"
    outside.write_bytes(b"y")
    assert d.cache_size() == (2, 102)
    assert d.clear_cache() == (2, 102)
    assert d.cache_size() == (0, 0) and (folder / "otra.txt").exists() and outside.exists()
