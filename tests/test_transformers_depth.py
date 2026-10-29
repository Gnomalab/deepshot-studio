"""Familia «transformers-depth» (MiDaS, ZoeDepth, Depth Pro): carga, cálculo, convención de salida y modelos de prueba."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

from deepshot import depth as d
from deepshot import models as m


def test_normalize_depth_follows_the_app_convention_for_inverse_and_metric_outputs():
    inverse = np.array([[0.0, 5.0], [10.0, 10.0]], np.float32)             # más grande = más cerca
    out = d.normalize_depth(inverse, metric=False)
    assert out.dtype == np.float32 and out[1, 0] == pytest.approx(0.0) and out[0, 0] == pytest.approx(1.0)
    meters = np.array([[1.0, 2.0], [4.0, 8.0]], np.float32)                # más grande = más lejos
    out = d.normalize_depth(meters, metric=True)
    assert out[0, 0] == pytest.approx(0.0) and out[1, 1] == pytest.approx(1.0)
    assert np.all(np.diff(out.reshape(-1)) > 0)                           # lo cercano sigue siendo lo más oscuro (0)
    assert d.normalize_depth(np.full((3, 3), 7.0, np.float32), metric=False).max() == 0.0   # mapa plano: sin dividir por 0


def _tiny_midas(folder):
    """Un MiDaS (DPT) diminuto con pesos al azar, guardado como un modelo de transformers de verdad."""
    from transformers import DPTConfig, DPTForDepthEstimation, DPTImageProcessorPil

    cfg = DPTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64, image_size=64,
                    patch_size=16, backbone_out_indices=[0, 1, 2, 3], neck_hidden_sizes=[8, 16, 32, 32],
                    fusion_hidden_size=16, head_in_index=-1)
    DPTForDepthEstimation(cfg).eval().save_pretrained(folder, safe_serialization=True)
    DPTImageProcessorPil(size={"height": 64, "width": 64}, keep_aspect_ratio=False).save_pretrained(folder)


def test_a_transformers_depth_model_loads_from_the_manager_and_gives_a_map_at_photo_size(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    models_dir = tmp_path / "models"
    manager = m.ModelManager(models_dir)
    snapshot = models_dir / "snap"
    snapshot.mkdir(parents=True)
    _tiny_midas(snapshot)
    (models_dir / "midas-3-0-dpt-large.complete.json").write_text(json.dumps({"snapshot": "snap"}), encoding="utf-8")
    assert manager.is_installed("midas-3-0-dpt-large")
    photo = tmp_path / "foto.png"
    Image.fromarray((np.random.rand(40, 60, 3) * 255).astype("uint8")).save(photo)

    estimator = d.DepthEstimator(device="cpu", manager=manager)
    depth = estimator.estimate(photo, d.settings_for(model="midas-3-0-dpt-large"), use_cache=False)
    assert depth.shape == (40, 60) and depth.dtype == np.float32
    assert 0.0 <= depth.min() <= depth.max() <= 1.0 and depth.max() > depth.min()
    assert estimator.loaded_model == "midas-3-0-dpt-large" and estimator.last_run is not None


def test_test_models_are_hidden_by_the_switch_but_still_found_by_id(monkeypatch, tmp_path):
    manager = m.ModelManager(tmp_path / "models")
    dev_ids = {e.id for e in m.MODELS_CATALOG if e.dev_only}
    assert {"depth-pro", "midas-3-1-beit-large", "zoedepth-nyu-kitti", "depth-anything-v2-large"} <= dev_ids
    monkeypatch.setenv("DEEPSHOT_DEV_MODELS", "1")
    assert dev_ids <= {e.id for e in manager.all()}
    monkeypatch.setenv("DEEPSHOT_DEV_MODELS", "0")
    shown = {e.id for e in manager.all()}
    assert not (dev_ids & shown) and m.DEFAULT_MODEL in shown
    assert manager.get("depth-pro").license == "apple-amlr"                  # oculto, pero se encuentra por su id
    # Todos los de prueba son de transformers con safetensors; los de licencia no permisiva llevan aviso.
    for entry in m.MODELS_CATALOG:
        if entry.dev_only:
            assert entry.family in ("depth-anything", "transformers-depth") and "model.safetensors" in entry.patterns
            assert m.is_permissive(entry.license) or entry.warning, entry.id
    assert manager.get("marigold-v2-log").min_vram_gb == 17.0 and not manager.get("marigold-v2-log").dev_only


def test_test_models_are_never_downloaded_by_a_calculation(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DEV_MODELS", "1")
    estimator = d.DepthEstimator(device="cpu", manager=m.ModelManager(tmp_path / "models"))
    for model_id in ("midas-3-1-beit-large", "zoedepth-nyu", "depth-pro"):
        with pytest.raises(m.ModelError):
            estimator.load(model_id)


def test_loading_is_serialised_so_the_startup_preload_and_a_calculation_never_load_at_once(tmp_path, monkeypatch):
    import threading
    import time

    estimator = d.DepthEstimator(device="cpu", manager=m.ModelManager(tmp_path / "models"))
    active, overlap = [], []

    def slow_load(model, progress):
        active.append(model)
        if len(active) > 1:
            overlap.append(True)
        time.sleep(0.15)
        active.pop()

    monkeypatch.setattr(estimator, "_load", slow_load)
    threads = [threading.Thread(target=estimator.load, args=("a", lambda *_: None)) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not overlap
