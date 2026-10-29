"""Tiempos de cada cálculo de profundidad y lista de cálculos por foto."""

import numpy as np
import pytest
from PIL import Image

from deepshot import depth as d
from deepshot import depth_runs as dr


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))


def photo(path, size=(32, 24), color=(90, 100, 110)):
    Image.new("RGB", size, color).save(path)
    return path


class FakeEstimator(d.DepthEstimator):
    """Sin modelo: `_infer` tarda un tiempo conocido y devuelve un mapa fijo."""

    def __init__(self, seconds=0.3):
        super().__init__()
        self.seconds = seconds
        self.calls = 0

    def _infer(self, image_path, settings, progress):
        import time

        self.calls += 1
        time.sleep(self.seconds)
        self._timing = (0.0, self.seconds)
        return np.full((24, 32), 0.4, dtype=np.float32)


def test_estimate_measures_the_inference_time_and_stores_it_next_to_the_cached_map(tmp_path):
    p = photo(tmp_path / "a.png")
    est = FakeEstimator(0.3)
    s = d.settings_for(quality="high")
    est.estimate(p, s)
    run = est.last_run
    assert run.model == s.model and run.processing_resolution == 1280 and run.ensemble_size == 1
    assert not run.from_cache and run.infer_seconds == pytest.approx(0.3) and run.seconds == pytest.approx(0.3)
    png = d.cache_path(p, s, raw=True)
    assert d.read_timing(png)["infer_seconds"] == pytest.approx(0.3)             # el tiempo queda en la caché
    # Segunda vez: del disco, sin ejecutar el modelo, pero con el tiempo del cálculo original.
    again = FakeEstimator(5.0)
    again.estimate(p, s)
    assert again.calls == 0 and again.last_run.from_cache and again.last_run.infer_seconds == pytest.approx(0.3)


def test_refine_time_is_added_and_a_second_refine_reads_the_raw_time_from_cache(tmp_path):
    p = photo(tmp_path / "a.png")
    est = FakeEstimator(0.2)
    est.estimate(p, d.settings_for(quality="fast"))                                      # sin refinar
    refined = d.settings_for(quality="fast", refine=1.0)
    est2 = FakeEstimator(5.0)
    est2.estimate(p, refined)                                                            # reutiliza el mapa sin refinar
    assert est2.calls == 0
    run = est2.last_run
    assert run.infer_seconds == pytest.approx(0.2) and run.refine_seconds > 0
    assert run.seconds == pytest.approx(run.infer_seconds + run.refine_seconds)
    assert d.read_timing(d.cache_path(p, refined))["refine_seconds"] == pytest.approx(run.refine_seconds)


def test_old_cache_without_timing_file_still_loads_with_unknown_time(tmp_path):
    p = photo(tmp_path / "a.png")
    s = d.DepthSettings()
    d.save_depth_png16(np.full((24, 32), 0.25, np.float32), d.cache_path(p, s))        # sin .json
    est = FakeEstimator()
    est.estimate(p, s)
    assert est.last_run.from_cache and est.last_run.seconds is None
    assert dr.format_seconds(None) == "—" and dr.format_seconds(12.34) == "12,3 s" and dr.format_seconds(125) == "2 min 05 s"
    assert d.read_timing(tmp_path / "no_existe.png") == {}


def test_quality_labels_and_ids():
    s = d.settings_for(quality="max")
    assert dr.quality_label(s) == "Máxima 1280 px ×3" and dr.quality_id(s) == "max"
    odd = d.DepthSettings(processing_resolution=1000, ensemble_size=1)
    assert dr.quality_label(odd) == "1000 px" and dr.quality_id(odd) is None


def test_run_list_replaces_same_settings_caps_and_never_forgets_the_one_in_use(tmp_path):
    runs = dr.RunList(max_runs=3)
    entries = [dr.RunEntry(d.settings_for(quality=q, refine=r)) for q, r in
               (("fast", 0.0), ("high", 0.0), ("max", 0.0), ("fast", 1.0))]
    current = entries[0]
    for e in entries[:3]:
        runs.add(e, keep_key=current.key)
    again = dr.RunEntry(d.settings_for(quality="high"), run=d.DepthRun("m", 1280, 1, 0.0, infer_seconds=2.0))
    runs.add(again, keep_key=current.key)
    assert len(runs.entries) == 3 and runs.get(again.key) is again                      # misma configuración: se sustituye
    runs.add(entries[3], keep_key=current.key)                                           # se pasa del máximo
    assert len(runs.entries) == 3 and runs.get(current.key) is current                  # el de en uso se queda
    assert runs.entries[-1] is entries[3]
    assert runs.remove(entries[3].key) and not runs.remove("no existe")
    runs.keep_only(current.key)
    assert runs.entries == [current]


def test_discover_finds_cached_runs_with_their_times_and_loads_maps_lazily(tmp_path):
    p = photo(tmp_path / "a.png")
    for quality in ("fast", "max"):
        FakeEstimator(0.2).estimate(p, d.settings_for(quality=quality))
    d.DepthEstimator()                                                                    # otra foto: no debe aparecer
    other = photo(tmp_path / "b.png", color=(1, 2, 3))
    FakeEstimator(0.2).estimate(other, d.settings_for(quality="high"))
    found = dr.discover(p, ["marigold-v1-0", "depth-anything-v2-small"])
    assert sorted(dr.quality_id(e.settings) for e in found) == ["fast", "max"]
    assert all(e.run.from_cache and e.seconds == pytest.approx(0.2) and e.depth is None for e in found)
    depth = found[0].get_depth()
    assert depth.shape == (24, 32) and np.allclose(depth, 0.4, atol=1e-3) and found[0].depth is depth
    found[0].cache_file.unlink()
    gone = dr.RunEntry(found[0].settings, cache_file=found[0].cache_file)
    assert gone.get_depth() is None                                                       # archivo desaparecido: sin romper
