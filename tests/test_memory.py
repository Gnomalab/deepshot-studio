"""Medir y liberar memoria."""

from deepshot import memory


def test_memory_readings_are_numbers_or_none_and_release_does_not_fail():
    ram = memory.process_memory_mb()
    assert ram is None or ram > 10                       # un proceso de Python con Qt/numpy usa bastante más
    gpu = memory.gpu_memory_mb()
    assert gpu is None or gpu >= 0
    memory.release_python_memory()                       # no debe lanzar aunque no haya GPU ni torch


def test_level_traffic_light_uses_percentage_and_absolute_minimum():
    from deepshot import memory
    assert memory.level(6000, 8000) == "ok"
    assert memory.level(1500, 8000) == "warn"          # 19 % libre
    assert memory.level(500, 8000) == "bad"            # menos de 700 MB
    assert memory.level(50_000, 64_000) == "ok"
    assert memory.level(5000, 64_000) == "bad"         # <10 %
    assert memory.worst("ok", None, "warn") == "warn" and memory.worst("bad", "warn") == "bad" and memory.worst() == "ok"


def test_ram_free_total_is_measured_on_this_system():
    from deepshot import memory
    got = memory.ram_free_total_mb()
    assert got is None or (0 < got[0] <= got[1])


def test_on_mac_the_memory_is_measured_and_the_gpu_cache_is_returned_with_mps(monkeypatch):
    import sys
    import types

    if sys.platform == "darwin":
        assert memory.process_memory_mb() is not None and memory.ram_free_total_mb() is not None   # con psutil
    calls = []
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: calls.append("cuda")),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        mps=types.SimpleNamespace(empty_cache=lambda: calls.append("mps"), driver_allocated_memory=lambda: 3e9))
    memory.empty_gpu_cache(fake)
    assert calls == ["mps"]                                  # en Mac se vacía la caché de MPS, no la de CUDA
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert memory.gpu_memory_mb() == 3000.0
    memory.empty_gpu_cache(types.SimpleNamespace())          # un torch raro no rompe nada
