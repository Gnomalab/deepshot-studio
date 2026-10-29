"""Tiempo transcurrido y estimado de las tareas largas."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from deepshot.timing import Durations, ProgressClock, format_clock  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_format_clock():
    assert format_clock(0) == "0:00" and format_clock(12.4) == "0:12" and format_clock(65) == "1:05"
    assert format_clock(3700) == "1:01:40" and format_clock(-5) == "0:00"


def test_estimate_from_fraction_waits_for_enough_data_and_converges():
    now = FakeClock()
    clock = ProgressClock(now=now)
    clock.update(0.10)
    assert clock.snapshot().remaining is None                    # demasiado pronto para fiarse
    now.t += 10
    clock.update(0.25)
    snap = clock.snapshot()
    assert snap.elapsed == 10 and abs(snap.remaining - 30) < 1e-6          # 10 s hechos = 25 % → faltan 30 s
    now.t += 30
    clock.update(1.0)
    assert clock.snapshot().remaining == 0


def test_estimate_from_history_when_there_is_no_fraction_and_overrun_is_flagged():
    now = FakeClock()
    clock = ProgressClock(expected=20, now=now)
    clock.update(-1)                                               # sin fracción medible
    now.t += 5
    snap = clock.snapshot()
    assert snap.remaining == 15 and not snap.overrun
    now.t += 30                                                    # ya pasó lo habitual
    snap = clock.snapshot()
    assert snap.remaining is None and snap.overrun
    plain = ProgressClock(now=now)
    assert plain.snapshot().remaining is None and not plain.snapshot().overrun   # sin historial ni fracción


def test_durations_keep_the_last_five_use_the_median_and_ignore_instant_tasks():
    class Store(dict):
        def value(self, key, default=None):
            return self.get(key, default)

        def setValue(self, key, value):
            self[key] = value

    d = Durations(Store())
    assert d.expected("depth/x") is None
    for seconds in (10, 12, 11, 100, 13, 14):                      # el 100 es un cálculo raro
        d.record("depth/x", seconds)
    d.record("depth/x", 0.05)                                      # caché: no cuenta
    assert d.recent("depth/x") == [12, 11, 100, 13, 14]
    assert d.expected("depth/x") == 13
    d._settings["timing/depth/y"] = "7.5"                          # QSettings devuelve un solo valor como texto
    assert d.expected("depth/y") == 7.5


def test_task_timer_text_and_stop():
    from PySide6.QtWidgets import QApplication

    from deepshot.ui.task_timer import TaskTimer, with_time

    QApplication.instance() or QApplication([])
    timer = TaskTimer()
    assert timer.text() == "" and with_time("Hola", timer) == "Hola"      # sin cronómetro, no se añade nada
    timer.start(expected=30)
    assert timer.is_running() and timer.text().startswith("0:0") and "quedan" in timer.text()
    assert "Calculando  ·  0:0" in with_time("Calculando", timer)
    assert timer.stop() >= 0 and not timer.is_running() and timer.text() == ""
