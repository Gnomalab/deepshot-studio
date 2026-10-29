"""Fase 0: la app arranca, muestra la ventana con tema y cierra sin errores."""

import os

# Sin ventana real: Qt dibuja en memoria (sirve también en CI).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402

from deepshot.app import create_app  # noqa: E402
from deepshot.paths import resource  # noqa: E402
from deepshot.ui.main_window import STEPS, MainWindow  # noqa: E402


def test_resources_exist():
    assert resource("theme", "theme.qss").is_file()
    assert resource("icons", "app.svg").is_file()


def test_window_opens_and_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    # Ajustes en un .ini temporal para no tocar los de la app real (registro de Windows).
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    app = create_app([])
    assert "QMainWindow" in app.styleSheet()  # tema cargado

    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.isVisible()
    assert not window.canvas.has_image()

    # Cada paso muestra solo sus secciones de ajustes.
    for i in range(len(STEPS)):
        window.set_step(i)
        for step, sections in window.sections_by_step.items():
            assert all(s.isVisibleTo(window) == (step == i) for s in sections)

    # Importar una imagen sintética la añade a la biblioteca y al lienzo.
    img = QImage(64, 48, QImage.Format_RGB32)
    img.fill(QColor("gray"))
    path = tmp_path / "foto prueba.png"  # con espacio, como las rutas reales
    img.save(str(path))
    window.add_photos([path])
    assert window.library.count() == 1
    assert window.canvas.has_image()

    window._probe.wait(120)   # no cerrar mientras otro hilo importa torch (cierre inestable)
    window.ask_before_closing = False
    window.close()
    app.processEvents()
