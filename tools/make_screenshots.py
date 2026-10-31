"""Genera las capturas de la guía (docs/images) con una escena de ejemplo hecha por código.

Uso (desde la carpeta del proyecto, con el entorno activado):
    QT_QPA_PLATFORM=offscreen python tools/make_screenshots.py

No usa ninguna foto personal ni ningún modelo: la foto y su mapa de profundidad se dibujan aquí mismo (un paisaje
con montañas, lago y pinos), el mapa se carga como si fuera «externo» y el resto sale de los presets de fábrica.
Los ajustes y datos van a una carpeta temporal, así que no toca los del usuario.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
TMP = Path(tempfile.mkdtemp(prefix="deepshot_shots_"))
os.environ["DEEPSHOT_DATA_DIR"] = str(TMP / "data")          # antes de importar la aplicación

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402

sys.path.insert(0, str(ROOT))
from deepshot import presets  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.areas import Area  # noqa: E402
from deepshot.tonemap import Band  # noqa: E402
from deepshot.ui.main_window import (  # noqa: E402
    STEP_DEPTH, STEP_EDIT, STEP_EXPORT, STEP_IMPORT, STEP_LOOK, VIEW_COMPARE, VIEW_DEPTH, VIEW_PHOTO, VIEW_RESULT,
    MainWindow,
)

W, H = 1600, 1066            # tamaño de la foto de ejemplo
WINDOW = (1500, 940)         # tamaño de la ventana en las capturas


# ---------------------------------------------------------------------------
# La escena de ejemplo: foto y mapa de profundidad (0 = cerca, 1 = lejos)
# ---------------------------------------------------------------------------
def make_scene() -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    horizon = 0.58 * H
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.ones((H, W), dtype=np.float32)

    # Cielo de atardecer con un sol difuso.
    t = np.clip(yy / horizon, 0, 1)[..., None]
    sky = (1 - t) * np.array([0.16, 0.20, 0.45]) + t * np.array([0.98, 0.62, 0.38])
    sun = np.exp(-(((xx - 0.68 * W) ** 2 + (yy - 0.47 * H) ** 2) / (2 * (0.07 * W) ** 2)))[..., None]
    rgb[:] = np.clip(sky + sun * np.array([0.55, 0.40, 0.15]), 0, 1)

    # Cuatro cordilleras: cada capa más cerca, más oscura y menos neblinosa.
    for i, (base, amp, depth_value, haze) in enumerate([(0.52, 0.10, 0.92, 0.55), (0.57, 0.12, 0.78, 0.40),
                                                        (0.63, 0.14, 0.64, 0.25), (0.70, 0.12, 0.50, 0.10)]):
        ridge = base * H - amp * H * (0.5 + 0.5 * np.sin(xx / W * (5 + 2 * i) + i * 1.7)
                                       + 0.25 * np.sin(xx / W * (17 + 5 * i) + i))
        mask = yy > ridge
        tone = np.array([0.22, 0.20, 0.34]) * (1 - i * 0.18)
        color = (1 - haze) * tone + haze * np.array([0.80, 0.55, 0.50])
        rgb[mask] = color
        depth[mask] = depth_value

    # Lago con reflejo del cielo, de lejos (arriba) a cerca (abajo).
    lake = yy > 0.76 * H
    k = np.clip((yy - 0.76 * H) / (0.24 * H), 0, 1)
    reflection = (1 - k[..., None]) * np.array([0.78, 0.50, 0.42]) + k[..., None] * np.array([0.10, 0.16, 0.28])
    ripple = 0.03 * np.sin(yy * 0.35 + xx * 0.01)[..., None]
    rgb[lake] = np.clip(reflection + ripple, 0, 1)[lake]
    depth[lake] = (0.45 - 0.40 * k)[lake]

    # Pinos del primer plano a izquierda y derecha.
    def pine(cx: float, base_y: float, height: float, d: float) -> None:
        for j in range(7):
            half = (0.05 + 0.012 * j) * height
            top = base_y - height + j * height * 0.13
            tri = (yy > top) & (yy < top + height * 0.28) & (np.abs(xx - cx) < half * (yy - top) / (height * 0.28))
            rgb[tri] = np.array([0.04, 0.07, 0.07]) * (1 + 0.05 * j)
            depth[tri] = d
        trunk = (np.abs(xx - cx) < height * 0.012) & (yy > base_y - height * 0.1) & (yy < base_y)
        rgb[trunk] = np.array([0.05, 0.04, 0.04])
        depth[trunk] = d

    for cx, base_y, height, d in [(0.08 * W, 1.02 * H, 0.95 * H, 0.06), (0.20 * W, 1.0 * H, 0.60 * H, 0.14),
                                  (0.92 * W, 1.04 * H, 0.90 * H, 0.05), (0.80 * W, 1.0 * H, 0.50 * H, 0.15)]:
        pine(cx, base_y, height, d)

    rgb += rng.normal(0, 0.012, rgb.shape).astype(np.float32)             # un poco de grano de cámara
    photo = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    from scipy.ndimage import gaussian_filter

    return photo, np.clip(gaussian_filter(depth, 1.2), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Capturas
# ---------------------------------------------------------------------------
def wait(app, seconds: float = 0.6) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def settle(app, win) -> None:
    """Deja que termine la vista previa en curso (se pinta en segundo plano)."""
    wait(app, 1.6)


def shot(app, widget, name: str) -> None:
    wait(app, 0.3)
    pixmap = widget.grab()
    OUT.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(OUT / f"{name}.png"), "PNG")
    print(f"  {name}.png  {pixmap.width()}×{pixmap.height()}")


def only_section(win, step: int, key: str) -> None:
    """Deja abierta una sola sección del panel de ajustes de un paso."""
    for sec in win.sections_by_step[step]:
        sec.set_expanded(sec.key == key, silent=True)


def main() -> None:
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(TMP / "settings"))
    app = create_app([])
    photo, depth = make_scene()
    photo_path = TMP / "sunset_lake.jpg"
    photo.save(photo_path, quality=93)

    win = MainWindow()
    win.ask_before_closing = False
    win.resize(*WINDOW)
    win.show()
    wait(app, 1.0)
    win.add_photos([photo_path])
    wait(app, 0.8)
    path = str(photo_path)
    win.depth_maps[path] = depth                                  # el mapa de ejemplo entra como un mapa «externo»
    win.depth_sources[path] = "external"
    win._sync_editor()
    wait(app, 0.8)

    print("Importar")
    win.set_step(STEP_IMPORT)
    shot(app, win, "01-import")

    print("Profundidad")
    win.set_step(STEP_DEPTH)
    win.view_switch.set_current_index(VIEW_DEPTH)
    settle(app, win)
    shot(app, win, "02-depth")

    print("Dar forma al efecto")
    win.set_step(STEP_EDIT)
    editor = win.editor
    editor.apply_style(presets.builtin("style", "photo_color"))
    editor.ramp_editor.preset.chosen.emit(presets.builtin("ramp", "sunset"))
    editor.base_panel.base.set_current_index(0)
    editor.bands_panel.add_item(Band(position=0.30, width=0.07, softness=0.06, color=(1.0, 0.9, 0.7), strength=0.9))
    editor.areas_panel.add_item(Area(name="Sun glow", center=(0.5, 0.15, 0.7), scale=(0.7, 0.45, 0.5), strength=0.8,
                                     edge_softness=0.9, falloff=0.6))
    editor.areas_panel.add_item(Area(name="Lake light", shape="box", center=(-0.2, -0.45, 0.35), scale=(0.9, 0.18, 0.3),
                                     strength=0.45, edge_softness=0.7))
    win.view_switch.set_current_index(VIEW_RESULT)
    only_section(win, STEP_EDIT, "ramp")
    settle(app, win)
    shot(app, win, "03-shape-ramp")
    only_section(win, STEP_EDIT, "bands")
    settle(app, win)
    shot(app, win, "04-shape-bands")
    only_section(win, STEP_EDIT, "areas")
    editor.areas_panel.select(0)
    settle(app, win)
    shot(app, win, "05-shape-areas")
    only_section(win, STEP_EDIT, "mix")
    settle(app, win)
    shot(app, win, "06-blend-masks")
    only_section(win, STEP_EDIT, "lens")
    editor.lens_panel.preset_bar.chosen.emit(presets.builtin("lens", "portrait"))
    settle(app, win)
    shot(app, win, "07-lens-blur")

    print("Vista 3D y comparar")
    only_section(win, STEP_EDIT, "bands")
    editor.bands_panel.select(0)
    win.toggle_3d()
    settle(app, win)
    win.view3d.camera.set_view(-32, 18)
    wait(app, 0.8)
    shot(app, win, "08-view-3d")
    win.toggle_3d()
    win.view_switch.set_current_index(VIEW_COMPARE)
    win._show_current()
    settle(app, win)
    shot(app, win, "09-compare")
    win.view_switch.set_current_index(VIEW_RESULT)

    print("Look")
    win.set_step(STEP_LOOK)
    editor.apply_look_preset(presets.builtin("look", "cinema_warm"))
    settle(app, win)
    shot(app, win, "10-look")

    print("Exportar")
    win.set_step(STEP_EXPORT)
    win.export_panel.folder.setText(str(Path.home() / "Pictures" / "Deepshot"))          # una carpeta neutra para la captura
    settle(app, win)
    shot(app, win, "11-export")

    print("Gestor de modelos y módulos")
    from deepshot.ui.model_manager import ModelManagerDialog

    dialog = ModelManagerDialog(win.models, win.estimator, win.active_model(), lambda: False, win)
    dialog.resize(1100, 620)
    dialog.show()
    dialog.folder_label.setText(dialog.tr("Carpeta de modelos: {0}").format("~/deepshot-studio/models"))   # sin rutas temporales
    wait(app, 0.8)
    shot(app, dialog, "12-model-manager")
    dialog.close()

    win.editor.shutdown()
    win.close()
    print(f"Listo: {OUT}")


if __name__ == "__main__":
    main()
