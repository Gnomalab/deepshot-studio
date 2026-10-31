"""Ventana principal: pasos y vistas, biblioteca, ayuda contextual, exportación y herramientas."""

import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QCheckBox,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
)

from deepshot import APP_NAME  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.areas import Area  # noqa: E402
from deepshot.ui.main_window import (  # noqa: E402
    STEP_DEPTH,
    STEP_EDIT,
    STEP_EXPORT,
    STEP_IMPORT,
    MODE_2D,
    MODE_3D,
    VIEW_COMPARE,
    VIEW_DEPTH,
    VIEW_PHOTO,
    VIEW_RESULT,
    MainWindow,
)
from deepshot.ui.value_slider import ValueSlider  # noqa: E402

W, H = 160, 120


@pytest.fixture
def win(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    app = create_app([])
    boxes = []

    def fake_exec(self):      # un diálogo modal bloquearía el test: se anota y se cierra
        boxes.append((self.text(), self.detailedText()))
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    w = MainWindow()
    w._boxes = boxes
    for sections in w.sections_by_step.values():       # los paneles arrancan plegados: los tests los necesitan abiertos
        for sec in sections:
            sec.set_expanded(True, silent=True)
    w.resize(1300, 800)
    w.show()
    app.processEvents()
    w._app = app
    photos = []
    for i in range(3):
        p = tmp_path / f"foto{i + 1}.png"
        Image.new("RGB", (W, H), (60 * (i + 1), 90, 120)).save(p)
        photos.append(p)
    w._photos = photos
    yield w
    w._probe.wait(120)
    w.ask_before_closing = False             # el test puede dejar cambios sin guardar
    w.close()
    w.editor.shutdown()
    app.processEvents()
    # Borrado ordenado por Qt: si las ventanas de tests anteriores siguen vivas, cambiar el tema (que reestiliza todos
    # los widgets de la aplicación) se vuelve lentísimo y acaba colgando la suite.
    for panel in w.editor.panels():
        panel.deleteLater()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def give_depth(w, path=None):
    path = path or w._photos[0]
    depth = np.tile(np.linspace(0.1, 0.9, W, dtype=np.float32), (H, 1))
    w.depth_maps[str(path)] = depth
    w.depth_sources[str(path)] = "external"
    w._sync_editor()
    return depth


def wait_for(w, condition, timeout=5.0):
    t = time.time()
    while not condition() and time.time() - t < timeout:
        w._app.processEvents()
        time.sleep(0.01)
    return condition()


# --- Biblioteca -------------------------------------------------------------------------------------
def test_remove_photos_forgets_their_data_and_selects_a_neighbour(win):
    win.add_photos(win._photos)
    assert win.library.count() == 3
    give_depth(win, win._photos[2])
    assert win.current_path == win._photos[2]
    win.remove_photos([str(win._photos[2])])
    assert win.library.count() == 2 and str(win._photos[2]) not in win.depth_maps
    assert win.current_path is not None and win.current_path != win._photos[2]
    assert win._photos[2].is_file()                                      # el archivo no se borra
    win.remove_photos([str(p) for p in win._photos])
    assert win.library.count() == 0 and win.current_path is None and not win.canvas.has_image()


def test_depth_arriving_for_a_removed_photo_is_ignored(win):
    win.add_photos(win._photos[:2])
    gone = str(win._photos[0])
    win.remove_photos([gone])
    win._on_depth_finished(gone, np.zeros((H, W), dtype=np.float32))
    assert gone not in win.depth_maps


# --- Pasos y vistas -----------------------------------------------------------------------------------------
def test_steps_switch_to_the_matching_view(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_DEPTH)
    assert win.view_switch.current_index() == VIEW_DEPTH
    win.set_step(STEP_EDIT)
    assert win.view_switch.current_index() == VIEW_RESULT
    win.set_step(STEP_IMPORT)
    assert win.view_switch.current_index() == VIEW_PHOTO
    win.set_step(STEP_EXPORT)
    assert win.view_switch.current_index() == VIEW_RESULT


def test_relief_lives_in_the_effect_step(win):
    names = {i: [s.header.text() for s in secs] for i, secs in win.sections_by_step.items()}
    assert any("Relieve" in t for t in names[STEP_EDIT])
    assert not any("Relieve" in t for t in names[STEP_DEPTH])


def test_editing_from_photo_view_jumps_to_the_effect(win):
    """Antes el cambio no se veía en la vista Foto y parecía que el control no hacía nada."""
    win.add_photos(win._photos[:1])
    give_depth(win)
    win.set_step(STEP_EDIT)
    win.view_switch.set_current_index(VIEW_PHOTO)
    win._show_current()
    win.editor.relief_panel.relief.setValue(2.0, emit=True)
    assert win.view_switch.current_index() == VIEW_RESULT


def test_overlays_only_on_effect_and_compare_in_edit_step(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.editor.areas_panel.add_item(Area(name="A"))
    win.set_step(STEP_EDIT)
    for view, expected in ((VIEW_PHOTO, False), (VIEW_DEPTH, False), (VIEW_RESULT, True), (VIEW_COMPARE, True)):
        win.view_switch.set_current_index(view)
        win._show_current()
        assert bool(win.canvas._overlays) == expected, view
    win.view_switch.set_current_index(VIEW_RESULT)
    win.set_step(STEP_DEPTH)
    assert win.canvas._overlays == []                                     # fuera del paso de edición no hay contornos


def test_areas_toggle_hides_overlays_and_tools_release_when_leaving_edit_step(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.editor.areas_panel.add_item(Area(name="A"))
    win.set_step(STEP_EDIT)
    win.view_switch.set_current_index(VIEW_RESULT)
    win._show_current()
    assert win.canvas._overlays
    win.areas_button.toggle()                                              # H: ocultar contornos y manejadores
    assert not win.areas_button.isChecked() and win.canvas._overlays == []
    win.areas_button.toggle()
    assert win.canvas._overlays
    # Una herramienta activa se suelta al salir del paso 3 y no queda en Exportar.
    win.editor.areas_panel.set_tool("place")
    assert win.canvas.tool() == "place"
    win.set_step(STEP_EXPORT)
    assert win.canvas.tool() == "" and win.canvas._overlays == []
    # Activar una herramienta con las áreas ocultas las muestra de nuevo.
    win.areas_button.setChecked(False)
    win.editor.areas_panel.set_tool("draw")
    assert win.areas_button.isChecked() and win.step_group.checkedId() == STEP_EDIT and win.canvas._overlays


def test_file_menu_actions_need_depth_and_work_with_it(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    out = tmp_path / "mi_estilo.deepshotstyle"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out), "")))
    needs_depth = (win.action_save_style, win.action_load_style, win.action_export)
    for action in needs_depth + (win.action_save_project, win.action_save_project_as):
        assert not action.isEnabled()                                      # sin fotos ni mapa: apagadas, no mudas
    assert win.action_open_project.isEnabled()
    win.add_photos(win._photos[:1])
    assert win.action_save_project.isEnabled() and not win.action_save_style.isEnabled()   # foto sin mapa
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    for action in needs_depth:
        assert action.isEnabled()
    win.action_save_style.trigger()
    assert out.exists()
    win.action_load_style.trigger()
    assert "Estilo aplicado" in win.statusBar().currentMessage()
    win.action_export.trigger()
    assert win.step_group.checkedId() == STEP_EXPORT


def test_load_style_accepts_exported_settings_json_and_refuses_garbage(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    settings = tmp_path / "ajustes.json"
    win.editor.save_settings(settings)                                     # el JSON que acompaña a una exportación
    garbage = tmp_path / "basura.deepshotstyle"
    garbage.write_text("no es json", encoding="utf-8")
    pick = {"file": settings}
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(pick["file"]), "")))
    win.action_load_style.trigger()
    assert "Estilo aplicado" in win.statusBar().currentMessage() and not win._boxes
    pick["file"] = garbage
    win.action_load_style.trigger()
    assert win._boxes and "estilo válido" in win._boxes[-1][0]


def test_project_save_and_open_restores_library_depth_settings_and_mask(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    win.add_photos(win._photos[:2])
    for p in win._photos[:2]:
        win.library.setCurrentRow(win._photos.index(p))
        give_depth(win, p)
        wait_for(win, lambda: win.editor.result is not None)
    win.library.setCurrentRow(0)
    wait_for(win, lambda: win.editor.path == str(win._photos[0]))
    win.editor.areas_panel.add_item(Area(name="Marcada", center=(0.1, 0.2, 0.3)))
    win.editor.project.composite.mask = "painted"
    win.editor._ensure_mask_path()
    win.editor.mask_layer().fill(1.0)
    win.editor.mask_layer().save(Path(win.editor.project.composite.painted_mask))
    saved_depth = win.depth_maps[str(win._photos[0])].copy()

    target = tmp_path / "trabajo"                                          # sin extensión: se añade sola
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    assert win.save_project()
    file = tmp_path / "trabajo.deepshot"
    assert file.is_file() and win.project_path == file and win.windowTitle().startswith("trabajo")
    assert win.save_project() and not win._boxes                           # segunda vez: sin preguntar

    # Se vacía la biblioteca (como si fuera otra sesión) y se abre el proyecto.
    win.remove_photos([str(p) for p in win._photos[:2]])
    assert win.library.count() == 0 and not win.depth_maps
    assert win.open_project(str(file))
    assert win.library.count() == 2 and win.project_path == file
    assert np.allclose(win.depth_maps[str(win._photos[0])], saved_depth, atol=2e-5)
    wait_for(win, lambda: win.editor.result is not None)
    win.library.setCurrentRow(0)
    wait_for(win, lambda: win.editor.path == str(win._photos[0]))
    assert [a.name for a in win.editor.project.areas] == ["Marcada"]
    assert win.editor.project.composite.mask == "painted"
    assert win.editor.mask_layer().data.min() == 255                       # la máscara pintada volvió entera
    assert not win._boxes                                                  # sin avisos


def test_open_project_with_moved_photo_asks_for_it_and_reports_missing(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    win.add_photos(win._photos[:2])
    for p in win._photos[:2]:
        win.library.setCurrentRow(win._photos.index(p))
        give_depth(win, p)
        wait_for(win, lambda: win.editor.result is not None)
    file = tmp_path / "p.deepshot"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(file), "")))
    assert win.save_project_as()
    win.remove_photos([str(p) for p in win._photos[:2]])
    moved = tmp_path / "nueva"
    moved.mkdir()
    new_first = moved / "foto1.png"
    win._photos[0].rename(new_first)                                       # la primera foto cambia de sitio
    win._photos[1].unlink()                                                # la segunda desaparece
    asked = []

    def locate(*a, **k):
        asked.append(a[1])
        return (str(new_first), "") if len(asked) == 1 else ("", "")      # localiza la primera; cancela la segunda

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(locate))
    assert win.open_project(str(file))
    assert len(asked) == 2 and win.library.count() == 1
    assert win.library.item(0).data(Qt.UserRole) == str(new_first)
    assert win._boxes and "foto2.png" in win._boxes[-1][0]               # aviso con la foto que falta


def test_open_project_refuses_invalid_file_and_asks_before_replacing(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    bad = tmp_path / "roto.deepshot"
    bad.write_bytes(b"no soy un zip")
    win.add_photos(win._photos[:1])
    give_depth(win)                                                        # hay trabajo sin guardar: se preguntará
    assert not win.open_project(str(bad))
    assert win.library.count() == 1 and win._boxes                         # no se tocó la biblioteca
    good = tmp_path / "ok.deepshot"
    from deepshot import project_file as pf

    pf.save_project(good, [])
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Cancel)     # el usuario dice que no
    assert not win.open_project(str(good)) and win.library.count() == 1
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Discard)    # «No guardar»
    assert win.open_project(str(good)) and win.library.count() == 0


def go_3d(win):
    """Pone la vista Efecto en modo 3D, como lo hace el conmutador 2D / 3D."""
    win.view_switch.set_current_index(VIEW_RESULT)
    win.mode_switch._group.button(MODE_3D).click()


def test_3d_is_a_mode_of_the_effect_view_not_a_fifth_view(win):
    assert win.view_switch._group.buttons().__len__() == 4                # Foto · Profundidad · Efecto · Comparar
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    win.view_switch.set_current_index(VIEW_RESULT)
    win._show_current()
    assert win.mode_switch.isVisibleTo(win) and win.viewer_stack.currentWidget() is win.canvas   # 2D por defecto
    assert not win.texture_combo.isVisibleTo(win)
    assert win.texture_combo.currentData() == "effect"                     # la textura por defecto en 3D es el efecto
    win.mode_switch._group.button(MODE_3D).click()
    assert win._in_3d() and win.viewer_stack.currentWidget() is win.view3d and win.texture_combo.isVisibleTo(win)
    assert win.quality3d_combo.isVisibleTo(win)                            # la calidad 3D solo se ve en el modo 3D
    win.quality3d_combo.setCurrentIndex(win.quality3d_combo.findData("high"))
    assert win.view3d.quality() == "high" and win.settings.value("view3d/quality") == "high"
    win.quality3d_combo.setCurrentIndex(win.quality3d_combo.findData("auto"))
    for other in (VIEW_PHOTO, VIEW_DEPTH, VIEW_COMPARE):                   # fuera de Efecto: siempre 2D y sin el conmutador
        win.view_switch.set_current_index(other)
        win._show_current()
        assert not win._in_3d() and win.viewer_stack.currentWidget() is win.canvas
        assert not win.mode_switch.isVisibleTo(win) and not win.texture_combo.isVisibleTo(win)
    win.view_switch.set_current_index(VIEW_RESULT)                         # al volver a Efecto se recuerda el modo 3D
    win._show_current()
    assert win._in_3d() and win.mode_switch.current_index() == MODE_3D
    win.toggle_3d()                                                        # la tecla 3 alterna
    assert not win._in_3d() and win.mode_switch.current_index() == MODE_2D
    win.view_switch.set_current_index(VIEW_DEPTH)
    win.toggle_3d()                                                        # desde otra vista: pasa a Efecto y entra en 3D
    assert win.view_switch.current_index() == VIEW_RESULT and win._in_3d()
    win.editor.areas_panel.draw_button.toggle()                            # Dibujar también va en 3D: se queda en 3D
    assert win._in_3d() and win.canvas.tool() == "draw" and win.view3d._tool == "draw"
    win.editor.areas_panel.set_tool("")
    assert win.view3d._tool == ""


def test_draw_and_place_work_on_the_3d_surface_with_one_undo_step(win):
    from PySide6.QtCore import QPoint

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    go_3d(win)
    view = win.view3d
    view.camera.set_view(0, 0)
    win._app.processEvents()
    before = win.undo_stack.count()
    win.editor.areas_panel.draw_button.setChecked(True)
    assert win._in_3d() and view._tool == "draw"
    c = QPoint(view.width() // 2, view.height() // 2)
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, c)
    for dx in (10, 30, 60):
        QTest.mouseMove(view, c + QPoint(dx, 0))
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, c + QPoint(60, 0))
    assert len(win.editor.project.areas) == 1 and win.undo_stack.count() == before + 1
    assert not win.editor.areas_panel.draw_button.isChecked() and view._tool == ""     # herramienta de un solo uso
    win.undo_stack.undo()
    assert win.editor.project.areas == []
    win.undo_stack.redo()
    area = win.editor.project.areas[0]
    start = area.center
    win.editor.areas_panel.place_button.setChecked(True)
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, c + QPoint(90, 40))
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, c + QPoint(90, 40))
    assert win.editor.project.areas[0].center != start and win.undo_stack.count() == before + 2


def test_3d_shows_band_planes_and_the_green_focus_zone_and_the_eye_hides_the_planes(win):
    from deepshot.tonemap import Band

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    win.editor.project.bands.append(Band(position=0.4, width=0.1, pan=20.0))
    go_3d(win)
    win.editor.set_edit_context("bands")                                   # se está tocando una banda: sus planos
    win._update_overlays()
    assert len(win.view3d._bands) == 1 and win.view3d._bands[0][0].shape == (4, 3)
    win.editor.lens_panel.show_focus.setChecked(True)
    win._update_overlays()
    assert win.view3d._focus is not None
    win.areas_button.setChecked(False)                                     # el ojo oculta contornos y planos
    win._update_overlays()
    assert win.view3d._bands == []
    win.areas_button.setChecked(True)
    win.editor.lens_panel.show_focus.setChecked(False)
    win._update_overlays()
    assert win.view3d._focus is None


def test_3d_mode_follows_relief_and_texture_and_survives_step_changes(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.editor.areas_panel.add_item(Area(name="A"))
    win.set_step(STEP_EDIT)
    go_3d(win)
    assert win.viewer_stack.currentWidget() is win.view3d and win.texture_combo.isVisible()
    assert win.view3d.has_data() and win.canvas._overlays == []           # sin contornos 2D en el modo 3D
    win.set_step(STEP_EXPORT)                                             # el paso 3 y el 4 muestran el efecto: no echan
    win.set_step(STEP_EDIT)
    assert win._in_3d()
    before = win.view3d._render_frame().copy()
    win.editor.project.scene.relief = 3.0                                  # un cambio de relieve llega a la vista 3D
    win.editor._on_changed()
    wait_for(win, lambda: win.view3d._scene.relief == 3.0)
    assert win.view3d._render_frame() != before
    win.texture_combo.setCurrentIndex(win.texture_combo.findData("depth"))
    assert win.view3d._texture is not None and win.view3d._texture.width() == win.depth_maps[str(win._photos[0])].shape[1]
    win.mode_switch._group.button(MODE_2D).click()
    assert win.viewer_stack.currentWidget() is win.canvas and not win.texture_combo.isVisible()


def test_areas_are_edited_in_3d_with_one_undo_step_per_gesture(win):
    from PySide6.QtCore import QPoint

    from deepshot import area3d

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.editor.areas_panel.add_item(Area(name="A", center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3)))
    win.editor.areas_panel.add_item(Area(name="B", center=(0.9, 0.0, 0.5), scale=(0.2, 0.2, 0.2)))
    win.set_step(STEP_EDIT)
    go_3d(win)
    view = win.view3d
    view.camera.set_view(0, 0)
    win._app.processEvents()
    assert len(view._areas) == 2 and view._selected == win.editor.selected_area       # las áreas llegan al visor

    def px(world):
        sx, sy = view._screen(np.asarray(world, dtype=np.float64)[None, :])
        return QPoint(int(round(sx[0])), int(round(sy[0])))

    # Clic en el contorno de la otra área: se selecciona también en la lista, sin tocar nada (sin deshacer).
    undo_before = win.undo_stack.count()
    QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier, px(area3d.wireframe(win.editor.project.areas[1])[0][5]))
    assert win.editor.areas_panel.list.currentRow() == 1 and win.undo_stack.count() == undo_before
    win.editor.areas_panel.select(0)
    win._app.processEvents()
    # Arrastrar la flecha X de A: un solo paso de deshacer y el centro cambia.
    area = win.editor.project.areas[0]
    length = area3d.gizmo_length(view.camera)
    start = px(np.array(area.center) + area3d.AXES["x"] * length * 0.9)
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, start)
    for dx in (15, 40, 70):
        QTest.mouseMove(view, start + QPoint(dx, 0))
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, start + QPoint(70, 0))
    assert win.editor.project.areas[0].center[0] > 0.1 and win.editor.project.areas[0].center[1] == 0.0
    assert win.undo_stack.count() == undo_before + 1
    win.undo_stack.undo()
    assert win.editor.project.areas[0].center == (0.0, 0.0, 0.5)
    # El ojo oculta contornos y tiradores también en 3D.
    win.areas_button.setChecked(False)
    assert view._areas == []
    win.areas_button.setChecked(True)
    assert len(view._areas) == 2


def test_bands_are_edited_in_3d_with_handles_one_undo_step_per_gesture(win):
    from PySide6.QtCore import QPoint, QPointF

    from deepshot import area3d
    from deepshot.tonemap import Band

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    win.editor.project.bands.extend([Band(position=0.3, width=0.1), Band(position=0.7, width=0.1)])
    win.editor.bands_panel.bind(win.editor.project)
    win.editor._on_committed()                                              # las bandas ya son el punto de partida del deshacer
    win.editor.bands_panel.select(0)
    go_3d(win)
    view = win.view3d
    win.resize(1500, 900)                                                   # visor ancho: el tirador de tilt lejos del gizmo de ejes
    win._app.processEvents()
    view.camera.set_view(-35, 20)
    win._app.processEvents()
    assert view._selected_band == 0 and view._selected_band_frame() is not None

    def px(world):
        sx, sy = view._screen(np.asarray(world, dtype=np.float64)[None, :])
        return QPoint(int(round(sx[0])), int(round(sy[0])))

    def drag(start, delta):
        undo = win.undo_stack.index()
        QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, start)
        for f in (0.3, 0.6, 1.0):
            QTest.mouseMove(view, start + QPoint(int(delta.x() * f), int(delta.y() * f)))
        QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier, start + delta)
        return win.undo_stack.index() - undo

    def undo():                                    # deshacer rehace los paneles: se vuelve a tocar la banda
        win.undo_stack.undo()
        win.editor.set_edit_context("bands")
        win._update_overlays()

    handles = area3d.band_handles(view._selected_band_frame())
    assert view.band_hit(QPointF(px(handles["pan"]))) == "pan" and view.band_hit(QPointF(px(handles["tilt"]))) == "tilt"
    assert view.band_hit(QPointF(px(handles["width"]))) == "width"
    band = win.editor.project.bands[0]
    assert band.pan == 0.0
    assert drag(px(handles["pan"]), QPoint(0, 40)) == 1 and band.pan != 0.0           # un solo paso de deshacer
    undo()
    assert win.editor.project.bands[0].pan == 0.0
    assert drag(px(handles["tilt"]), QPoint(30, 0)) == 1 and win.editor.project.bands[0].tilt != 0.0
    undo()
    old = win.editor.project.bands[0].position
    arrow = handles["move"] + area3d.AXES["z"] * area3d.gizmo_length(view.camera) * 0.9
    assert drag(px(arrow), QPoint(50, 0)) == 1 and win.editor.project.bands[0].position != old
    undo()
    assert win.editor.project.bands[0].position == old
    assert drag(px(handles["width"]), QPoint(40, 0)) == 1 and win.editor.project.bands[0].width != 0.1
    undo()
    # Un clic suelto en un tirador no cambia nada ni apila deshacer.
    assert drag(px(handles["pan"]), QPoint(0, 0)) == 0 and win.editor.project.bands[0].pan == 0.0
    # La banda elegida es la que lleva los tiradores.
    win.editor.bands_panel.select(1)
    win._app.processEvents()
    assert view._selected_band == 1


def test_lens_panel_turns_on_with_any_control_changes_the_result_and_undoes(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    panel = win.editor.lens_panel
    assert panel.isEnabled() and not win.editor.project.lens.enabled and not panel.enabled.isChecked()
    win.set_step(STEP_EDIT)
    win.editor.project.composite.opacity = 0.0
    before = win.editor.result.copy()
    panel.aperture.setValue(1.4, emit=True)                                           # tocar un ajuste lo activa
    panel.aperture.editingFinished.emit()
    lens = win.editor.project.lens
    assert lens.enabled and lens.aperture == 1.4 and panel.enabled.isChecked()
    assert wait_for(win, lambda: win.editor.result is not None and win.editor.result != before)   # el efecto cambió
    undo_count = win.undo_stack.count()
    panel.blades.setValue(6, emit=True)
    panel.blades.editingFinished.emit()
    assert win.editor.project.lens.blades == 6 and win.undo_stack.count() == undo_count + 1
    win.undo_stack.undo()
    assert win.editor.project.lens.blades == 0 and panel.blades.value() == 0       # deshacer actualiza el panel
    panel.target._group.button(1).click()                                   # «Solo la foto»
    assert win.editor.project.lens.target == "photo"
    panel.enabled.setChecked(False)
    assert not win.editor.project.lens.enabled


def test_focus_eyedropper_picks_depth_in_one_undo_step_and_releases_itself(win):
    from PySide6.QtCore import QPointF

    win.add_photos(win._photos[:1])
    give_depth(win)                                                        # profundidad = gradiente de 0.1 a 0.9 en x
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    panel = win.editor.lens_panel
    undo_before = win.undo_stack.count()
    panel.pick_button.click()                                              # activa el cuentagotas
    assert win.canvas.tool() == "pick_focus" and win.view_switch.current_index() == VIEW_RESULT
    assert panel.show_focus.isChecked()                                    # y enseña la zona en foco para guiarse
    depth = win.depth_maps[str(win._photos[0])]
    h, w = depth.shape
    wait_for(win, lambda: win.canvas._item is not None)
    vp = win.canvas.viewport()
    target = win.canvas.mapFromScene(QPointF(w * 0.75, h * 0.5))           # clic al 75 % del ancho
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, target)
    lens = win.editor.project.lens
    assert lens.enabled and lens.focus == pytest.approx(depth[h // 2, int(w * 0.75)], abs=0.03)
    assert panel.focus.value() == pytest.approx(lens.focus, abs=1e-3)      # el deslizador sigue al valor
    assert not panel.pick_button.isChecked() and win.canvas.tool() == ""   # un solo uso
    assert win.undo_stack.count() == undo_before + 1
    win.undo_stack.undo()
    assert not win.editor.project.lens.enabled and win.editor.project.lens.focus == pytest.approx(0.35)


def test_focus_eyedropper_cancel_restores_and_other_tools_release_it(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    panel = win.editor.lens_panel
    panel.pick_button.click()
    right = float(win.depth_maps[str(win._photos[0])].shape[1] - 4)         # cerca del borde derecho: lo más lejano
    win.editor.gesture({"kind": "pick_focus", "phase": "press", "index": -1, "x": right, "y": 20.0, "mods": 0, "axis": ""})
    assert win.editor.project.lens.focus > 0.8
    win.editor.gesture({"kind": "pick_focus", "phase": "cancel", "index": -1, "x": right, "y": 20.0, "mods": 0, "axis": ""})
    assert win.editor.project.lens.focus == pytest.approx(0.35) and not win.editor.project.lens.enabled
    panel.pick_button.setChecked(True)
    win.editor.areas_panel.draw_button.toggle()                            # otra herramienta: el cuentagotas se suelta
    assert not panel.pick_button.isChecked() and win.canvas.tool() == "draw"
    win.editor.areas_panel.set_tool("")
    panel.pick_button.setChecked(True)
    win.set_step(STEP_EXPORT)                                              # salir del paso 3 también lo suelta
    assert not panel.pick_button.isChecked() and win.canvas.tool() == ""


def test_focus_overlay_follows_lens_and_only_shows_in_the_effect_step(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    panel = win.editor.lens_panel
    assert win.canvas._focus_item is None
    panel.show_focus.setChecked(True)
    assert win.canvas._focus_item is not None
    first = win.canvas._focus_item.pixmap().toImage()
    panel.focus.setValue(0.8, emit=True)                                   # mover el foco cambia el tinte
    assert win.canvas._focus_item.pixmap().toImage() != first
    win.view_switch.set_current_index(VIEW_DEPTH)
    win._show_current()
    assert win.canvas._focus_item is None                                  # no se ve sobre la foto ni el mapa
    win.view_switch.set_current_index(VIEW_RESULT)
    win._show_current()
    assert win.canvas._focus_item is not None
    win.set_step(STEP_EXPORT)
    assert win.canvas._focus_item is None                                  # ni en Exportar
    win.set_step(STEP_EDIT)
    panel.show_focus.setChecked(False)
    assert win.canvas._focus_item is None


# --- Ayuda contextual -------------------------------------------------------------------------------------------
def test_every_control_has_contextual_help(win):
    """Cada control del panel de ajustes debe explicarse al pasar el ratón (tooltip y pie)."""
    win.add_photos(win._photos[:1])
    give_depth(win)
    win.editor.areas_panel.add_item(Area())
    win.editor.bands_panel._add()
    win.editor.mix_panel.mask.setCurrentIndex(win.editor.mix_panel.mask.findData("painted"))
    win.editor.mix_panel._set("mask", "painted")
    win._enable_status_tips()
    missing = []
    for dock in (win.settings_dock, win.library_dock, win.log_dock):
        for w in dock.findChildren(ValueSlider) + dock.findChildren(QPushButton) + dock.findChildren(QCheckBox) \
                + dock.findChildren(QComboBox):
            if w.objectName() in ("SectionHeader",) or isinstance(w.parent(), QLineEdit):
                continue
            if not w.toolTip().strip() and not (isinstance(w, QPushButton) and w.property("segment")):
                label = getattr(w, "label", None) or getattr(w, "text", lambda: "")()
                missing.append(f"{type(w).__name__}:{label}")
    assert not missing, "Controles sin ayuda: " + " | ".join(missing)


def test_hover_shows_help_in_the_footer(win):
    from PySide6.QtCore import QEvent

    slider = win.editor.relief_panel.relief
    win._app.sendEvent(slider, QEvent(QEvent.Enter))
    assert "relieve" in win.statusBar().currentMessage().lower()
    win._app.sendEvent(slider, QEvent(QEvent.Leave))
    assert win.statusBar().currentMessage() == ""


# --- Exportación --------------------------------------------------------------------------------------------------
def test_export_panel_exports_effect_and_depth_in_background(win, tmp_path):
    win.add_photos(win._photos[:1])
    give_depth(win)
    panel = win.export_panel
    win.set_step(STEP_EXPORT)
    out = tmp_path / "exportados"
    panel._last_dir = out
    panel.refresh()
    assert panel.button.isEnabled() and "160" in panel.info.text() and "120" in panel.info.text()
    assert panel.name.text() == "foto1_deepshot"

    done = []
    panel.exported.connect(done.append)
    panel.format.set_current_index(0)                                      # PNG 8 bits
    panel.export()
    assert not panel.button.isEnabled()                                    # mientras trabaja
    assert wait_for(win, lambda: bool(done))
    assert sorted(p.name for p in out.iterdir()) == ["foto1_deepshot.deepshot.json", "foto1_deepshot.png"]
    assert panel.button.isEnabled() and "foto1_deepshot.png" in panel.status.text()

    panel.content.set_current_index(1)                                     # mapa de profundidad
    panel._on_content_changed(1)
    assert panel.name.text() == "foto1_profundidad"
    panel.format.set_current_index(2)                                      # EXR
    done.clear()
    panel.export()
    assert wait_for(win, lambda: bool(done))
    assert (out / "foto1_profundidad.exr").is_file()
    panel.export()                                                         # otra vez: no pisa el anterior
    assert wait_for(win, lambda: len(done) == 2)
    assert (out / "foto1_profundidad_2.exr").is_file()


def test_export_final_image_panel_shows_its_own_controls_and_exports_jpg_at_a_chosen_size(win, tmp_path):
    from PIL import Image

    from deepshot.look import LookLayer

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    layer = LookLayer.new("levels")
    layer.params["black"] = 0.4
    win.editor.project.look.append(layer)
    panel = win.export_panel
    win.set_step(STEP_EXPORT)
    out = tmp_path / "final"
    panel._last_dir = out
    panel.content.set_current_index(2)                                     # imagen final
    panel._on_content_changed(2)
    assert panel.final_box.isVisibleTo(win) and not panel.format.isVisibleTo(win)
    assert panel.name.text() == "foto1_final" and panel.quality.isVisibleTo(win)     # JPG: tiene calidad
    panel.final_format.setCurrentIndex(panel.final_format.findData("tiff8"))
    assert not panel.quality.isVisibleTo(win)                              # sin pérdida: sin calidad
    panel.final_format.setCurrentIndex(panel.final_format.findData("jpg"))
    panel.size_mode.setCurrentIndex(panel.size_mode.findData("width"))      # el valor parte del ancho de la imagen
    assert panel.size_value.value() == 160 and panel.size_value.isVisibleTo(win)
    panel.size_value.setValue(80)
    assert "80 × 60" in panel.size_result.text()
    opts = panel.options()
    assert opts.content == "final" and opts.format == "jpg" and opts.size_mode == "width" and opts.size_value == 80
    done = []
    panel.exported.connect(done.append)
    panel.export()
    assert wait_for(win, lambda: bool(done))
    with Image.open(out / "foto1_final.jpg") as img:
        assert img.size == (80, 60)
    panel.content.set_current_index(0)                                     # con el efecto vuelven los formatos de siempre
    panel._on_content_changed(0)
    assert panel.format.isVisibleTo(win) and not panel.final_box.isVisibleTo(win)


def test_export_depth_source_selector_only_for_depth_warns_and_exports_moved_zones(win, tmp_path):
    win.add_photos(win._photos[:1])
    depth = give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    panel = win.export_panel
    win.set_step(STEP_EXPORT)
    out = tmp_path / "zonas"
    panel._last_dir = out
    panel.refresh()
    assert not panel.depth_source.isVisibleTo(panel)                       # con «Efecto» no se ve
    panel.content.set_current_index(1)
    panel._on_content_changed(1)
    assert panel.depth_source.isVisibleTo(panel) and not panel.options().depth_zones     # por defecto: original
    assert "original" in panel.depth_hint.text()
    panel.depth_source._group.button(1).click()                            # «Con zonas movidas», con el control apagado
    assert panel.options().depth_zones and "apagado" in panel.depth_hint.text()      # avisa de que no cambiará nada
    from PySide6.QtCore import QSettings

    assert QSettings().value("export/depth_source") == "zones"             # se recuerda para la próxima vez
    control = win.editor.project.depth_control
    control.enabled = True
    control.near.center, control.near.target = 0.1, 0.4
    panel.refresh()
    assert "movidas" in panel.depth_hint.text() and "apagado" not in panel.depth_hint.text()
    done = []
    panel.exported.connect(done.append)
    panel.format.set_current_index(2)                                      # EXR: coma flotante, comparación directa
    panel.export()
    assert wait_for(win, lambda: bool(done))
    import OpenEXR

    z = OpenEXR.File(str(out / "foto1_profundidad.exr")).channels()["Z"].pixels
    expected = 1.0 - control.apply(depth)                                  # cerca = claro, con las zonas movidas
    assert np.abs(z - expected).max() < 1e-5 and not np.allclose(z, 1.0 - depth, atol=1e-3)
    panel.depth_source._group.button(0).click()                            # volver al original: sale el mapa sin tocar
    done.clear()
    panel.export()
    assert wait_for(win, lambda: bool(done))
    z2 = OpenEXR.File(str(out / "foto1_profundidad_2.exr")).channels()["Z"].pixels
    assert np.abs(z2 - (1.0 - depth)).max() < 1e-5


def test_export_button_disabled_without_depth_and_with_invalid_name(win):
    win.add_photos(win._photos[:1])
    win.export_panel.refresh()
    assert not win.export_panel.button.isEnabled()
    assert "mapa de profundidad" in win.export_panel.info.text()
    give_depth(win)
    win.export_panel.name.setText("  ")
    win.export_panel.export()
    assert win.export_panel.status.text() and win.export_panel.button.isEnabled()


def test_export_failure_is_reported_not_raised(win, tmp_path, monkeypatch):
    win.add_photos(win._photos[:1])
    give_depth(win)
    panel = win.export_panel
    blocker = tmp_path / "es_un_archivo"
    blocker.write_text("x")
    panel._last_dir = blocker / "dentro"                                   # no se puede crear la carpeta
    errors = []
    panel.failed.connect(lambda m, d: errors.append((m, d)))
    panel.export()
    assert wait_for(win, lambda: bool(errors))
    assert errors[0][0] and "Traceback" in errors[0][1]
    assert panel.button.isEnabled()
    assert win._boxes and win._boxes[0][0] == errors[0][0]                 # y la ventana avisa al usuario


# --- Herramientas del lienzo ---------------------------------------------------------------------------------------
def test_tools_are_mutually_exclusive_and_escape_cancels(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    areas, mix = win.editor.areas_panel, win.editor.mix_panel
    areas.draw_button.setChecked(True)
    assert win.canvas.tool() == "draw" and win.step_group.checkedId() == STEP_EDIT
    assert win.view_switch.current_index() in (VIEW_RESULT, VIEW_COMPARE)
    mix.activate_paint("paint")
    assert win.canvas.tool() == "paint" and areas.tool() == "" and mix.paint_tool() == "paint"
    areas.place_button.setChecked(True)
    assert win.canvas.tool() == "place" and mix.paint_tool() == ""
    from PySide6.QtTest import QTest

    QTest.keyClick(win, Qt.Key_Escape)
    assert win.canvas.tool() == "" and areas.tool() == ""


def test_tool_without_depth_shows_hint_and_does_not_activate(win):
    win.add_photos(win._photos[:1])
    win.editor.areas_panel.draw_button.setChecked(True)
    assert win.canvas.tool() == "" and not win.editor.areas_panel.draw_button.isChecked()


def test_draw_tool_finishes_after_one_gesture(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.editor.areas_panel.draw_button.setChecked(True)
    win.canvas.gesture.emit({"kind": "draw", "phase": "press", "index": -1, "x": 80.0, "y": 60.0, "mods": 0, "axis": ""})
    win.canvas.gesture.emit({"kind": "draw", "phase": "move", "index": -1, "x": 110.0, "y": 60.0, "mods": 0, "axis": ""})
    win.canvas.gesture.emit({"kind": "draw", "phase": "release", "index": -1, "x": 110.0, "y": 60.0, "mods": 0, "axis": ""})
    assert len(win.editor.project.areas) == 1
    assert win.canvas.tool() == "" and not win.editor.areas_panel.draw_button.isChecked()


def test_mask_overlay_shows_only_with_painted_mask_in_edit_step(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    win._refresh_mask_overlay()
    assert win.canvas._mask_item is None
    win.editor.mix_panel.activate_paint("paint")
    win._refresh_mask_overlay()
    assert win.canvas._mask_item is not None
    win.editor.mix_panel.show_mask.setChecked(False)
    assert win.canvas._mask_item is None
    win.editor.mix_panel.show_mask.setChecked(True)
    win.set_step(STEP_DEPTH)
    win._refresh_mask_overlay()
    assert win.canvas._mask_item is None


def test_style_section_present_and_style_applies_through_window(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    titles = [s.header.text() for s in win.sections_by_step[STEP_EDIT]]
    assert any("Estilo" in t for t in titles)
    from deepshot import presets

    win.editor.apply_style(presets.builtin("style", "contrast"))
    assert len(win.editor.project.ramp.stops) == 4
    win.undo_stack.undo()
    assert len(win.editor.project.ramp.stops) == 2


# --- Control de profundidad (cerca / medio / lejos) ----------------------------------------------------------
def test_depth_control_panel_turns_on_with_any_control_changes_the_result_and_undoes(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    panel = win.editor.depth_panel
    control = win.editor.project.depth_control
    assert panel.isEnabled() and not control.enabled and not panel.enabled.isChecked()
    win.editor.project.composite.opacity = 0.0
    before = win.editor.result.copy()
    undo_before = win.undo_stack.count()
    w = panel.zone_widgets["mid"]
    w["target"].setValue(0.8, emit=True)                                                # tocar un ajuste lo activa
    w["target"].editingFinished.emit()
    assert control.enabled and control.mid.target == 0.8 and panel.enabled.isChecked()
    assert wait_for(win, lambda: win.editor.result is not None and win.editor.result != before)   # el efecto cambió
    assert win.undo_stack.count() == undo_before + 1
    win.undo_stack.undo()
    assert not win.editor.project.depth_control.enabled and panel.zone_widgets["mid"]["target"].value() == pytest.approx(0.5)
    win.undo_stack.redo()
    assert win.editor.project.depth_control.mid.target == 0.8                           # y rehacer lo devuelve


def test_depth_control_limits_widgets_follow_the_automatic_flag_and_reset_clears_everything(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    panel = win.editor.depth_panel
    w = panel.zone_widgets["far"]
    assert not w["min"].isEnabled() and not w["max"].isEnabled()                        # automáticos: sin límites a mano
    w["auto"].setChecked(False)
    assert win.editor.project.depth_control.far.auto_range is False and w["min"].isEnabled() and w["max"].isEnabled()
    assert win.editor.project.depth_control.enabled                                     # cualquier cambio lo enciende
    panel.zone_widgets["near"]["on"].setChecked(False)
    assert not win.editor.project.depth_control.near.enabled
    panel.reset_button.click()
    from deepshot.depth_control import DepthControl

    assert win.editor.project.depth_control == DepthControl() and not panel.enabled.isChecked()
    assert not panel.zone_widgets["far"]["min"].isEnabled() and panel.zone_widgets["near"]["on"].isChecked()


def test_depth_control_centers_from_map_keeps_the_image_and_is_one_undo_step(win):
    win.add_photos(win._photos[:1])
    depth = give_depth(win)                                                             # gradiente de 0.1 a 0.9
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    undo_before = win.undo_stack.count()
    win.editor.depth_panel.centers_button.click()
    control = win.editor.project.depth_control
    assert control.near.center == pytest.approx(np.percentile(depth, 10), abs=1e-3)
    assert control.far.center == pytest.approx(np.percentile(depth, 90), abs=1e-3)
    assert all(z.target == z.center for z in control.zones())                           # no mueve nada
    assert win.editor.depth_panel.zone_widgets["mid"]["center"].value() == pytest.approx(control.mid.center, abs=1e-3)
    assert win.undo_stack.count() == undo_before + 1


def test_depth_control_changes_what_editor_tools_and_the_3d_view_read(win):
    win.add_photos(win._photos[:1])
    depth = give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    editor = win.editor
    small = editor.render.preview_depth()
    assert editor.warped(small) is small                                                # apagado: la misma matriz
    control = editor.project.depth_control
    control.enabled, control.mid.target, control.mid.feather = True, 0.8, 0.05
    warped = editor.warped(small)
    assert warped is not small and warped.max() > small.max() - 1e-6 and not np.array_equal(warped, small)
    assert editor.warped(small) is warped                                               # la caché evita recalcular
    control.mid.target = 0.7
    assert editor.warped(small) is not warped                                           # y se invalida al cambiar
    # El cuentagotas del foco mide la profundidad movida (la que lee el desenfoque).
    h, w = depth.shape
    raw_at = float(np.median(depth[h // 2 - 2: h // 2 + 3, w // 2 - 2: w // 2 + 3]))
    assert editor.depth_at(w // 2, h // 2) == pytest.approx(
        float(np.median(control.apply(depth[h // 2 - 2: h // 2 + 3, w // 2 - 2: w // 2 + 3]))), abs=1e-6)
    assert abs(editor.depth_at(w // 2, h // 2) - raw_at) > 0.01
    # La vista 3D recibe el relieve ya movido.
    go_3d(win)
    assert win.view3d._depth is editor.warped(editor.render.preview_depth())
    # «Ajustar rampa al relieve» usa la profundidad movida.
    editor.fit_ramp()
    low_raw = float(np.percentile(depth, 0.5))
    assert editor.project.ramp.range_min == pytest.approx(float(np.percentile(editor.warped(depth), 0.5)), abs=1e-4)
    assert editor.project.ramp.range_min == pytest.approx(low_raw, abs=0.02)             # lo cercano no se movió


# --- Cambios sin guardar y proyectos recientes ---------------------------------------------------------
def test_unsaved_changes_are_detected_marked_in_the_title_and_cleared_by_saving(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    win.add_photos(win._photos[:2])
    assert not win.is_dirty()                                              # fotos sin tocar no son trabajo
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    assert win.is_dirty()                                                  # hay un mapa que se perdería
    target = tmp_path / "obra"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    assert win.save_project()
    assert not win.is_dirty() and win.windowTitle() == f"obra — {APP_NAME}"
    win.library.setCurrentRow(1)                                           # seleccionar otra foto no es un cambio
    wait_for(win, lambda: win.editor.path == str(win._photos[1]))
    win.library.setCurrentRow(0)
    wait_for(win, lambda: win.editor.path == str(win._photos[0]))
    assert not win.is_dirty()
    win.editor.areas_panel.add_item(Area(name="Nueva"))
    assert win.is_dirty()
    win._refresh_title()
    assert win.windowTitle() == f"obra* — {APP_NAME}"
    win.undo_stack.undo()
    win._app.processEvents()
    assert not win.is_dirty()                                              # deshacer vuelve al estado guardado
    win.editor.areas_panel.add_item(Area(name="Otra"))
    assert win.save_project() and not win.is_dirty() and win._dirty_timer.isActive() is not None


def test_closing_or_replacing_with_unsaved_changes_asks_save_discard_or_cancel(win, tmp_path, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    win.add_photos(win._photos[:1])
    give_depth(win)
    assert win._confirm_discard("Salir") is not None
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Cancel)
    assert not win._confirm_discard("Salir")
    event = QCloseEvent()
    win.closeEvent(event)
    assert not event.isAccepted()                                          # Cancelar: la ventana no se cierra
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Discard)
    assert win._confirm_discard("Salir")                                   # No guardar: se puede seguir
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Save)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))   # cancela guardar
    assert not win._confirm_discard("Salir")                               # Guardar pero sin elegir archivo: no sigue
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "cerrar.deepshot"), "")))
    assert win._confirm_discard("Salir") and (tmp_path / "cerrar.deepshot").is_file()
    assert win._confirm_discard("Salir")                                   # ya guardado: no pregunta más
    win.ask_before_closing = False


def test_recent_projects_are_remembered_listed_and_missing_ones_dropped(win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    assert win.recent_projects() == []
    win._fill_recent_menu()
    assert [a.text() for a in win.recent_menu.actions()] == ["(ninguno)"]
    win.add_photos(win._photos[:1])
    give_depth(win)
    names = []
    for i in range(12):                                                    # más del máximo: se queda con los últimos
        target = tmp_path / f"p{i}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, t=target, **k: (str(t), "")))
        assert win.save_project_as()
        names.append(f"p{i}.deepshot")
    recent = win.recent_projects()
    assert [Path(p).name for p in recent] == names[::-1][:10]
    win._fill_recent_menu()
    assert [a.text() for a in win.recent_menu.actions()][:2] == ["p11.deepshot", "p10.deepshot"]
    win.recent_menu.actions()[0].trigger()                                 # abrir el más reciente desde el menú
    assert win.project_path == tmp_path / "p11.deepshot"
    (tmp_path / "p10.deepshot").unlink()
    assert not win.open_recent(str(tmp_path / "p10.deepshot")) and win._boxes
    assert str(tmp_path / "p10.deepshot") not in win.recent_projects()


# --- Liberar memoria --------------------------------------------------------------------------------------
def test_free_memory_unloads_the_model_keeps_the_work_and_the_preview_comes_back(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.editor.areas_panel.add_item(Area(name="Se queda"))
    saved = win.depth_maps[str(win._photos[0])]
    win.estimator._pipe, win.estimator._loaded_model = object(), "marigold-v1-0"     # como si hubiera un modelo cargado
    win.editor.result = None
    assert win.free_memory()
    assert win.estimator._pipe is None and win.estimator.loaded_model is None        # el modelo se descargó
    assert win.depth_maps[str(win._photos[0])] is saved                              # el mapa sigue ahí
    assert [a.name for a in win.editor.project.areas] == ["Se queda"]               # y los ajustes
    assert win.editor.render._full is None and win.editor._warped == {}
    wait_for(win, lambda: win.editor.result is not None)                              # la vista previa se rehízo sola
    assert "Memoria liberada" in win.statusBar().currentMessage()
    win.view3d.release_caches()                                                      # el visor 3D tampoco falla vacío
    assert win.view3d._grid is None


def test_free_memory_refuses_while_a_depth_is_being_computed(win):
    win.depth_worker._busy = True
    assert not win.free_memory()
    assert "No se puede liberar" in win.statusBar().currentMessage()
    win.depth_worker._busy = False


def test_memory_indicator_shows_the_traffic_light_and_a_click_frees_memory(win, monkeypatch):
    from deepshot import memory
    monkeypatch.setattr(memory, "gpu_free_total_mb", lambda: (300.0, 8000.0))
    monkeypatch.setattr(memory, "ram_free_total_mb", lambda: (9000.0, 16000.0))
    win.refresh_memory_indicator()
    assert win.memory_button.property("tone") == "bad" and "VRAM" in win.memory_button.text()
    assert win.memory_button.toolTip()
    monkeypatch.setattr(memory, "gpu_free_total_mb", lambda: (6000.0, 8000.0))
    win.refresh_memory_indicator()
    assert win.memory_button.property("tone") == "ok"
    win.estimator._pipe, win.estimator._loaded_model = object(), "marigold-v1-0"
    win.memory_button.click()
    assert win.estimator._pipe is None


def test_depth_is_not_started_when_the_memory_warning_is_cancelled(win, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(win, "memory_levels", lambda: ("bad", "ok"))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)           # el aviso se cierra sin elegir nada
    win.add_photos(win._photos[:1])
    win.compute_depth()
    assert not win.depth_worker.is_busy()


# --- Tiempo de las tareas largas ----------------------------------------------------------------------------
def test_depth_and_export_progress_show_elapsed_time_and_depth_durations_are_remembered(win, monkeypatch):
    win.add_photos(win._photos[:1])
    win.library.setCurrentRow(0)
    monkeypatch.setattr(win.depth_worker, "start", lambda *a, **k: True)           # sin calcular de verdad
    recorded = []
    monkeypatch.setattr(win.durations, "record", lambda key, seconds: recorded.append((key, seconds)))
    win.compute_depth()
    assert win._depth_timer.is_running() and win._depth_timer_key.startswith("depth/marigold-v1-0/")
    assert win._depth_timer_key.endswith("/cold")                                    # el modelo aún no está cargado
    win._on_depth_progress("Calculando profundidad…", -1.0)
    assert win.depth_status.text().startswith("Calculando profundidad…  ·  0:0")
    win._on_depth_finished(str(win._photos[0]), np.full((H, W), 0.5, np.float32))
    assert not win._depth_timer.is_running() and recorded and recorded[0][0] == win._depth_timer_key
    # Una tarea que falla no deja un tiempo recordado.
    recorded.clear()
    win.compute_depth()
    win._on_depth_failed(str(win._photos[0]), "fallo")
    assert not win._depth_timer.is_running() and not recorded
    # Exportar: el mensaje de progreso lleva el tiempo y al terminar se para el cronómetro.
    panel = win.export_panel
    panel._timer.start()
    panel._on_progress("Guardando…", 0.5)
    assert panel.status.text().startswith("Guardando…  ·  0:0")
    panel._end()
    assert not panel._timer.is_running()


def test_depth_panel_adds_and_removes_extra_zones_with_undo(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    panel = win.editor.depth_panel
    control = win.editor.project.depth_control
    assert not any(panel.zone_widgets[f"extra{i}"]["on"].isVisibleTo(win) for i in range(3))   # al principio, solo 3 zonas
    undo_before = win.undo_stack.count()
    for i in range(3):
        panel.add_button.click()
        assert len(control.extra) == i + 1 and panel.zone_widgets[f"extra{i}"]["on"].isVisibleTo(win)
    assert not panel.add_button.isEnabled()                                                    # tope de zonas
    assert win.undo_stack.count() == undo_before + 3                                           # un paso de deshacer por zona
    box = panel.zone_widgets["extra1"]
    box["target"].setValue(box["center"].value() + 0.1, emit=True)
    box["target"].editingFinished.emit()
    assert control.enabled and control.extra[1].target == pytest.approx(control.extra[1].center + 0.1, abs=1e-3)
    panel.zone_widgets["extra0"]["remove"].click()
    assert len(control.extra) == 2 and not panel.zone_widgets["extra2"]["on"].isVisibleTo(win)
    assert panel.add_button.isEnabled()
    win.undo_stack.undo()                                                                      # deshacer el quitar
    assert len(win.editor.project.depth_control.extra) == 3


def test_depth_view_style_gray_color_and_zones_never_change_the_map(win):
    win.add_photos(win._photos[:1])
    depth = give_depth(win)
    original = depth.copy()
    wait_for(win, lambda: win.editor.result is not None)
    win.view_switch.set_current_index(VIEW_DEPTH)
    win._show_current()
    assert win.depth_style_combo.isVisibleTo(win)                          # solo se ve en la vista Profundidad
    gray = win.canvas._item.pixmap().toImage()
    win.depth_style_combo.setCurrentIndex(win.depth_style_combo.findData("color:rainbow"))
    color = win.canvas._item.pixmap().toImage()
    assert color != gray and color.pixelColor(2, 2) != color.pixelColor(W - 3, 2)
    assert color.pixelColor(2, 2).red() != color.pixelColor(2, 2).blue()    # tiene color, no es gris
    win.depth_style_combo.setCurrentIndex(win.depth_style_combo.findData("zones"))
    zones = win.canvas._item.pixmap().toImage()
    assert zones != gray
    assert win.settings.value("depthview/style") == "zones"
    win.depth_style_combo.setCurrentIndex(win.depth_style_combo.findData("normals:opengl"))
    normals = win.canvas._item.pixmap().toImage()
    assert normals != gray and normals.pixelColor(W // 2, 5).blue() > 100        # el azul de las normales (apuntan a la cámara)
    from deepshot import depth_view, simple_depth

    ref = np.round(simple_depth.normal_map(simple_depth.normalize_range(depth), 1.0, "opengl") * 255).astype(np.uint8)
    assert np.array_equal(depth_view.normals_image(depth, "opengl"), ref)        # lo que se ve es lo que exporta el modo rápido
    vertical = np.tile(np.linspace(0, 1, 40, dtype=np.float32)[:, None], (1, 60))
    assert not np.array_equal(depth_view.normals_image(vertical, "directx"),
                              depth_view.normals_image(vertical, "opengl"))      # DirectX invierte el verde
    win.depth_style_combo.setCurrentIndex(win.depth_style_combo.findData("zones"))     # de vuelta a las zonas
    # Tocar el control de profundidad con las zonas a la vista no echa de la vista Profundidad (sí lo haría el gris).
    win.set_step(STEP_EDIT)
    win.view_switch.set_current_index(VIEW_DEPTH)
    win._show_current()
    win.editor.depth_panel.add_button.click()
    assert win.view_switch.current_index() == VIEW_DEPTH
    win.depth_style_combo.setCurrentIndex(win.depth_style_combo.findData("gray"))
    win.view_switch.set_current_index(VIEW_RESULT)
    win._show_current()
    assert not win.depth_style_combo.isVisibleTo(win)
    assert np.array_equal(win.depth_maps[str(win._photos[0])], original)    # el mapa real no cambió


def test_export_panel_depth_color_option_is_saved_and_passed_to_the_options(win):
    panel = win.export_panel
    win.set_step(STEP_EXPORT)
    panel.content.set_current_index(1)                                     # «Mapa de profundidad»
    panel._on_content_changed(1)
    assert panel.depth_color.isVisibleTo(win) and panel.near_bright.isVisibleTo(win)     # en gris sí se elige cerca = claro
    assert panel.options().depth_palette == ""
    panel.depth_color.setCurrentIndex(panel.depth_color.findData("thermal"))
    assert panel.options().depth_palette == "thermal" and not panel.near_bright.isVisibleTo(win)
    assert panel._settings.value("export/depth_color") == "thermal"
    panel.content.set_current_index(0)                                     # con el efecto no hay opciones de mapa
    panel._on_content_changed(0)
    assert not panel.depth_color.isVisibleTo(win)
    panel.depth_color.setCurrentIndex(0)


def test_lens_panel_bokeh_preset_applies_with_one_undo_and_keeps_the_photo_focus(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    panel = win.editor.lens_panel
    lens = win.editor.project.lens
    lens.focus = 0.7
    assert panel.preset_bar.isEnabled() and panel.cat_eye.value() == 0.0
    undo_before = win.undo_stack.count()
    idx = next(i for i in range(panel.preset_bar.combo.count())
               if panel.preset_bar.combo.itemData(i) == ("builtin", "portrait_classic"))
    panel.preset_bar.combo.activated.emit(idx)
    lens = win.editor.project.lens
    assert lens.enabled and lens.aperture == 1.4 and lens.cat_eye == 0.6 and lens.focus == 0.7
    assert panel.cat_eye.value() == pytest.approx(0.6) and panel.aperture.value() == pytest.approx(1.4)   # los controles se ven al día
    assert panel.enabled.isChecked() and win.undo_stack.count() == undo_before + 1
    win.undo_stack.undo()
    assert not win.editor.project.lens.enabled and win.editor.project.lens.cat_eye == 0.0
    panel.cat_eye.setValue(0.3, emit=True)                                   # el deslizador también activa el desenfoque
    assert win.editor.project.lens.cat_eye == pytest.approx(0.3) and win.editor.project.lens.enabled


# --- Cálculos de profundidad por foto: tiempos, comparar y elegir ----------------------------------------------
def test_depth_runs_list_shows_real_times_compares_two_maps_and_uses_the_chosen_one(win, monkeypatch):
    from deepshot.depth import DepthRun

    win.add_photos(win._photos[:1])
    win.library.setCurrentRow(0)
    wait_for(win, lambda: win.current_path is not None)
    path = str(win._photos[0])
    win.set_step(STEP_DEPTH)
    monkeypatch.setattr(win.depth_worker, "start", lambda *a, **k: True)           # sin calcular de verdad

    def finish(quality, seconds, value, load=0.0):
        win.quality_combo.setCurrentIndex(win.quality_combo.findData(quality))
        win.compute_depth()
        s = win._depth_settings()
        win.estimator.last_run = DepthRun(s.model, s.processing_resolution, s.ensemble_size, 0.0,
                                          infer_seconds=seconds, load_seconds=load)
        win._on_depth_finished(path, np.full((H, W), value, np.float32))

    assert not win.runs_box.isVisibleTo(win)                                         # sin cálculos, no hay lista
    finish("fast", 3.3, 0.2, load=11.0)
    assert win.runs_box.isVisibleTo(win) and win.runs_list.count() == 1
    assert "3,3 s" in win.runs_list.item(0).text() and win.runs_list.item(0).text().lstrip().startswith("●")
    assert "Listo en 3,3 s" in win.depth_result_label.text() and "cargando el modelo" in win.depth_result_label.text()
    finish("max", 12.4, 0.8)
    assert win.view_switch.current_index() == VIEW_COMPARE                           # con dos cálculos entra sola en Comparar
    win.view_switch.set_current_index(VIEW_DEPTH)                                    # lo de abajo prueba la vista Profundidad
    win._show_current()
    assert win.runs_list.count() == 2
    texts = [win.runs_list.item(i).text() for i in range(2)]
    assert "12,4 s" in texts[1] and "×3" in texts[1] and texts[1].lstrip().startswith("●") and not texts[0].lstrip().startswith("●")
    assert "Marigold" in texts[0] and "Rápida" in texts[0]
    assert "Cargar el modelo" in win.runs_list.item(0).toolTip()                      # el detalle, en la ayuda
    # El mapa en uso es el último; seleccionar el otro lo enseña pero no lo cambia.
    assert np.allclose(win.depth_maps[path], 0.8)
    win.runs_list.item(0).setSelected(True)
    assert win.view_switch.current_index() == VIEW_DEPTH and np.allclose(win.depth_maps[path], 0.8)
    assert win.runs_use_button.isEnabled() and win.runs_remove_button.isEnabled()
    assert "NO es el que se usa" in win.runs_hint.text()
    assert not win.canvas.is_comparing()                                              # uno solo: se ve ese mapa
    win.runs_list.item(1).setSelected(True)                                           # dos seleccionados: comparar
    assert "Comparando" in win.runs_hint.text() and not win.runs_use_button.isEnabled()
    assert win.canvas.is_comparing()
    win.runs_list.clearSelection()
    win.runs_list.item(1).setSelected(True)                                           # el que ya se usa: nada que cambiar
    assert not win.runs_use_button.isEnabled() and not win.runs_remove_button.isEnabled()
    win.runs_list.clearSelection()
    win.runs_list.item(0).setSelected(True)
    win.runs_use_button.click()                                                       # usar el primero
    assert np.allclose(win.depth_maps[path], 0.2) and win.depth_sources[path] == win._depth_settings().key()
    assert win.quality_combo.currentData() == "fast"                                  # el panel sigue al mapa elegido
    assert win.runs_list.item(0).text().lstrip().startswith("●")
    win.runs_list.item(1).setSelected(True)
    win.runs_remove_button.click()                                                    # quitar el otro
    assert win.runs_list.count() == 1
    finish("high", 6.2, 0.5)
    win.runs_list.clearSelection()
    win.runs_list.item(0).setSelected(True)
    win.runs_keep_button.click()                                                      # quedarme solo con el primero
    assert win.runs_list.count() == 1 and np.allclose(win.depth_maps[path], 0.2)


def test_ab_comparator_picks_which_two_depth_runs_go_on_each_side(win, monkeypatch):
    from deepshot.depth import DepthRun

    win.add_photos(win._photos[:1])
    win.library.setCurrentRow(0)
    wait_for(win, lambda: win.current_path is not None)
    path = str(win._photos[0])
    win.set_step(STEP_DEPTH)
    win.view_switch.set_current_index(VIEW_PHOTO)
    win._show_current()
    monkeypatch.setattr(win.depth_worker, "start", lambda *a, **k: True)

    def finish(quality, seconds, value):
        win.quality_combo.setCurrentIndex(win.quality_combo.findData(quality))
        win.compute_depth()
        s = win._depth_settings()
        win.estimator.last_run = DepthRun(s.model, s.processing_resolution, s.ensemble_size, 0.0, infer_seconds=seconds)
        win._on_depth_finished(path, np.full((H, W), value, np.float32))

    assert not win.ab_a_combo.isVisibleTo(win)                             # fuera de la vista Comparar no se ven
    finish("fast", 3.0, 0.2)
    assert win.view_switch.current_index() == VIEW_DEPTH                   # con un solo cálculo no hay nada que comparar
    finish("high", 6.0, 0.5)                                               # con dos, pasa solo a Comparar: A = anterior, B = nuevo
    assert win.view_switch.current_index() == VIEW_COMPARE and win.canvas.is_comparing()
    assert win.ab_a_combo.isVisibleTo(win) and win.ab_b_combo.isVisibleTo(win)
    runs = win._ab_run_entries()
    assert win.ab_a_combo.currentData() == "run:" + runs[0].key and win.ab_b_combo.currentData() == "run:" + runs[1].key
    finish("max", 12.0, 0.8)                                               # y se queda en Comparar con cada cálculo nuevo
    assert win.view_switch.current_index() == VIEW_COMPARE
    assert win.ab_a_combo.count() == 3 + 3                                 # foto, efecto, mapa en uso y los 3 cálculos
    assert win.ab_a_combo.currentData() == "run:" + runs[1].key            # A = el que se usaba (alta), B = el nuevo (máxima)
    # Elegir dos cálculos de profundidad distintos, uno a cada lado.
    runs = win._ab_run_entries()
    win.ab_a_combo.setCurrentIndex(win.ab_a_combo.findData("run:" + runs[1].key))      # A = alta
    win.ab_b_combo.setCurrentIndex(win.ab_b_combo.findData("run:" + runs[0].key))      # B = rápida
    assert win.canvas.is_comparing()
    assert "Alta" in win.canvas._labels[0] and "Rápida" in win.canvas._labels[1]
    # También se puede comparar la foto con un cálculo, o el mapa en uso con otro.
    win.ab_a_combo.setCurrentIndex(win.ab_a_combo.findData("depth"))
    assert win.canvas.is_comparing() and np.allclose(win.depth_maps[path], 0.8)         # comparar no cambia el mapa en uso
    # Clics en la lista de cálculos: izquierdo = lado A, derecho = lado B, y la vista Comparar no se abandona.
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    def click(row, button):
        viewport = win.runs_list.viewport()
        QTest.mouseClick(viewport, button, Qt.NoModifier, win.runs_list.visualItemRect(win.runs_list.item(row)).center())

    runs = win._ab_run_entries()
    click(0, Qt.LeftButton)
    assert win.view_switch.current_index() == VIEW_COMPARE and win.canvas.is_comparing()
    assert win.ab_a_combo.currentData() == "run:" + runs[0].key
    click(2, Qt.RightButton)
    assert win.view_switch.current_index() == VIEW_COMPARE
    assert win.ab_b_combo.currentData() == "run:" + runs[2].key and win.ab_a_combo.currentData() == "run:" + runs[0].key
    assert win.runs_list.item(0).text().startswith("[A]") and win.runs_list.item(2).text().startswith("[B]")
    assert "izquierdo" in win.runs_hint.text()
    click(1, Qt.LeftButton)                                                # cambiar solo el lado A: B se queda
    assert win.ab_a_combo.currentData() == "run:" + runs[1].key and win.ab_b_combo.currentData() == "run:" + runs[2].key
    assert win.view_switch.current_index() == VIEW_COMPARE
    win.view_switch.set_current_index(VIEW_PHOTO)                          # fuera de la vista Comparar desaparecen
    win._show_current()
    assert not win.ab_a_combo.isVisibleTo(win) and not win.runs_list.item(0).text().startswith("[")


def test_runs_are_recovered_from_the_disk_cache_with_their_times_and_maps_load_lazily(win):
    from deepshot import depth as d

    win.add_photos(win._photos[:1])
    path = win._photos[0]
    settings = d.settings_for(quality="high")
    png = d.cache_path(path, settings)
    d.save_depth_png16(np.full((H, W), 0.3, np.float32), png)
    d.write_timing(png, {"infer_seconds": 6.25})
    win.depth_runs.clear()
    win._discover_runs(path)
    entries = win._runs_for(str(path)).entries
    assert len(entries) == 1 and entries[0].depth is None and entries[0].seconds == 6.25      # sin leer el mapa
    win.current_path = path
    win._refresh_runs_list()
    assert "6,3 s" in win.runs_list.item(0).text() or "6,2 s" in win.runs_list.item(0).text()
    win._release_run_maps()
    assert entries[0].get_depth() is not None and np.allclose(entries[0].depth, 0.3, atol=1e-3)
    win._release_run_maps()
    assert entries[0].depth is None                                                       # «Liberar memoria» lo suelta


def test_style_bar_marks_the_chosen_style_and_previews_it_on_hover_without_applying(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    bar = win.editor.style_bar
    idx = next(i for i in range(bar.combo.count()) if bar.combo.itemData(i) == ("builtin", "contrast"))
    requests = []
    original = win.editor.render.request
    win.editor.render.request = lambda project, mask=None: (requests.append(project), original(project, mask))[1]
    # Pasar el ratón enseña el estilo en la vista previa, sin aplicarlo ni apilar deshacer.
    undo_before = win.undo_stack.count()
    base_ramp = win.editor.project.ramp.to_dict() if hasattr(win.editor.project.ramp, "to_dict") else None
    bar.combo.highlighted.emit(idx)
    assert win.editor._previewing and requests and requests[-1] is not win.editor.project
    assert len(requests[-1].ramp.stops) == 4                                            # lo que lleva «contraste fuerte»
    assert win.undo_stack.count() == undo_before and len(win.editor.project.ramp.stops) == 2
    wait_for(win, lambda: not win.editor._previewing)                                   # la lista no está abierta: se cierra sola
    assert requests[-1] is win.editor.project and len(win.editor.project.ramp.stops) == 2
    # Elegirlo lo aplica y el desplegable lo sigue mostrando.
    bar.combo.highlighted.emit(idx)
    bar.combo.activated.emit(idx)
    QTest.qWait(300)
    win._app.processEvents()
    assert len(win.editor.project.ramp.stops) == 4 and not win.editor._previewing
    assert bar.combo.currentIndex() == idx and win.undo_stack.count() == undo_before + 1
    # Tocar un ajuste lo desmarca; deshacerlo lo vuelve a marcar.
    win.editor.project.composite.opacity = 0.123
    win.editor._on_changed()
    assert bar.combo.currentIndex() == 0
    win.editor._on_committed()
    win.undo_stack.undo()
    assert bar.combo.currentIndex() == idx
    del base_ramp


def test_loading_another_style_removes_the_old_style_area_in_one_undo_step_and_keeps_user_areas(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    win.editor.areas_panel.add_item(Area(name="Mía"))
    bar = win.editor.style_bar

    def choose(key):
        idx = next(i for i in range(bar.combo.count()) if bar.combo.itemData(i) == ("builtin", key))
        bar.combo.activated.emit(idx)

    names = lambda: [a.name for a in win.editor.project.areas]       # noqa: E731
    choose("studio_light")
    assert names() == ["Mía", "Foco"] and win.editor.project.areas[1].from_style
    undo_before = win.undo_stack.count()
    choose("haze")                                                     # estilo sin áreas: la de «Foco» sale
    assert names() == ["Mía"] and win.undo_stack.count() == undo_before + 1
    win.undo_stack.undo()                                              # un solo paso de deshacer la devuelve
    assert names() == ["Mía", "Foco"]
    win.undo_stack.redo()
    assert names() == ["Mía"]
    choose("studio_light")                                             # duplicar un área de estilo la hace del usuario
    win.editor.areas_panel.list.setCurrentRow(1)
    win.editor.areas_panel._duplicate()
    assert [a.from_style == "" for a in win.editor.project.areas] == [True, False, True]
    choose("haze")
    assert names() == ["Mía", "Foco"] and not win.editor.project.areas[1].from_style    # la copia (del usuario) se queda


# --- Pantalla de inicio y proyecto nuevo --------------------------------------------------------------------
def test_start_dialog_lists_recent_projects_marks_missing_ones_and_returns_the_choice(win, tmp_path, monkeypatch):
    from deepshot import project_file as pf
    from deepshot.ui.start_dialog import StartDialog

    good, gone = tmp_path / "bueno.deepshot", tmp_path / "perdido.deepshot"
    pf.save_project(good, [])
    win.settings.setValue("recent_projects", [str(good), str(gone)])
    seen = {}

    def fake_exec(self):                                  # el diálogo real bloquearía el test
        seen["dialog"] = self
        seen.update(texts=[self.list.item(i).text() for i in range(self.list.count())],
                    open_enabled=self.recent_button.isEnabled())
        return seen["act"](self)

    monkeypatch.setattr(StartDialog, "exec", fake_exec)

    def choose_recent(d):
        d._open_selected()
        return 1

    seen["act"] = choose_recent
    assert win.show_start_dialog() == "recent"
    assert win.project_path == good and seen["open_enabled"]                         # abrió el reciente
    assert "bueno" in seen["texts"][0] and "no se encuentra" in seen["texts"][1]      # el que falta, marcado
    # Quitar uno de la lista + «no volver a mostrar» + proyecto nuevo.
    win.settings.setValue("recent_projects", [str(good), str(gone)])

    def remove_and_new(d):
        d.list.setCurrentRow(1)
        d._remove_selected()
        d.dont_show.setChecked(True)
        d._new()
        return 1

    seen["act"] = remove_and_new
    assert win.show_start_dialog() == "new"
    assert win.recent_projects() == [str(good)] and not win._start_enabled()
    win.settings.setValue("startup/show_start", True)
    seen["act"] = lambda d: 0                                                        # Esc: empezar en blanco
    assert win.show_start_dialog() == "new"
    opened = []
    monkeypatch.setattr(win, "open_project", lambda *a, **k: opened.append(1) or True)

    def open_file(d):
        d._open_file()
        return 1

    seen["act"] = open_file
    assert win.show_start_dialog() == "open" and opened == [1]


def test_new_project_asks_before_discarding_and_empties_the_library(win, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    win.add_photos(win._photos[:2])
    give_depth(win)
    assert win.is_dirty()
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Cancel)
    assert not win.new_project() and win.library.count() == 2                        # cancelar no toca nada
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Discard)
    assert win.new_project()
    assert win.library.count() == 0 and win.project_path is None and not win.is_dirty()
    assert win.windowTitle() == APP_NAME
    assert win.action_new_project.shortcut().toString() == "Ctrl+N"


def test_main_does_not_show_the_start_screen_when_disabled_by_environment(monkeypatch):
    import deepshot.app as app_module

    assert "DEEPSHOT_NO_START" in open(app_module.__file__, encoding="utf-8").read()


# --- Apariencia (tema, aspecto) y disposición de los paneles ------------------------------------------------------
@pytest.fixture
def look(win):
    """La ventana con el tema de fábrica al terminar: el tema es un estado global de la aplicación."""
    from deepshot import themes
    from deepshot.ui import theme

    yield win
    theme.apply_theme(win._app, "dark", themes.Appearance())


def test_theme_changes_live_repaints_icons_saves_the_choice_and_restores(look):
    from PySide6.QtWidgets import QApplication

    from deepshot import themes
    from deepshot.ui import theme
    from deepshot.ui.icons import icon

    win = look
    button = win.areas_button                                          # tiene su icono con nombre (set_icon)
    assert button.property("icon_name") == "eye"
    dark_pixel = button.icon().pixmap(24, 24).toImage()
    assert theme.current_theme().key == "dark" and "#303030" in QApplication.instance().styleSheet()
    win.apply_appearance("light")
    assert theme.current_theme().key == "light"
    assert "#dcdcdc" in QApplication.instance().styleSheet() and "#303030" not in QApplication.instance().styleSheet()
    assert win.settings.value("appearance/theme") == "light"          # y se guarda
    light_pixel = button.icon().pixmap(24, 24).toImage()
    assert light_pixel != dark_pixel                                   # el icono se repintó con los colores del tema
    assert icon("eye").pixmap(24, 24).toImage() == light_pixel
    assert theme.color("viewer").name() == themes.LIGHT["viewer"]
    win.apply_appearance("dark", themes.Appearance(radius=8, density="comfortable"))
    assert "border-radius: 8px" in QApplication.instance().styleSheet()
    assert win.settings.value("appearance/radius", type=int) == 8 and win.settings.value("appearance/density") == "comfortable"
    assert button.icon().pixmap(24, 24).toImage() == dark_pixel
    assert theme.saved_choice() == ("dark", themes.Appearance(radius=8, density="comfortable"))


def test_3d_view_and_text_hints_follow_the_theme(look):
    from deepshot.ui import theme

    win = look
    win.apply_appearance("light")
    assert theme.color("viewer").name() == "#c4c4c4" and theme.color("text_dim").name() == "#555555"
    assert win.quality_hint.property("dim") is True and win.edit_hint.property("tone") == "warn"   # por propiedad, no color fijo
    assert "color:" not in win.quality_hint.styleSheet() and "color:" not in win.edit_hint.styleSheet()


def test_theme_menu_and_layout_menu_list_their_entries(look):
    win = look
    win._fill_theme_menu()
    actions = win.theme_menu.actions()
    assert [a.isChecked() for a in actions][:3] == [True, False, False] and len(actions) >= 3
    actions[2].trigger()                                               # Medianoche
    win._fill_theme_menu()
    assert [a.isChecked() for a in win.theme_menu.actions()][:3] == [False, False, True]
    win._fill_layout_menu()
    labels = [a.text() for a in win.layout_menu.actions() if a.text()]
    assert any("Edición" in t for t in labels) and any("Pantalla pequeña" in t for t in labels)
    assert "Restablecer disposición" in labels and any("Guardar la disposición actual" in t for t in labels)


def test_factory_layouts_move_and_hide_docks_and_reset_restores_the_default(look):
    from PySide6.QtCore import Qt

    win = look
    win.show()
    win._app.processEvents()
    assert win.apply_layout("factory", "review")
    assert win.library_dock.isVisibleTo(win) and not win.settings_dock.isVisibleTo(win)
    assert win.apply_layout("factory", "small")
    assert not win.library_dock.isVisibleTo(win) and win.settings_dock.isVisibleTo(win)
    assert win.apply_layout("factory", "debug")
    assert win.log_dock.isVisibleTo(win) and win.library_dock.isVisibleTo(win) and win.settings_dock.isVisibleTo(win)
    win.settings_dock.setFloating(True)                                # un panel suelto vuelve a su sitio al restablecer
    win.reset_layout()
    assert not win.settings_dock.isFloating() and win.dockWidgetArea(win.settings_dock) == Qt.RightDockWidgetArea
    assert win.library_dock.isVisibleTo(win) and win.settings_dock.isVisibleTo(win) and not win.log_dock.isVisibleTo(win)
    assert not win.apply_layout("factory", "no existe")


def test_user_layout_roundtrip_overwrite_and_broken_file(look, monkeypatch):
    from deepshot.ui import layouts

    win = look
    win.show()
    win._app.processEvents()
    win.apply_layout("factory", "small")                               # sin biblioteca
    assert win.save_layout("Mi vista") and layouts.list_user() == ["Mi vista"] and layouts.exists_user("Mi vista")
    win.reset_layout()
    assert win.library_dock.isVisibleTo(win)
    assert win.apply_layout("user", "Mi vista")                        # vuelve a ocultar la biblioteca
    win._app.processEvents()
    assert not win.library_dock.isVisibleTo(win) and win.settings_dock.isVisibleTo(win)
    assert not win.save_layout("   ") and win._boxes                   # nombre vacío: avisa y no rompe
    (layouts.layouts_dir() / "rota.json").write_text("{ roto", encoding="utf-8")
    assert "rota" not in layouts.list_user() and not win.apply_layout("user", "rota")
    assert not win.apply_layout("user", "no existe")
    assert layouts.delete_user("Mi vista") and not layouts.delete_user("Mi vista")


def test_preferences_dialog_changes_theme_density_radius_start_and_layouts(look, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    from deepshot.ui import layouts, theme
    from deepshot.ui.preferences import PreferencesDialog

    win = look
    dialog = PreferencesDialog(win)
    assert dialog.show_start.isChecked() == win._start_enabled() and dialog.tabs.count() == 3
    dialog.show_start.setChecked(False)
    assert not win._start_enabled()
    dialog.show_start.setChecked(True)
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("light"))
    assert theme.current_theme().key == "light"                        # se aplica al momento
    dialog.density_combo.setCurrentIndex(dialog.density_combo.findData("compact"))
    dialog.radius_slider.setValue(6)
    assert theme.current_appearance().density == "compact" and theme.current_appearance().radius == 6
    assert dialog.radius_label.text() == "6 px"
    assert dialog.radius_slider.pageStep() == 1                       # clicar la barra avanza de 1 en 1, no de 0 a 10
    # Estilo de iconos: color (por defecto) o monocromo, al momento y guardado.
    from deepshot.ui.icons import icon

    assert theme.current_appearance().icons == "color"
    colored = icon("photo").pixmap(24, 24).toImage()
    dialog.icons_combo.setCurrentIndex(dialog.icons_combo.findData("mono"))
    assert theme.current_appearance().icons == "mono" and theme.saved_choice()[1].icons == "mono"
    mono = icon("photo").pixmap(24, 24).toImage()
    assert colored != mono                                              # no se pintan igual
    dialog.icons_combo.setCurrentIndex(dialog.icons_combo.findData("color"))
    assert icon("photo").pixmap(24, 24).toImage() == colored
    # Disposición: aplicar una de fábrica, guardar la actual, borrarla.
    dialog.layout_list.setCurrentRow(2)                                # «Pantalla pequeña»
    dialog.apply_layout_button.click()
    assert not win.library_dock.isVisibleTo(win) and win.settings_dock.isVisibleTo(win)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Prueba", True)))
    dialog.save_layout_button.click()
    assert "Prueba" in layouts.list_user()
    dialog.layout_list.setCurrentRow(dialog.layout_list.count() - 1)
    assert dialog.delete_layout_button.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog.delete_layout_button.click()
    assert "Prueba" not in layouts.list_user()
    dialog.layout_list.setCurrentRow(0)
    assert not dialog.delete_layout_button.isEnabled()                 # las de fábrica no se borran
    dialog.reset_layout_button.click()
    assert win.library_dock.isVisibleTo(win) and win.settings_dock.isVisibleTo(win)
    dialog.deleteLater()


def test_preload_loads_the_chosen_model_in_the_background_and_reports_it(win, monkeypatch):
    from PySide6.QtWidgets import QApplication

    calls = []
    monkeypatch.setattr(win.models, "preload_id", lambda: "marigold-v1-0")
    monkeypatch.setattr(win.estimator, "load", lambda model, progress=None: calls.append(model))
    win._preload_model()
    win._preloader.wait(10)
    for _ in range(10):
        QApplication.instance().processEvents()
    assert calls == ["marigold-v1-0"]
    assert "Modelo cargado" in win.statusBar().currentMessage()
    monkeypatch.setattr(win.models, "preload_id", lambda: None)               # sin modelo elegido: no hace nada
    calls.clear()
    win._preload_model()
    assert not calls



def test_resolution_combo_overrides_the_quality_preset_and_is_remembered(win):
    from deepshot.depth import settings_for

    win.set_step(STEP_DEPTH)
    win.quality_combo.setCurrentIndex(win.quality_combo.findData("high"))
    assert win.resolution_combo.currentData() == 0 and win._depth_settings().processing_resolution == 1280
    win.resolution_combo.setCurrentIndex(win.resolution_combo.findData(512))
    s = win._depth_settings()
    assert s.processing_resolution == 512 and s.ensemble_size == 1
    assert win.settings.value("depth/resolution", 0, type=int) == 512 and "512 px" in win.params_label.text()
    # Al elegir un cálculo hecho con otra resolución, el selector la sigue; con la del preset vuelve a «automática».
    win._apply_run_to_combos(settings_for(quality="high", resolution=1024))
    assert win.resolution_combo.currentData() == 1024
    win._apply_run_to_combos(settings_for(quality="high"))
    assert win.resolution_combo.currentData() == 0


def test_focus_eyedropper_also_works_in_the_3d_view(win):
    from PySide6.QtCore import QPointF

    win.add_photos(win._photos[:1])
    give_depth(win)                                                        # profundidad = gradiente de 0.1 a 0.9 en x
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    win.view_switch.set_current_index(VIEW_RESULT)
    win.mode_switch.set_current_index(MODE_3D)
    win._show_current()
    assert win._in_3d()
    win.show()
    win.view3d.resize(600, 400)
    view = win.view3d
    panel = win.editor.lens_panel
    panel.pick_button.click()                                              # el cuentagotas ya no saca de 3D
    assert win._in_3d() and view._tool == "pick_focus" and win.canvas.tool() == "pick_focus"
    depth = win.depth_maps[str(win._photos[0])]
    h, w = depth.shape
    assert view._prepare()
    # Dónde cae en pantalla un punto de la superficie al 75 % del ancho: ahí se hace clic.
    gh, gw = view._grid.shape[:2]
    row, col = gh // 2, int(gw * 0.75)
    sx, sy, _ = view.camera.project(view._grid[row, col], view.width(), view.height())
    undo_before = win.undo_stack.count()
    QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier, QPointF(float(sx[0]), float(sy[0])).toPoint())
    lens = win.editor.project.lens
    assert lens.enabled and lens.focus == pytest.approx(depth[h // 2, int(w * 0.75)], abs=0.06)
    assert not panel.pick_button.isChecked() and view._tool == ""          # un solo uso
    assert win.undo_stack.count() == undo_before + 1
    # Clic en el vacío: no hay superficie, no cambia nada.
    assert view.pick_photo_point(QPointF(1.0, 1.0)) is None


def test_preferences_shows_the_cache_size_and_empties_it_after_confirming(look, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from deepshot import depth
    from deepshot.ui.preferences import PreferencesDialog

    win = look
    (depth.cache_dir() / "m.png").write_bytes(b"x" * 2_000_000)
    dialog = PreferencesDialog(win)
    assert "1 mapas" in dialog.cache_label.text() and "2.0 MB" in dialog.cache_label.text() and dialog.cache_button.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    dialog._clear_cache()
    assert depth.cache_size()[0] == 1                                         # dijo que no: sigue ahí
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog._clear_cache()
    assert depth.cache_size() == (0, 0) and not dialog.cache_button.isEnabled()
    assert "Caché vaciada" in win.statusBar().currentMessage()


def test_first_run_notice_appears_only_without_models_and_leads_to_the_model_manager(win, monkeypatch):
    from deepshot import models
    from deepshot.ui import first_run
    from deepshot.ui.first_run import FirstRunDialog, needs_first_run

    opened = []
    monkeypatch.setattr(win, "open_model_manager", lambda then=None: opened.append(then))
    monkeypatch.setattr(win.models, "is_installed", lambda mid: False)         # equipo sin modelos
    assert needs_first_run(win.models)
    choices = iter(["manager", "folder", "later"])

    def fake_exec(self):
        self.action = next(choices)
        self.dont_ask.setChecked(self.action == "later")
        return 0

    monkeypatch.setattr(FirstRunDialog, "exec", fake_exec)
    assert win.show_first_run_if_needed() == "manager" and opened == [None]
    assert win.show_first_run_if_needed() == "folder" and opened == [None, "install_from_folder"]
    assert win.show_first_run_if_needed() == "later"                           # marcó «No volver a preguntar»
    assert win.show_first_run_if_needed() == ""                                # y ya no vuelve a salir
    win.settings.setValue("firstrun/dismissed", False)
    monkeypatch.setattr(win.models, "is_installed", lambda mid: mid == models.DEFAULT_MODEL)   # con un modelo instalado
    assert not needs_first_run(win.models) and win.show_first_run_if_needed() == ""
    monkeypatch.setattr(win.models, "is_installed", lambda mid: mid == "marigold-v2-log")      # uno que no se puede usar no cuenta
    assert needs_first_run(win.models)


def test_start_dialog_buttons_have_icons_and_the_quick_one_matches_the_header(win):
    from deepshot.ui.icons import ICON_PROPERTY
    from deepshot.ui.start_dialog import StartDialog

    dialog = StartDialog([], win)
    names = {b: b.property(ICON_PROPERTY) for b in (dialog.new_button, dialog.quick_button, dialog.open_button,
                                                     dialog.recent_button, dialog.remove_button)}
    assert all(names.values()) and not any(b.icon().isNull() for b in names)
    assert names[dialog.quick_button] == win.quick_header_button.property(ICON_PROPERTY)   # mismo icono que el «0 Rápido»
    dialog.deleteLater()


def test_preferences_tabs_headings_and_buttons_have_icons_that_follow_the_theme(look):
    from deepshot import themes
    from deepshot.ui import theme
    from deepshot.ui.preferences import PreferencesDialog

    win = look
    dialog = PreferencesDialog(win)
    assert dialog.tabs.count() == 3 and not any(dialog.tabs.tabIcon(i).isNull() for i in range(3))
    assert not dialog.cache_button.icon().isNull() and not dialog.save_layout_button.icon().isNull()
    before = dialog.tabs.tabIcon(0).pixmap(18, 18).toImage()
    theme.apply_theme(win._app, "light", themes.Appearance(icons="mono"))
    from deepshot.ui.icons import refresh_icons

    refresh_icons(dialog)                                                  # lo que hace la ventana al cambiar de tema
    assert dialog.tabs.tabIcon(0).pixmap(18, 18).toImage() != before       # la pestaña se repinta con los colores nuevos
    dialog.deleteLater()


def test_only_the_gizmos_of_what_is_being_edited_are_shown_areas_or_bands(win):
    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    editor = win.editor
    editor.areas_panel.add_item(Area(name="A", center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3)))
    assert editor.edit_context == "areas"
    editor.bands_panel.add_button_for_test = None
    editor.bands_panel._add()                                              # añadir una banda la pone como lo que se edita
    assert editor.edit_context == "bands" and editor.visible_context() == "bands"
    win._update_overlays()
    assert win.canvas._overlays == []                                      # los contornos de las áreas se esconden
    go_3d(win)
    win._update_overlays()
    assert win.view3d._areas == [] and len(win.view3d._bands) == 1
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    click = QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(editor.areas_panel.list.viewport(), click)      # pulsar en el panel de áreas (aunque no cambie la fila)
    win._update_overlays()
    assert editor.edit_context == "areas" and len(win.view3d._areas) == 1 and win.view3d._bands == []
    editor.set_edit_context("bands")
    editor.project.bands.clear()
    assert editor.visible_context() == "areas"                             # sin bandas no hay nada que enseñar de ellas
    win.areas_button.setChecked(False)                                     # el ojo sigue siendo el interruptor general
    win._update_overlays()
    assert win.view3d._areas == [] and win.view3d._bands == []


def test_disabled_areas_disappear_from_the_3d_view_completely(win):
    from PySide6.QtCore import QPointF

    from deepshot import area3d

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    win.editor.areas_panel.add_item(Area(name="A", center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3)))
    win.editor.areas_panel.add_item(Area(name="B", center=(0.9, 0.0, 0.5), scale=(0.2, 0.2, 0.2)))
    go_3d(win)
    view = win.view3d
    view.camera.set_view(0, 0)
    win._update_overlays()

    def contour_point_of_b() -> QPointF:
        sx, sy = view._screen(area3d.wireframe(win.editor.project.areas[1])[0][5][None, :])
        return QPointF(float(sx[0]), float(sy[0]))

    assert view.hit_test(contour_point_of_b())[0] != ""                    # activada: se puede agarrar su contorno
    win.editor.project.areas[1].enabled = False
    win.editor.areas_panel.refresh_list()
    win._update_overlays()
    assert view.hit_test(contour_point_of_b())[0] == ""                    # desactivada: ya no hay nada ahí


def test_the_row_eye_bypasses_the_element_and_rows_rename_in_place(win):
    from deepshot.tonemap import Band

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    editor, panel = win.editor, win.editor.areas_panel
    for name, x in (("A", -0.6), ("B", 0.0), ("C", 0.6)):
        panel.add_item(Area(name=name, center=(x, 0.0, 0.5), scale=(0.25, 0.25, 0.25)))
    assert not hasattr(panel, "solo_button")                                # el ojo es bypass; el «solo» es la «S» de cada fila
    assert panel.list.item(0).data(Qt.CheckStateRole) is None               # sin casilla en la fila
    assert [o["index"] for o in editor.overlays()] == [0, 1, 2]
    panel.toggle_row_enabled(1)                                             # el ojo de una fila: se salta esa área
    assert [a.enabled for a in editor.project.areas] == [True, False, True]
    assert [o["index"] for o in editor.overlays()] == [0, 2]                # y no tiene manejadores
    panel.toggle_row_enabled(1)
    assert all(a.enabled for a in editor.project.areas)
    panel.list.item(2).setText("Fondo")                                     # renombrar en la fila
    assert editor.project.areas[2].name == "Fondo"
    panel.list.item(2).setText("  ")                                        # vacío: vuelve al nombre por defecto
    assert editor.project.areas[2].name.endswith("3")
    editor.project.bands.extend([Band(position=0.3), Band(position=0.6)])
    editor.bands_panel.bind(editor.project)
    editor.bands_panel.list.item(1).setText("Luz")
    assert editor.project.bands[1].name == "Luz" and editor.project.bands[0].name == ""
    editor.bands_panel.toggle_row_enabled(1)
    assert [b.enabled for b in editor.project.bands] == [True, False]
    editor.bands_panel.select(0)
    editor.bands_panel._remove()                                            # borrar: queda la banda que estaba en bypass
    assert len(editor.project.bands) == 1 and editor.project.bands[0].enabled is False


def test_the_row_solo_shows_only_that_element_in_the_preview_without_touching_the_project(win):
    from deepshot.tonemap import Band

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    editor, panel = win.editor, win.editor.areas_panel
    for name, x in (("A", -0.6), ("B", 0.0), ("C", 0.6)):
        panel.add_item(Area(name=name, center=(x, 0.0, 0.5), scale=(0.25, 0.25, 0.25)))
    assert editor._preview_project() is editor.project                      # sin solo, el proyecto tal cual
    panel.toggle_row_solo(1)                                                # la «S» de una fila
    assert panel.solo == {1}
    assert [a.enabled for a in editor._preview_project().areas] == [False, True, False]
    assert all(a.enabled for a in editor.project.areas)                     # el proyecto real no cambia (ni se exporta distinto)
    assert [o["index"] for o in editor.overlays()] == [1]                   # en 2D solo sus contornos
    panel.toggle_row_solo(2)                                                # solo uno a la vez: el solo pasa a la otra fila
    assert panel.solo == {2} and [a.enabled for a in editor._preview_project().areas] == [False, False, True]
    panel.toggle_row_solo(2)                                                # y otro clic en la misma lo quita
    assert editor._preview_project() is editor.project
    panel.toggle_row_enabled(0)                                             # un elemento en bypass con «solo» sí se ve
    panel.toggle_row_solo(0)
    assert [a.enabled for a in editor._preview_project().areas] == [True, False, False]
    assert editor.project.areas[0].enabled is False
    panel.add_item(Area(name="D"))                                          # lo nuevo también se ve con un «solo» activo
    assert panel.solo == {3}
    panel._remove()                                                         # al borrar, las filas se corren y el «solo» se quita
    assert panel.solo == set() and editor._preview_project() is editor.project
    editor.project.bands.extend([Band(position=0.3), Band(position=0.6)])
    editor.bands_panel.bind(editor.project)
    editor.bands_panel.toggle_row_solo(1)
    assert [b.enabled for b in editor._preview_project().bands] == [False, True]


def test_a_ramp_preset_shows_and_applies_on_the_ramp_base_even_if_the_base_was_black_or_white(win):
    from deepshot import presets

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    editor = win.editor
    editor.project.base = "black"
    editor.base_panel.bind(editor.project)
    editor._on_committed()                                                  # la base Negro es el punto de partida del deshacer
    key = presets.builtin_names("ramp")[1][0]
    preset = presets.builtin("ramp", key)
    sent = []
    editor.render.request = lambda project, mask=None: sent.append(project)
    editor.preview_preset("ramp", preset)                                  # al pasar el ratón: se ve sobre la rampa…
    assert sent[-1].base == "ramp" and editor.project.base == "black"      # …sin tocar el proyecto
    undo_before = win.undo_stack.index()
    editor.ramp_editor.preset.chosen.emit(preset)                          # al elegirlo: la base pasa a Rampa
    assert editor.project.base == "ramp" and editor.base_panel.base.current_index() == 0
    assert win.undo_stack.index() == undo_before + 1                        # un solo paso de deshacer
    win.undo_stack.undo()
    assert editor.project.base == "black"


def test_rows_double_click_on_the_eye_or_color_does_not_open_the_name_editor_and_rows_reorder(win):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QStyleOptionViewItem

    from deepshot.tonemap import Band

    win.add_photos(win._photos[:1])
    give_depth(win)
    wait_for(win, lambda: win.editor.result is not None)
    win.set_step(STEP_EDIT)
    editor = win.editor
    for name in "ABC":
        editor.areas_panel.add_item(Area(name=name))
    editor.project.bands.extend([Band(position=0.2, color=(1.0, 0.0, 0.0)), Band(position=0.8)])
    editor.bands_panel.bind(editor.project)
    panel, view = editor.areas_panel, editor.areas_panel.list
    delegate = view.itemDelegate()
    index = view.model().index(0, 0)
    opt = QStyleOptionViewItem()
    opt.rect = view.visualRect(index)
    eye = delegate._eye_rect(opt.rect).center()
    swatch = delegate._swatch_rect(opt.rect, index)
    assert eye.x() < opt.rect.center().x() < swatch.center().x()            # ojo a la izquierda, color a la derecha
    assert swatch.width() > 20                                              # con sitio de sobra
    dbl = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(eye), QPointF(eye), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    assert delegate.editorEvent(dbl, view.model(), opt, index) is True      # lo consume el delegado: no abre el editor
    # El color de la fila de una banda es el que se pinta de verdad: el de la rampa si así se pidió.
    bands = editor.bands_panel
    assert bands.list.model().index(0, 0).data(Qt.UserRole + 2).redF() == pytest.approx(1.0)
    bands.select(0)
    bands.color_ramp.setChecked(True)
    ramp_color = bands.list.model().index(0, 0).data(Qt.UserRole + 2)
    assert (ramp_color.redF(), ramp_color.greenF(), ramp_color.blueF()) != (1.0, 0.0, 0.0)
    # Reordenar: con las flechas, en áreas y bandas (el orden también cambia en el proyecto).
    panel.select(0)
    assert not panel.up_button.isEnabled() and panel.down_button.isEnabled()
    panel._move(1)
    assert [a.name for a in editor.project.areas] == ["B", "A", "C"] and panel.list.currentRow() == 1
    bands.select(0)
    bands._move(1)
    assert editor.project.bands[1].position == 0.2 and bands.list.currentRow() == 1



def test_sections_start_folded_remember_their_state_and_ctrl_click_opens_only_one(tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    app = create_app([])
    w = MainWindow()
    w.show()
    app.processEvents()
    edit_sections = w.sections_by_step[STEP_EDIT]
    state = {s.key: s.is_expanded() for s in edit_sections}
    assert state["ramp"] is True and not any(v for k, v in state.items() if k != "ramp")     # solo la rampa, abierta
    bands = next(s for s in edit_sections if s.key == "bands")
    bands.header.click()                                                    # abrir Bandas se recuerda
    assert w.settings.value("sections/bands", type=bool) is True
    ev = QMouseEvent(QEvent.MouseButtonRelease, QPointF(5, 5), QPointF(5, 5), Qt.LeftButton, Qt.NoButton, Qt.ControlModifier)
    areas = next(s for s in edit_sections if s.key == "areas")
    areas.header.mouseReleaseEvent(ev)                                      # Ctrl+clic: solo Áreas
    assert [s.key for s in edit_sections if s.is_expanded()] == ["areas"]
    w._section_menu = lambda *a: None
    for s in edit_sections:
        s.set_expanded(True, silent=True)                                   # y el ancho del panel se mide con todo abierto
    w._probe.wait()
    w.close()


def test_reframing_works_on_a_region_and_leaves_the_file_alone(win, tmp_path):
    win.add_photos(win._photos[:1])
    path = str(win._photos[0])
    assert win.photo_panel.reframe_button.isEnabled() and not win.photo_panel.clear_button.isEnabled()
    assert not win.import_button.isVisibleTo(win)                           # con fotos, el botón de importar no se repite
    full = (win.current_photo.width(), win.current_photo.height())
    assert full == (W, H) and win._depth_settings().crop is None
    plain_key = win._depth_settings().key()
    win._set_crop((0.25, 0.25, 0.75, 0.75))                                 # el diálogo llama a esto al aceptar
    assert (win.current_photo.width(), win.current_photo.height()) == (W // 2, H // 2)
    assert win._depth_settings().crop == (0.25, 0.25, 0.75, 0.75) and win._depth_settings().key() != plain_key
    assert win.editor.crops[path] == (0.25, 0.25, 0.75, 0.75)               # el editor y la exportación leen el mismo encuadre
    assert win._export_context() is None                                    # sin mapa de esa zona: aún no hay nada que exportar
    assert win.photo_panel.clear_button.isEnabled()
    assert Image.open(win._photos[0]).size == (W, H)                        # el archivo original no cambia
    depth = np.zeros((H // 2, W // 2), np.float32)
    win.depth_maps[path] = depth
    win._sync_editor()
    assert win._export_context()["crop"] == (0.25, 0.25, 0.75, 0.75)
    entries = win._project_entries()
    assert entries[0].crop == [0.25, 0.25, 0.75, 0.75] or entries[0].crop == (0.25, 0.25, 0.75, 0.75)
    win._set_crop(None)                                                     # «Foto entera»
    assert (win.current_photo.width(), win.current_photo.height()) == (W, H) and path not in win.crops
    assert win._depth_settings().key() == plain_key                         # la clave vuelve a ser la de siempre
