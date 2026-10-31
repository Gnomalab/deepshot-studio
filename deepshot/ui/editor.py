"""Sesión de edición: un proyecto por foto, vista previa en vivo y deshacer/rehacer.

Une los paneles (rampa, base, relieve, bandas, áreas, mezcla), el motor de vista
previa y el lienzo. La ventana principal solo la crea y conecta sus señales.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QUndoCommand, QUndoStack
from PySide6.QtWidgets import QApplication, QWidget

from deepshot import area_edit, presets
from deepshot.areas import Area
from deepshot.depth import load_photo
from deepshot.paint import MaskLayer, mask_path_for
from deepshot.project import Project
from deepshot.tonemap import ramp_preset
from deepshot.ui.editor_panels import (
    AreasPanel,
    BandsPanel,
    BasePanel,
    DepthControlPanel,
    LensPanel,
    MixPanel,
    ReliefPanel,
)
from deepshot.ui.look_panel import LookPanel
from deepshot.ui.preset_bar import PresetBar
from deepshot.ui.ramp_editor import RampEditor
from deepshot.ui.render_controller import RenderController

log = logging.getLogger(__name__)


class _ProjectEdit(QUndoCommand):
    """Paso de deshacer: estado del proyecto antes y después de un gesto."""

    def __init__(self, session: "EditSession", path: str, before: dict, after: dict, text: str) -> None:
        super().__init__(text)
        self.session, self.path, self.before, self.after = session, path, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:      # al apilarlo, el cambio ya está hecho en el proyecto
            self._first = False
            return
        self.session.restore(self.path, self.after)

    def undo(self) -> None:
        self.session.restore(self.path, self.before)


class _MaskEdit(QUndoCommand):
    """Paso de deshacer de la máscara pintada: copia comprimida de la capa antes y después."""

    def __init__(self, session: "EditSession", path: str, before: bytes, after: bytes, text: str) -> None:
        super().__init__(text)
        self.session, self.path, self.before, self.after = session, path, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:      # al apilarlo, el trazo ya está pintado
            self._first = False
            return
        self.session.restore_mask(self.path, self.after)

    def undo(self) -> None:
        self.session.restore_mask(self.path, self.before)


class _ContextWatcher(QObject):
    """Filtro de eventos de la aplicación: un clic o el foco dentro del panel de áreas o de bandas avisa a la sesión de
    edición (para que solo se vean los tiradores de lo que se está tocando)."""

    def __init__(self, session: "EditSession") -> None:
        super().__init__(session)
        self._session = session

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.MouseButtonPress, QEvent.FocusIn) and isinstance(obj, QWidget):
            self._session._watch_context(obj)
        return False


class EditSession(QObject):
    resultReady = Signal(QImage, bool, float)   # imagen, es completa, milisegundos
    overlaysChanged = Signal()
    editableChanged = Signal(bool)
    userEdited = Signal()                       # el usuario cambió algún ajuste del efecto
    gestureFinished = Signal(str)               # terminó un gesto de edición de áreas (tipo)
    maskChanged = Signal()                      # cambió la máscara pintada (para repintar su vista)

    def __init__(self, undo_stack: QUndoStack, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.undo_stack = undo_stack
        self.projects: dict[str, Project] = {}
        self.path: str | None = None
        self.depth: np.ndarray | None = None
        self._focus_before: tuple[float, bool] | None = None     # foco previo mientras actúa el cuentagotas
        self.photo_size: tuple[int, int] | None = None   # (ancho, alto) de la foto
        self.result: QImage | None = None
        self._snapshot: dict | None = None               # último estado guardado en deshacer
        self.selected_area = 0
        self._gesture: dict | None = None                # gesto de área en curso (ver gesture())
        self.masks: dict[str, MaskLayer] = {}            # máscara pintada de cada foto (carga perezosa)
        self._stroke: dict | None = None                 # trazo de pincel en curso
        self._wheel_timer = QTimer(self)                 # junta los pasos de rueda en un solo deshacer
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.setInterval(500)
        self._wheel_timer.timeout.connect(self._on_committed)

        self.render = RenderController(self)
        self.render.rendered.connect(self._on_rendered)

        # Paneles
        self.base_panel = BasePanel()
        self.ramp_editor = RampEditor()
        self.relief_panel = ReliefPanel()
        self.depth_panel = DepthControlPanel()
        self.bands_panel = BandsPanel()
        self.areas_panel = AreasPanel()
        self.mix_panel = MixPanel()
        self.lens_panel = LensPanel()
        self.look_panel = LookPanel()                    # paso Look: pila de capas de ajuste
        self._warped: dict[int, tuple] = {}              # profundidad con el control de profundidad aplicado (caché)
        for panel in self.panels():
            panel.changed.connect(self._on_changed)
            panel.committed.connect(self._on_committed)
            panel.setEnabled(False)
        # Estilos completos (rampa + base + bandas + mezcla), de fábrica o propios.
        self.style_bar = PresetBar("style", self.tr("Cargar estilo…"),
                                   lambda: presets.from_project("style", self.project))
        self.style_bar.setEnabled(False)
        self.style_bar.chosen.connect(self.apply_style)
        self.lens_panel.presetChosen.connect(self.apply_lens_preset)
        # Al pasar el ratón por la lista de un preset se ve el resultado en la vista previa sin aplicarlo.
        for kind, bar in (("style", self.style_bar), ("ramp", self.ramp_editor.preset),
                          ("lens", self.lens_panel.preset_bar)):
            bar.previewed.connect(lambda preset, k=kind: self.preview_preset(k, preset))
            bar.previewEnded.connect(self.end_preview)
        self._previewing = False
        self.ramp_editor.fitRequested.connect(self.fit_ramp)
        self.depth_panel.centersRequested.connect(self.centers_from_map)
        self.areas_panel.selectionChanged.connect(self._on_area_selected)
        # Como en Blender, solo se ven los tiradores de aquello con lo que se trabaja: áreas o bandas.
        self.edit_context = "areas"
        self.areas_panel.selectionChanged.connect(lambda _row: self.set_edit_context("areas"))
        self.areas_panel.changed.connect(lambda: self.set_edit_context("areas"))
        self.areas_panel.toolChanged.connect(lambda tool: self.set_edit_context("areas") if tool else None)
        self.bands_panel.selectionChanged.connect(lambda _row: self.set_edit_context("bands"))
        self.bands_panel.selectionChanged.connect(lambda _row: self.overlaysChanged.emit())   # el tirador 3D sigue a la fila
        self.bands_panel.changed.connect(lambda: self.set_edit_context("bands"))
        # Y tocar cualquier control de uno de los dos paneles (aunque no cambie nada, p. ej. pulsar la fila ya elegida).
        self._context_watcher = _ContextWatcher(self)
        QApplication.instance().installEventFilter(self._context_watcher)
        self.mix_panel.maskOperation.connect(self.edit_mask)
        self.mix_panel.maskFileChosen.connect(self._on_mask_file)

    # =====================================================================
    # Foto actual
    # =====================================================================
    @property
    def project(self) -> Project | None:
        return self.projects.get(self.path) if self.path else None

    def panels(self) -> list:
        return [self.base_panel, self.ramp_editor, self.relief_panel, self.depth_panel, self.bands_panel,
                self.areas_panel, self.mix_panel, self.lens_panel, self.look_panel]

    def warped(self, depth: np.ndarray | None) -> np.ndarray | None:
        """`depth` con el control de profundidad del proyecto aplicado: la profundidad que de verdad leen la rampa,
        las bandas, el desenfoque y las áreas. Se guarda en caché porque se pide en cada gesto del ratón."""
        if depth is None or self.project is None:
            return depth
        control = self.project.depth_control
        if control.is_identity():
            return depth
        key = control.key()
        hit = self._warped.get(id(depth))
        if hit is None or hit[0] is not depth or hit[1] != key:
            if len(self._warped) > 4:
                self._warped.clear()
            hit = (depth, key, control.apply(depth))
            self._warped[id(depth)] = hit
        return hit[2]

    def set_photo(self, path: Path | None, depth: np.ndarray | None) -> None:
        """Cambia de foto. Sin mapa de profundidad no se puede editar."""
        self.path = str(path) if path else None
        self.depth = depth
        self.result = None
        editable = path is not None and depth is not None
        for panel in self.panels():
            panel.setEnabled(editable)
        self.style_bar.setEnabled(editable)
        self.editableChanged.emit(editable)
        if not editable:
            self.render.set_source(None, None)
            self.overlaysChanged.emit()
            return
        photo = load_photo(Path(path))
        self.photo_size = photo.size
        if self.path not in self.projects:
            self.projects[self.path] = self.new_project(depth)
        self.render.set_source(depth, photo)
        self._bind()
        self._snapshot = self.project.to_dict()
        self._request_render()
        self.overlaysChanged.emit()
        self.maskChanged.emit()

    @staticmethod
    def new_project(depth: np.ndarray) -> Project:
        """Proyecto inicial: rampa lineal (cerca blanco, lejos negro) ajustada al relieve."""
        project = Project(ramp=ramp_preset("linear"))
        project.ramp.fit_to_relief(depth)
        return project

    def _bind(self) -> None:
        p = self.project
        self.base_panel.bind(p)
        self.ramp_editor.set_ramp(p.ramp)
        self.relief_panel.bind(p)
        self.depth_panel.bind(p)
        self.bands_panel.bind(p)
        self.areas_panel.bind(p)
        self.mix_panel.bind(p)
        self.lens_panel.bind(p)
        self.look_panel.bind(p)

    # =====================================================================
    # Cambios, vista previa y deshacer
    # =====================================================================
    def preset_bars(self) -> list:
        return [self.style_bar, self.ramp_editor.preset, self.lens_panel.preset_bar]

    def preview_preset(self, kind: str, preset: dict) -> None:
        """Enseña en la vista previa cómo quedaría un preset sin aplicarlo ni apilar deshacer."""
        if self.project is None or self.depth is None:
            return
        try:
            project = Project.from_dict(presets.apply_to_project_dict(kind, self.project.to_dict(), preset))
        except (ValueError, KeyError, TypeError):
            return
        self._previewing = True
        self.render.request(project, self._mask_array())

    def end_preview(self) -> None:
        """Se acabó la vista previa de un preset: vuelve a pintarse el proyecto real."""
        if self._previewing:
            self._previewing = False
            self._request_render()

    def _on_changed(self) -> None:
        for bar in self.preset_bars():
            bar.sync()                       # el nombre del preset elegido solo se ve mientras los ajustes sean los suyos
        if self.project is not None:
            self._ensure_mask_path()
            self._request_render()
            self.overlaysChanged.emit()
            self.userEdited.emit()

    def _request_render(self) -> None:
        """Pide una vista previa con los ajustes actuales y, si hace falta, la máscara pintada."""
        if self.project is not None:
            self.render.request(self.project, self._mask_array())

    def visible_areas(self) -> list[Area]:
        """Las áreas del proyecto (las desactivadas por su ojo no se dibujan ni llevan manejadores)."""
        return self.project.areas


    def project_dict(self, path: str) -> dict | None:
        """Ajustes guardados de una foto (None si aún no se editó ninguna vez)."""
        project = self.projects.get(path)
        return project.to_dict() if project is not None else None

    def set_project_dict(self, path: str, data: dict) -> None:
        """Deja los ajustes de una foto tal como vienen de un proyecto guardado (ValueError si no valen).

        Sirve para fotos que no son la actual: al seleccionarlas, `set_photo` usará estos ajustes."""
        self.projects[path] = Project.from_dict(data)
        self.masks.pop(path, None)          # la máscara se vuelve a leer de su PNG

    def release_caches(self) -> None:
        """Suelta cachés que se rehacen solas (profundidad movida, niveles de render) y vuelve a pedir la vista."""
        self._warped.clear()
        self.render.release_caches()
        if self.path is not None and self.render.has_source():
            self._request_render()

    def forget(self, path: str) -> None:
        """Olvida los ajustes de una foto que se quitó de la biblioteca."""
        self.projects.pop(path, None)
        self.masks.pop(path, None)
        if self.path == path:
            self.set_photo(None, None)

    def _on_committed(self) -> None:
        """Fin de un gesto: si el proyecto cambió, se apila un paso de deshacer."""
        if self.project is None:
            return
        after = self.project.to_dict()
        if after != self._snapshot:
            self.undo_stack.push(_ProjectEdit(self, self.path, self._snapshot, after, self.tr("Editar ajustes")))
            self._snapshot = after

    def restore(self, path: str, data: dict) -> None:
        """Aplica un estado guardado (deshacer/rehacer)."""
        self.projects[path] = Project.from_dict(data)
        if path == self.path:
            self._snapshot = data
            self._previewing = False
            self._bind()
            for bar in self.preset_bars():
                bar.sync()                   # deshacer hasta un preset vuelve a mostrar su nombre
            self._request_render()
            self.overlaysChanged.emit()
            self.maskChanged.emit()

    def _on_rendered(self, image: QImage, full: bool, ms: float) -> None:
        self.result = image
        self.resultReady.emit(image, full, ms)

    # =====================================================================
    # Acciones
    # =====================================================================
    def fit_ramp(self) -> None:
        if self.project is None or self.depth is None:
            return
        self.project.ramp.fit_to_relief(self.warped(self.depth))
        self.ramp_editor.set_ramp(self.project.ramp)
        self._on_changed()
        self._on_committed()

    def centers_from_map(self) -> None:
        """Pone los centros de las tres zonas donde hay profundidad en este mapa (los destinos quedan iguales)."""
        if self.project is None or self.depth is None:
            return
        self.project.depth_control.set_centers_from_depth(self.depth)
        self.depth_panel.refresh()
        self._on_changed()
        self._on_committed()

    def apply_style(self, preset: dict) -> None:
        """Aplica un estilo (preset completo) al proyecto de la foto actual; se puede deshacer."""
        if self.project is None:
            return
        before = self.project.to_dict()
        after = presets.apply_to_project_dict("style", before, preset)
        self.restore(self.path, after)
        self.undo_stack.push(_ProjectEdit(self, self.path, before, after, self.tr("Aplicar estilo")))
        self.userEdited.emit()

    def apply_lens_preset(self, preset: dict) -> None:
        """Aplica un preset de bokeh (apertura, diafragma, ojo de gato…) y enciende el desenfoque; se puede deshacer.
        El foco y el destino de la foto no se tocan."""
        if self.project is None:
            return
        before = self.project.to_dict()
        after = presets.apply_to_project_dict("lens", before, preset)
        self.restore(self.path, after)
        self.undo_stack.push(_ProjectEdit(self, self.path, before, after, self.tr("Aplicar bokeh")))
        self.lens_panel.refresh()
        self.userEdited.emit()

    def place_area_at(self, u: float, v: float) -> bool:
        """Centra el área seleccionada en el punto 3D bajo el clic. Crea una si no hay."""
        p = self.project
        depth = self.warped(self.render.preview_depth())
        if p is None or depth is None:
            return False
        if not p.areas:
            self.areas_panel._add()
        area = self.areas_panel.current()
        if area is None:
            return False
        area.center = p.scene.point_at(depth, u, v)
        self.areas_panel.update_current_from_project()
        self._on_changed()
        self._on_committed()
        return True

    def _watch_context(self, widget: QWidget) -> None:
        """Un clic o el foco en `widget`: si está dentro del panel de áreas o de bandas, ese pasa a ser el contexto."""
        if self.areas_panel.isAncestorOf(widget) or widget is self.areas_panel:
            self.set_edit_context("areas")
        elif self.bands_panel.isAncestorOf(widget) or widget is self.bands_panel:
            self.set_edit_context("bands")

    def set_edit_context(self, context: str) -> None:
        """Qué se está editando: «areas» o «bands». Cambia lo que se dibuja sobre la imagen y en 3D."""
        if context != self.edit_context:
            self.edit_context = context
            self.overlaysChanged.emit()

    def visible_context(self) -> str:
        """El contexto que de verdad se enseña: sin bandas no hay nada que ver de ellas, así que valen las áreas."""
        if self.edit_context == "bands" and (self.project is None or not self.project.bands):
            return "areas"
        return self.edit_context

    def _on_area_selected(self, row: int) -> None:
        self.selected_area = row
        self.overlaysChanged.emit()

    # =====================================================================
    # Máscara pintada (pincel)
    # =====================================================================
    BRUSH_DEFAULTS = {"size": 6.0, "hardness": 0.5, "opacity": 1.0}   # tamaño: % del lado largo de la foto

    def mask_file(self) -> Path | None:
        """Archivo PNG donde se guarda la máscara pintada de la foto actual (en la carpeta de datos)."""
        return mask_path_for(self.path) if self.path is not None else None

    def mask_layer(self, create: bool = True) -> MaskLayer | None:
        """Máscara pintada de la foto actual. La lee del PNG guardado si existe; si no, empieza vacía."""
        if self.path is None or self.photo_size is None:
            return None
        layer = self.masks.get(self.path)
        if layer is None and create:
            path = Path(self.project.composite.painted_mask) if self.project.composite.painted_mask else None
            try:
                layer = MaskLayer.load(path, self.photo_size) if path is not None and path.is_file() \
                    else MaskLayer(self.photo_size)
            except OSError:
                log.warning("No se pudo leer la máscara pintada %s: se empieza una vacía", path)
                layer = MaskLayer(self.photo_size)
            self.masks[self.path] = layer
        return layer

    def _ensure_mask_path(self) -> None:
        """Si la máscara elegida es «pintada» y aún no hay archivo, se le asigna el de la foto
        (así va en el mismo paso de deshacer que el cambio de máscara)."""
        comp = self.project.composite
        if comp.mask == "painted" and not comp.painted_mask and self.mask_file() is not None:
            comp.painted_mask = str(self.mask_file())
            layer = self.mask_layer()
            if layer is not None and not Path(comp.painted_mask).is_file():
                layer.save(Path(comp.painted_mask))

    def _mask_array(self) -> np.ndarray | None:
        """Copia de la máscara (uint8) para el hilo de render, o None si no se usa la pintada."""
        if self.project is None or self.project.composite.mask != "painted":
            return None
        layer = self.mask_layer()
        return layer.data.copy() if layer is not None else None

    def brush_radius(self, size_percent: float) -> float:
        """Radio del pincel en píxeles de la foto para un tamaño en % del lado largo."""
        w, h = self.photo_size or (1, 1)
        return max(size_percent, 0.05) / 100.0 * max(w, h) / 2.0

    def paint_gesture(self, g: dict) -> None:
        """Trazo con el pincel o el borrador (viene del lienzo). Un trazo = un paso de deshacer."""
        if self.project is None or self.project.composite.mask != "painted":
            return
        layer = self.mask_layer()
        if layer is None:
            return
        erase = g["kind"] == "erase"
        params = self.mask_params()
        radius = self.brush_radius(params["size"])
        phase = g["phase"]
        if phase == "press":
            self._stroke = {"before": layer.snapshot(), "erase": erase}
            layer.begin_stroke()
        if self._stroke is None:
            return
        if phase in ("press", "move"):
            layer.stroke_to(g["x"], g["y"], radius, params["hardness"], params["opacity"], erase)
            self._request_render()
            self.maskChanged.emit()
        elif phase == "release":
            layer.stroke_to(g["x"], g["y"], radius, params["hardness"], params["opacity"], erase)
            self._finish_mask_edit(layer, self._stroke["before"],
                                   self.tr("Borrar máscara") if erase else self.tr("Pintar máscara"))
            self._stroke = None
        elif phase == "cancel":
            layer.restore(self._stroke["before"])
            self._stroke = None
            self._request_render()
            self.maskChanged.emit()

    def mask_params(self) -> dict:
        """Tamaño, dureza y opacidad del pincel (los del panel de mezcla)."""
        return self.mix_panel.brush_params()

    def edit_mask(self, operation: str, source: Path | None = None) -> None:
        """Operaciones sobre toda la máscara: "fill", "clear", "invert" o "load" (desde un PNG)."""
        layer = self.mask_layer()
        if layer is None:
            return
        before = layer.snapshot()
        if operation == "fill":
            layer.fill(1.0)
        elif operation == "clear":
            layer.fill(0.0)
        elif operation == "invert":
            layer.invert()
        elif operation == "load" and source is not None:
            layer.data = MaskLayer.load(source, self.photo_size).data
            layer.version += 1
        else:
            return
        text = {"fill": self.tr("Rellenar máscara"), "clear": self.tr("Vaciar máscara"),
                "invert": self.tr("Invertir máscara"), "load": self.tr("Cargar máscara")}[operation]
        self._finish_mask_edit(layer, before, text)

    def _on_mask_file(self, path: str) -> None:
        try:
            self.edit_mask("load", Path(path))
        except OSError as exc:
            log.warning("No se pudo abrir la máscara %s: %s", path, exc)

    def _finish_mask_edit(self, layer: MaskLayer, before: bytes, text: str) -> None:
        """Guarda la máscara en su archivo y apila el paso de deshacer."""
        self._ensure_mask_path()
        self._on_committed()               # por si el trazo asignó el archivo (cambio del proyecto)
        file = self.mask_file()
        if file is not None:
            try:
                layer.save(file)
            except OSError as exc:
                log.warning("No se pudo guardar la máscara pintada: %s", exc)
        after = layer.snapshot()
        if after != before:
            self.undo_stack.push(_MaskEdit(self, self.path, before, after, text))
        self._request_render()
        self.maskChanged.emit()

    def restore_mask(self, path: str, blob: bytes) -> None:
        """Deshacer / rehacer de la máscara."""
        layer = self.masks.get(path)
        if layer is None:
            return
        layer.restore(blob)
        if path == self.path:
            file = self.mask_file()
            if file is not None:
                try:
                    layer.save(file)
                except OSError:
                    pass
            self._request_render()
            self.maskChanged.emit()

    # =====================================================================
    # Áreas sobre la imagen
    # =====================================================================
    DEFAULT_DRAW_SCALE = (0.25, 0.25, 0.15)    # tamaño de un área al dibujarla con un simple clic

    def geometry(self) -> area_edit.ImageGeometry | None:
        """Foto, cámara y profundidad: lo que hace falta para pasar de píxeles a la escena."""
        p = self.project
        if p is None or self.photo_size is None:
            return None
        return area_edit.ImageGeometry(self.photo_size[0], self.photo_size[1], p.scene,
                                       self.warped(self.render.preview_depth()))

    def overlays(self) -> list[dict]:
        """Contornos de las áreas sobre la imagen, en píxeles de la foto (ver area_edit.outline).

        Es una proyección aproximada (elipse o rectángulo con los semiejes X e Y y el giro Z),
        suficiente para ver dónde está cada área; la forma exacta la da el render.
        """
        geo = self.geometry()
        if geo is None:
            return []
        result = []
        for i, a in enumerate(self.project.areas):
            if not a.enabled:
                continue
            result.append({**area_edit.outline(a, geo), "index": i, "selected": i == self.selected_area,
                           "strength": a.strength, "soft": a.edge_softness, "name": a.name,
                           "color": a.display_color(i)})
        return result

    def brush(self, tool: str) -> dict | None:
        """Vista previa del tamaño junto al puntero para las herramientas «dibujar» y «colocar»."""
        geo = self.geometry()
        if geo is not None and tool in ("paint", "erase"):
            radius = self.brush_radius(self.mask_params()["size"])
            return {"cx": 0.0, "cy": 0.0, "rx": radius, "ry": radius, "angle": 0.0, "box": False,
                    "soft": 1.0 - self.mask_params()["hardness"], "color": (1.0, 1.0, 1.0)}
        if geo is None or tool not in ("draw", "place"):
            return None
        if tool == "place" and self.areas_panel.current() is not None:
            area = self.areas_panel.current()
        else:
            area = Area(center=(0.0, 0.0, 0.3), scale=self.DEFAULT_DRAW_SCALE)
        out = area_edit.outline(area, geo)
        out.update(soft=area.edge_softness, color=(1.0, 1.0, 1.0))
        return out

    def gesture(self, g: dict) -> None:
        """Gesto del ratón o del teclado sobre un área (viene del lienzo, ver CanvasView.gesture).

        press crea o elige el área y guarda cómo era (`area0`); move la recalcula siempre desde ese
        estado inicial; release apila un único paso de deshacer; cancel la deja como estaba.
        """
        phase, kind = g["phase"], g["kind"]
        if kind in ("paint", "erase"):
            self.paint_gesture(g)
            return
        if kind == "pick_focus":
            self.pick_focus_gesture(g)
            return
        geo = self.geometry()
        if self.project is None or geo is None:
            return
        pos = (g["x"], g["y"])
        shift = bool(g.get("mods", 0) & int(Qt.ShiftModifier.value))
        ctrl = bool(g.get("mods", 0) & int(Qt.ControlModifier.value))

        if phase == "press":
            self._begin_gesture(kind, g.get("index", -1), pos)
        st = self._gesture
        if st is None:
            return
        if phase == "cancel":
            self._cancel_gesture()
            return
        if phase == "release":
            self._finish_gesture(pos, geo)
            return
        st["axis"] = g.get("axis", "")
        self._apply_gesture(st, pos, geo, shift, ctrl)

    FOCUS_SAMPLE = 2          # el cuentagotas promedia (2·n+1)² píxeles: un punto de ruido no engaña al foco

    def depth_at(self, x: float, y: float) -> float | None:
        """Profundidad (mediana de un entorno pequeño) bajo un punto de la foto, en píxeles de la foto."""
        if self.depth is None:
            return None
        h, w = self.depth.shape
        cx, cy = min(max(int(x), 0), w - 1), min(max(int(y), 0), h - 1)
        n = self.FOCUS_SAMPLE
        patch = self.depth[max(cy - n, 0): cy + n + 1, max(cx - n, 0): cx + n + 1]
        # El desenfoque lee la profundidad ya movida: el foco debe medirse en esa misma escala.
        return float(np.median(self.project.depth_control.apply(patch) if self.project is not None else patch))

    def pick_focus_gesture(self, g: dict) -> None:
        """Cuentagotas del foco: el valor sigue al puntero mientras se pulsa y al soltar queda un paso de deshacer.
        Cancelar (Esc) devuelve el foco que había."""
        if self.project is None:
            return
        lens, phase = self.project.lens, g["phase"]
        if phase == "press":
            self._focus_before = (lens.focus, lens.enabled)
        if phase in ("press", "move", "release"):
            value = self.depth_at(g["x"], g["y"]) if "x" in g else None      # el visor 3D puede soltar fuera de la superficie
            if value is not None:
                lens.focus, lens.enabled = round(value, 4), True
                self.lens_panel.refresh()
                self._on_changed()
        if phase == "release":
            self._on_committed()
            self.gestureFinished.emit("pick_focus")
        elif phase == "cancel" and getattr(self, "_focus_before", None) is not None:
            lens.focus, lens.enabled = self._focus_before
            self.lens_panel.refresh()
            self._on_changed()
            self.gestureFinished.emit("pick_focus")

    def gesture3d(self, g: dict) -> None:
        """Gesto sobre un área en la vista 3D (viene de `View3D.areaGesture`).

        Los valores nuevos (centro, semiejes, giro) ya vienen calculados por el visor a partir del área tal como
        era al pulsar. Mismo ciclo que en 2D: press elige el área, move la cambia, release apila un único paso de
        deshacer (un clic que solo selecciona no apila nada) y cancel la deja como estaba.
        """
        if self.project is None:
            return
        phase = g["phase"]
        if phase == "press":
            self._begin_gesture("edit3d", g["index"], (0.0, 0.0))
            return
        st = self._gesture
        if st is None or st["kind"] != "edit3d":
            return
        if phase == "move":
            area = self.project.areas[st["index"]]
            area.center, area.scale, area.rotation = tuple(g["center"]), tuple(g["scale"]), tuple(g["rotation"])
            st["moved"] = True
            self.areas_panel.sync_current()
            self._on_changed()
        elif phase == "release":
            self._finish_gesture((0.0, 0.0), None)
        elif phase == "cancel":
            self._cancel_gesture()

    def band_gesture3d(self, g: dict) -> None:
        """Gesto sobre los tiradores de una banda en la vista 3D (viene de `View3D.bandGesture`). Mismo ciclo que en las
        áreas: press recuerda la banda, move le pone los valores nuevos, release apila un único paso de deshacer y
        cancel la deja como estaba."""
        if self.project is None or not 0 <= g["index"] < len(self.project.bands):
            return
        band, phase = self.project.bands[g["index"]], g["phase"]
        if phase == "press":
            self.bands_panel.select(g["index"])
            self._gesture = {"kind": "band3d", "index": g["index"], "band0": copy.deepcopy(band), "moved": False}
            return
        st = self._gesture
        if st is None or st["kind"] != "band3d":
            return
        if phase == "move":
            band.position, band.width, band.pan, band.tilt = g["position"], g["width"], g["pan"], g["tilt"]
            st["moved"] = True
            self.bands_panel.sync_current()
            self._on_changed()
        elif phase == "release":
            self._gesture = None
            if st["moved"]:
                self._on_committed()
        elif phase == "cancel":
            self._gesture = None
            b0 = st["band0"]
            band.position, band.width, band.pan, band.tilt = b0.position, b0.width, b0.pan, b0.tilt
            self.bands_panel.sync_current()
            self._on_changed()

    def _begin_gesture(self, kind: str, index: int, pos: tuple[float, float]) -> None:
        panel, areas = self.areas_panel, self.project.areas
        created = False
        if kind == "draw" or (kind == "place" and not areas):
            panel.add_item(Area(name=self.tr("Área {0}").format(len(areas) + 1), center=(0.0, 0.0, 0.3),
                                scale=self.DEFAULT_DRAW_SCALE), commit=False)
            index, created = len(areas) - 1, True
        elif not 0 <= index < len(areas):
            index = panel.list.currentRow()
        if not 0 <= index < len(areas):
            self._gesture = None
            return
        if index != self.selected_area or panel.list.currentRow() != index:
            panel.select(index)
        self._gesture = {"kind": kind, "index": index, "press": pos, "area0": copy.deepcopy(areas[index]),
                         "created": created, "axis": "", "moved": False}

    def _apply_gesture(self, st: dict, pos: tuple[float, float], geo, shift: bool, ctrl: bool) -> None:
        kind, area0, axis, press = st["kind"], st["area0"], st["axis"], st["press"]
        moved = (pos[0] - press[0]) ** 2 + (pos[1] - press[1]) ** 2 > 9.0     # más de 3 px
        st["moved"] = st["moved"] or moved
        if kind not in ("draw", "place") and not st["moved"]:
            return          # zona muerta de 3 px: un simple clic solo selecciona, no modifica el área
        if kind == "move":
            new = area_edit.move(area0, press, pos, geo, snap_to_surface=shift, axis=axis)
        elif kind in ("scale_x", "scale_y"):
            new = area_edit.scale_handle(area0, kind[-1], pos, geo, uniform=shift)
        elif kind == "scale_all":
            new = area_edit.scale_all(area0, press, pos, geo, axis=axis)
        elif kind == "rotate":
            new = area_edit.rotate_handle(area0, pos, geo, snap=ctrl)
        elif kind in ("pan", "tilt"):
            new = area_edit.pan_tilt(area0, kind, press, pos, snap=ctrl)
        elif kind == "rotate_free":
            new = area_edit.rotate_free(area0, press, pos, geo, axis=axis or "z", snap=ctrl)
        elif kind == "draw":
            if st["moved"]:
                new = area_edit.draw(area0, press, pos, geo, free=shift)
            else:       # un simple clic: área de tamaño por defecto, pegada a la superficie
                hit = geo.surface_point(*press)
                new = replace(area0, center=hit if hit is not None else area0.center,
                              scale=self.DEFAULT_DRAW_SCALE)
        elif kind == "place":
            new = area_edit.place(area0, pos, geo)
        else:
            return
        area = self.project.areas[st["index"]]
        area.center, area.scale, area.rotation = new.center, new.scale, new.rotation
        self.areas_panel.sync_current()
        self._on_changed()

    def _finish_gesture(self, pos, geo) -> None:
        st, self._gesture = self._gesture, None
        self._on_committed()      # un clic que solo selecciona no cambia nada: no apila deshacer
        self.gestureFinished.emit(st["kind"])

    def _cancel_gesture(self) -> None:
        st, self._gesture = self._gesture, None
        areas = self.project.areas
        if st["created"]:
            del areas[st["index"]]                      # el área creada por el gesto desaparece
            self.areas_panel.refresh_list(select=len(areas) - 1)
        elif 0 <= st["index"] < len(areas):
            a0 = st["area0"]
            area = areas[st["index"]]
            area.center, area.scale, area.rotation = a0.center, a0.scale, a0.rotation
            self.areas_panel.sync_current()
        self._on_changed()
        self.gestureFinished.emit(st["kind"])

    def adjust_selected(self, what: str, steps: float) -> None:
        """Ajuste con la rueda sobre la imagen: "scale" (todo el tamaño), "depth" (posición en Z) o
        "softness" (suavidad del borde). Los pasos seguidos forman un solo paso de deshacer."""
        area = self.areas_panel.current()
        if self.project is None or area is None or not steps:
            return
        if what == "scale":
            k = 1.06 ** steps
            area.scale = tuple(max(v * k, area_edit.MIN_SCALE) for v in area.scale)
        elif what == "depth":
            x, y, z = area.center
            area.center = (x, y, z + 0.02 * steps)
        elif what == "softness":
            area.edge_softness = min(max(area.edge_softness + 0.05 * steps, 0.0), 1.0)
        else:
            return
        self.areas_panel.sync_current()
        self._on_changed()
        self._wheel_timer.start()

    def shutdown(self) -> None:
        """Detiene la vista previa en segundo plano (al cerrar la ventana)."""
        QApplication.instance().removeEventFilter(self._context_watcher)
        self.render.shutdown()

    # =====================================================================
    # Guardar y cargar ajustes
    # =====================================================================
    def save_settings(self, path: Path) -> None:
        if self.project is not None:
            self.project.save(path)
            log.info("Ajustes guardados: %s", path)

    def load_settings(self, path: Path) -> None:
        """Carga un JSON de ajustes en la foto actual (se puede deshacer)."""
        if self.project is None:
            return
        data = Project.load(path).to_dict()
        before = self.project.to_dict()
        self.restore(self.path, data)
        self.undo_stack.push(_ProjectEdit(self, self.path, before, data, self.tr("Cargar ajustes")))
        log.info("Ajustes cargados: %s", path)
