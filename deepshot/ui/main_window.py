"""Ventana principal: barra de pasos, biblioteca, lienzo, ajustes y registro."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QByteArray, QCoreApplication, QEvent, QObject, QSettings, QSize, Qt, QT_TRANSLATE_NOOP, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QAction, QIcon, QImage, QImageReader, QKeySequence, QPixmap, QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import numpy as np
from PIL import Image

from deepshot import (APP_NAME, AUTHOR, ORG_NAME, RELEASE_DATE, WEBSITE, __version__, depth_runs, depth_view, memory,
                      presets, project_file, themes)
from deepshot import area3d, framing, modules
from deepshot.paint import mask_path_for
from deepshot.project import Project
from deepshot.dof import focus_map
from deepshot.depth import (
    DepthRun,
    read_timing,
    DEFAULT_QUALITY,
    QUALITY_PRESETS,
    RESOLUTION_CHOICES,
    settings_for,
    DepthEstimator,
    DepthSettings,
    cache_path,
    load_depth_png,
    load_external_depth,
)
from deepshot.ui.canvas import CanvasView, depth_to_qimage
from deepshot.ui.depth_worker import DepthWorker
from deepshot.models import DEFAULT_MODEL, ModelError, ModelManager, can_compute
from deepshot.ui.editor import EditSession
from deepshot.ui.crop_dialog import CropDialog
from deepshot.ui.export_panel import ExportPanel
from deepshot.ui.photo_panel import PhotoInfoPanel
from deepshot.ui.help import GuideDialog, ShortcutsDialog
from deepshot.ui import layouts, theme
from deepshot.paths import resource
from deepshot.ui.icons import icon, refresh_icons, set_icon
from deepshot.ui.preferences import PreferencesDialog
from deepshot.ui.log_panel import LogPanel
from deepshot.ui.simple_depth_dialog import SimpleDepthDialog
from deepshot.ui.start_dialog import StartDialog
from deepshot.ui.task_timer import TaskTimer, with_time
from deepshot.timing import Durations
from deepshot import view3d as view3d_math
from deepshot.ui.first_run import FirstRunDialog, needs_first_run
from deepshot.ui.memory_indicator import MemoryIndicator
from deepshot.ui.view3d_widget import View3D
from deepshot.ui.model_manager import ModelManagerDialog
from deepshot.ui.widgets import CollapsibleSection, HoverHelp, LibraryList, SegmentedControl

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
IMAGE_FILTER = "Imágenes (*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp)"
DEPTH_FILTER = "Mapas de profundidad (*.png *.tif *.tiff)"
VIEW_PHOTO, VIEW_DEPTH, VIEW_RESULT, VIEW_COMPARE = 0, 1, 2, 3
MODE_2D, MODE_3D = 0, 1        # modo de la vista Efecto: imagen plana o relieve en 3D
STEP_IMPORT, STEP_DEPTH, STEP_EDIT, STEP_LOOK, STEP_EXPORT = 0, 1, 2, 3, 4
STEPS = [QT_TRANSLATE_NOOP("MainWindow", "Importar"), QT_TRANSLATE_NOOP("MainWindow", "Profundidad"),
         QT_TRANSLATE_NOOP("MainWindow", "Enfoque y expresión"), QT_TRANSLATE_NOOP("MainWindow", "Look"),
         QT_TRANSLATE_NOOP("MainWindow", "Exportar")]
STEP_ICONS = ["import", "layers", "sliders", "look", "export"]
LOGO_SIZE = 44                      # lado del logo de la cabecera, en píxeles
STEP_TIPS = [QT_TRANSLATE_NOOP("MainWindow", "Importar y elegir la fotografía"), QT_TRANSLATE_NOOP("MainWindow", "Calcular o cargar el mapa de profundidad"),
             QT_TRANSLATE_NOOP("MainWindow", "Dar forma al efecto: rampa, bandas, áreas y mezcla"),
             QT_TRANSLATE_NOOP("MainWindow", "Acabado opcional: niveles, contraste, saturación, grano, nitidez… en el orden que quieras"),
             QT_TRANSLATE_NOOP("MainWindow", "Guardar el resultado o el mapa de profundidad")]
THUMB_SIZE = 96
RECENT_MAX = 10                       # proyectos recientes que se recuerdan en Archivo
STYLE_EXTENSION = ".deepshotstyle"      # archivo suelto con un estilo (JSON)


def load_image(path: Path) -> QImage:
    """Carga una imagen respetando la orientación EXIF de las fotos de cámara/móvil."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    return reader.read()


class _RunsClickFilter(QObject):
    """Lista de cálculos en la vista Comparar: clic izquierdo = ese cálculo a la izquierda (A), derecho = a la derecha (B).

    Es un filtro de eventos aparte (no una subclase del QListWidget). El clic izquierdo sigue seleccionando como
    siempre (para «Usar este mapa» o «Quitar»); el derecho solo reparte el lado y no cambia la selección."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease) and self._window._comparing_view():
            if kind == QEvent.MouseButtonPress:
                item = self._window.runs_list.itemAt(event.position().toPoint())
                if item is not None and event.button() == Qt.LeftButton:
                    self._window._assign_run_side("A", item)
                elif item is not None and event.button() == Qt.RightButton:
                    self._window._assign_run_side("B", item)
            if event.button() == Qt.RightButton:
                return True                                   # el clic derecho no selecciona ni abre menús
        return False


class _DeviceProbe(QObject):
    """Detecta el dispositivo de inferencia en un hilo aparte (importar torch tarda)."""

    finished = Signal(str)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self, timeout: float | None = None) -> None:
        """Espera a que termine la detección (los tests lo usan antes de cerrar)."""
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            from deepshot.device import detect_device

            info = detect_device()
            vram = f" · {info.vram_gb} GB" if info.vram_gb else ""
            text = f"{info.kind.upper()} · {info.name}{vram}"
        except Exception as exc:  # torch ausente o mal instalado
            log.error("No se pudo detectar el dispositivo de inferencia: %s", exc)
            text = self.tr("Sin motor de inferencia")
        try:
            self.finished.emit(text)
        except RuntimeError:
            pass  # la ventana se cerró antes de terminar la detección


class _Preloader(QObject):
    """«Cargar al arrancar»: carga un modelo en la GPU en un hilo aparte (tarda segundos y no debe bloquear la ventana)."""

    done = Signal(str, float)       # nombre del modelo y segundos que tardó
    failed = Signal(str, str)       # nombre del modelo y motivo

    def start(self, estimator, model_id: str, label: str) -> None:
        self._thread = threading.Thread(target=self._run, args=(estimator, model_id, label), daemon=True)
        self._thread.start()

    def wait(self, timeout: float | None = None) -> None:
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout)

    def _run(self, estimator, model_id: str, label: str) -> None:
        import time

        started = time.perf_counter()
        try:
            estimator.load(model_id)          # con el cerrojo de carga: un cálculo que llegue antes espera a que acabe
            seconds = time.perf_counter() - started
            try:
                self.done.emit(label, seconds)
            except RuntimeError:
                pass
        except Exception as exc:
            log.warning("No se pudo cargar %s al arrancar: %s", model_id, exc)
            try:
                self.failed.emit(label, str(exc))
            except RuntimeError:
                pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(900, 560)
        self.setAcceptDrops(True)
        self.setDockNestingEnabled(True)

        self.settings = QSettings()
        self.undo_stack = QUndoStack(self)  # todas las ediciones pasarán por aquí
        self.durations = Durations(QSettings())  # cuánto tardó cada tarea otras veces (para estimar)
        self._depth_timer = TaskTimer(self)
        self._depth_timer_key = ""
        self._depth_message = ""
        self.ask_before_closing = True      # los tests lo apagan para poder cerrar la ventana sin diálogo
        self._saved_fingerprint = ""        # estado del proyecto la última vez que se guardó o abrió
        self._dirty_timer = QTimer(self)    # recalcula el asterisco del título sin hacerlo en cada cambio
        self._dirty_timer.setSingleShot(True)
        self._dirty_timer.setInterval(300)
        self._dirty_timer.timeout.connect(self._refresh_title)
        self.current_path: Path | None = None
        self.current_photo: QImage | None = None      # la foto con el reencuadre aplicado (lo que ven todos los pasos)
        self.full_photo: QImage | None = None         # la foto entera (solo la usa el diálogo de reencuadre)
        self.crops: dict[str, tuple] = {}             # ruta de la foto -> reencuadre previo (framing.py)
        self.project_path: Path | None = None        # archivo .deepshot abierto o guardado
        self.depth_maps: dict[str, np.ndarray] = {}  # ruta de la foto -> mapa (0 = cerca)
        self.depth_sources: dict[str, str] = {}      # ruta -> id del modelo o "external" (importado)
        self.depth_runs: dict[str, depth_runs.RunList] = {}     # ruta -> cálculos de profundidad de esa foto
        self._depth_job_settings: DepthSettings | None = None   # ajustes del cálculo en marcha
        self._runs_updating = False                  # True mientras se rellena la lista de cálculos

        # Profundidad: el modelo se carga una sola vez y se reutiliza.
        self.models = ModelManager()
        self.estimator = DepthEstimator(manager=self.models)
        self.depth_worker = DepthWorker(self.estimator, self)
        self.depth_worker.progress.connect(self._on_depth_progress)
        self.depth_worker.finished.connect(self._on_depth_finished)
        self.depth_worker.failed.connect(self._on_depth_failed)

        self.canvas = CanvasView()
        self.canvas.import_requested.connect(self.import_photos)

        # Edición: un proyecto por foto, vista previa en vivo y deshacer (ui/editor.py).
        self.editor = EditSession(self.undo_stack, self)
        self.editor.crops = self.crops                  # el editor lee el reencuadre de aquí (set_photo)
        self.editor.resultReady.connect(self._on_result)
        self.editor.overlaysChanged.connect(self._update_overlays)
        self.editor.editableChanged.connect(self._on_editable)
        self.editor.userEdited.connect(self._on_user_edit)
        for signal in (self.editor.userEdited, self.editor.maskChanged, self.undo_stack.indexChanged):
            signal.connect(lambda *_: self._dirty_timer.start())          # cambió algo que se guarda en el proyecto
        # Edición de áreas directamente sobre la imagen (ver ui/canvas.py y ui/editor.py).
        self.editor.areas_panel.toolChanged.connect(self._on_tool_changed)
        self.editor.mix_panel.paintToolChanged.connect(self._on_paint_tool_changed)
        self.editor.mix_panel.brushChanged.connect(self._refresh_brush)
        self.editor.mix_panel.showMaskChanged.connect(lambda _on: self._refresh_mask_overlay())
        self.editor.maskChanged.connect(self._refresh_mask_overlay)
        self.editor.gestureFinished.connect(self._on_gesture_finished)
        self.canvas.gesture.connect(self.editor.gesture)
        self.canvas.areaWheel.connect(self.editor.adjust_selected)
        self.canvas.hint.connect(self._on_canvas_hint)
        self.export_panel = ExportPanel(self._export_context)
        self.export_panel.exported.connect(self._on_exported)
        self.export_panel.failed.connect(self._on_export_failed)
        self._build_header()
        self._build_central()
        self._build_library_dock()
        self._build_settings_dock()
        self._build_log_dock()
        self._build_menus()
        self._build_status_bar()
        self._restore_state()
        self.set_step(0)
        self._enable_status_tips()
        self._hover_help = HoverHelp(self)
        QApplication.instance().installEventFilter(self._hover_help)
        self._update_params_label()

        self._probe = _DeviceProbe()
        self._probe.finished.connect(self.device_label.setText)
        self._probe.start()
        self._preloader = _Preloader()
        self._preloader.done.connect(lambda label, s: self.statusBar().showMessage(
            self.tr("Modelo cargado: {0} ({1:.0f} s)").format(label, s), 8000))
        self._preloader.failed.connect(lambda label, why: self.statusBar().showMessage(
            self.tr("No se pudo cargar {0} al arrancar: {1}").format(label, why), 10000))
        QTimer.singleShot(1500, self._preload_model)
        self._load_modules()
        self.editor.look_panel.modulesRequested.connect(lambda: self.open_model_manager())
        log.info("%s %s iniciado", APP_NAME, __version__)

    def _load_modules(self) -> None:
        """Da de alta los módulos instalados y aprobados (capas de Look extra). Los que no están aprobados (p. ej. uno
        copiado a mano a la carpeta de módulos) no se cargan hasta que el usuario lo confirma."""
        report = modules.register_installed()
        for module_id, why in report.failed.items():
            log.warning("El módulo %s no se cargó: %s", module_id, why)
            self.statusBar().showMessage(self.tr("El módulo «{0}» no se pudo cargar: {1}").format(module_id, why), 10000)
        if report.pending:
            QTimer.singleShot(0, lambda: self._ask_module_approval(report.pending))

    def _ask_module_approval(self, pending: list) -> None:
        names = "\n".join("• {0} (v{1}, {2})".format(i.name, i.version, i.license) for i in pending)
        answer = QMessageBox.warning(
            self, self.tr("Módulos sin aprobar"),
            self.tr("Hay módulos en la carpeta de módulos que todavía no has aprobado:\n\n{0}\n\nUn módulo es código "
                    "Python que se ejecutará en tu equipo con tus permisos. Apruébalos solo si los has escrito tú o "
                    "confías en su autor. Si dices que no, no se cargan.\n\n¿Aprobarlos y cargarlos?").format(names),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            for info in pending:
                modules.approve(info)
            modules.register_installed()

    def _preload_model(self) -> None:
        """«Cargar al arrancar» (casilla del gestor de modelos): carga en la GPU el modelo elegido, en segundo plano."""
        model_id = self.models.preload_id()
        if not model_id or self.estimator.loaded_model == model_id:
            return
        try:
            label = self.models.get(model_id).display_label
        except ModelError:
            return
        self.statusBar().showMessage(self.tr("Cargando el modelo {0}…").format(label))
        self._preloader.start(self.estimator, model_id, label)

    # =====================================================================
    # Construcción de la interfaz
    # =====================================================================
    def _build_header(self) -> None:
        """Franja superior fija: logo y nombre a la izquierda, los cuatro pasos en el centro y el crédito a la derecha."""
        bar = QToolBar(self.tr("Cabecera"))
        bar.setObjectName("HeaderBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.toggleViewAction().setVisible(False)  # no se puede ocultar desde el menú contextual

        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(12, 6, 14, 6)
        row.setSpacing(12)

        # Logo grande (SVG propio, nítido en pantallas de alta densidad).
        logo = QLabel()
        logo.setObjectName("AppLogo")
        logo.setFixedSize(LOGO_SIZE, LOGO_SIZE)
        logo.setPixmap(QIcon(str(resource("icons", "app.svg"))).pixmap(QSize(LOGO_SIZE, LOGO_SIZE),
                                                                      self.devicePixelRatioF()))
        logo.setToolTip(f"{APP_NAME} {__version__}")
        title = QLabel(APP_NAME)
        title.setObjectName("AppTitle")
        version = QLabel(f"v{__version__}")
        version.setObjectName("AppVersion")
        version.setToolTip(self.tr("Versión {0} · {1}").format(__version__, RELEASE_DATE))
        tagline = QLabel(self.tr("the art of depthing"))   # lema (TAGLINE), literal para que lupdate lo encuentre
        tagline.setObjectName("AppTagline")
        tagline_row = QHBoxLayout()
        tagline_row.setContentsMargins(0, 0, 0, 0)
        tagline_row.setSpacing(6)
        tagline_row.addWidget(tagline)
        tagline_row.addWidget(version)
        tagline_row.addStretch()
        names = QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        names.addWidget(title)
        names.addLayout(tagline_row)

        # Los cuatro pasos del flujo, en el centro de la cabecera.
        steps = QWidget()
        steps.setObjectName("StepButtons")
        steps_layout = QHBoxLayout(steps)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(2)
        self.step_group = QButtonGroup(self)
        for i, name in enumerate(STEPS):
            button = QPushButton(f"{i + 1}  {self.tr(name)}")
            set_icon(button, STEP_ICONS[i])
            button.setIconSize(QSize(20, 20))
            button.setCheckable(True)
            button.setToolTip(self.tr(STEP_TIPS[i]) + f" (Alt+{i + 1})")
            self.step_group.addButton(button, i)
            steps_layout.addWidget(button)
        self.step_group.idClicked.connect(self.set_step)

        # «0 · Rápido»: el modo sencillo (mapas de profundidad por lotes) a un clic, aparte de los cuatro pasos.
        self.quick_header_button = QPushButton(self.tr("0  Rápido"))
        set_icon(self.quick_header_button, "layers")
        self.quick_header_button.setIconSize(QSize(20, 20))
        self.quick_header_button.setToolTip(self.tr("Mapa de profundidad rápido: crea mapas de profundidad (o de normales) "
                                                    "de una o varias fotos de golpe, sin abrir el editor"))
        self.quick_header_button.clicked.connect(lambda _checked=False: self.open_simple_depth())
        quick_divider = QFrame()
        quick_divider.setFrameShape(QFrame.VLine)
        quick_divider.setFrameShadow(QFrame.Sunken)

        credit = QLabel(self.tr('By <a href="{url}" style="text-decoration:none;">{org}</a>'
                                ' · {year}').format(url=WEBSITE, org=ORG_NAME, year=RELEASE_DATE[:4]))
        credit.setObjectName("AppCredit")
        credit.setOpenExternalLinks(True)  # abre la web en el navegador del sistema
        credit.setToolTip(WEBSITE)
        row.addWidget(logo)
        row.addLayout(names)
        row.addStretch(1)
        row.addWidget(self.quick_header_button)
        row.addSpacing(6)
        row.addWidget(quick_divider)
        row.addSpacing(6)
        row.addWidget(steps)
        row.addStretch(1)
        row.addWidget(credit)
        bar.addWidget(content)
        self.addToolBar(Qt.TopToolBarArea, bar)

    def _build_central(self) -> None:
        """Barra de pasos arriba y lienzo debajo."""
        step_bar = QWidget()
        step_bar.setObjectName("StepBar")
        bar_layout = QHBoxLayout(step_bar)
        bar_layout.setContentsMargins(6, 0, 6, 0)
        bar_layout.setSpacing(0)
        # (Los cuatro pasos están ahora en la cabecera, en el centro; aquí solo van las opciones de vista.)

        # Conmutador de vista del lienzo: foto original o mapa de profundidad.
        # Conmutador de vista: foto, profundidad o las dos con divisor (comparar).
        self.view_switch = SegmentedControl([(self.tr("Foto"), "photo"), (self.tr("Profundidad"), "depth"),
                                             (self.tr("Efecto"), "effect"), (self.tr("Comparar"), "compare")])
        self.view_switch.set_tooltips([
            self.tr("Fotografía original"),
            self.tr("Mapa de profundidad (cerca = claro)"),
            self.tr("Imagen con el efecto: rampa, bandas, áreas y mezcla (es la vista de edición)"),
            self.tr("Foto y efecto a la vez, con un divisor que se arrastra (C)"),
        ])
        self.view_switch.changed.connect(lambda _i: self._show_current())
        bar_layout.addSpacing(4)
        bar_layout.addWidget(self.view_switch)
        bar_layout.addSpacing(6)
        # Orientación del divisor al comparar: vertical (izq | der) u horizontal (arriba / abajo).
        self.split_button = QPushButton("⇆")
        self.split_button.setFixedWidth(30)
        self.split_button.setStyleSheet("padding: 0;")
        self.split_button.setToolTip(self.tr("Girar el divisor (Mayús+C)"))
        self.split_button.clicked.connect(self.toggle_split_orientation)
        self.split_button.hide()
        bar_layout.addWidget(self.split_button)
        # Modo de la vista Efecto: imagen plana (2D) o relieve del mapa en 3D para colocar las áreas en el espacio.
        self.mode_switch = SegmentedControl([(self.tr("2D"), "plane"), (self.tr("3D"), "cube")])
        self.mode_switch.set_tooltips([
            self.tr("Ver el efecto como imagen plana (la tecla 3 alterna 2D y 3D)"),
            self.tr("Ver el relieve del mapa de profundidad en 3D: gira la escena para ver y colocar las áreas "
                    "en profundidad (la tecla 3 alterna 2D y 3D)"),
        ])
        self.mode_switch.changed.connect(lambda _i: self._show_current())
        self.mode_switch.hide()                      # solo se ve en la vista Efecto
        bar_layout.addWidget(self.mode_switch)
        # Imagen que cubre el relieve en el modo 3D.
        self.texture_combo = QComboBox()
        for text, key in ((self.tr("Textura: foto"), "photo"), (self.tr("Textura: profundidad"), "depth"),
                          (self.tr("Textura: efecto"), "effect")):
            self.texture_combo.addItem(text, key)
        self.texture_combo.setCurrentIndex(self.texture_combo.findData("effect"))     # por defecto, el efecto
        self._refresh_combo_icons()
        self.texture_combo.setToolTip(self.tr("Qué imagen se pega sobre el relieve en el modo 3D"))
        self.texture_combo.currentIndexChanged.connect(lambda _i: self._refresh_3d())
        self.texture_combo.hide()
        bar_layout.addWidget(self.texture_combo)
        # Calidad 3D: cuántos puntos tiene el relieve (más = menos «cuadraditos», pero se dibuja más despacio).
        self.quality3d_combo = QComboBox()
        for text, key in ((self.tr("Calidad 3D: automática"), "auto"), (self.tr("Calidad 3D: baja"), "low"),
                          (self.tr("Calidad 3D: media"), "medium"), (self.tr("Calidad 3D: alta"), "high"),
                          (self.tr("Calidad 3D: muy alta"), "ultra"), (self.tr("Calidad 3D: máxima"), "max")):
            self.quality3d_combo.addItem(text, key)
        self.quality3d_combo.setToolTip(self.tr(
            "Cuántos puntos forman el relieve en 3D. Más puntos dan una superficie más fina (sin cuadraditos) "
            "pero se dibuja más despacio al soltar el ratón. «Automática» elige la más alta que tu equipo "
            "dibuja con soltura. Al girar siempre va fluido."))
        saved = str(self.settings.value("view3d/quality", "auto"))
        self.quality3d_combo.setCurrentIndex(max(0, self.quality3d_combo.findData(saved)))
        self.quality3d_combo.currentIndexChanged.connect(self._on_quality3d_changed)
        self.quality3d_combo.hide()
        bar_layout.addWidget(self.quality3d_combo)
        # Cómo se ve el mapa en la vista Profundidad: gris, en color o con las zonas del control marcadas.
        self.depth_style_combo = QComboBox()
        for text, key in ((self.tr("Mapa: gris"), "gray"), (self.tr("Mapa: color rojo / azul"), "color:red_blue"),
                          (self.tr("Mapa: color arcoíris"), "color:rainbow"), (self.tr("Mapa: color térmico"), "color:thermal"),
                          (self.tr("Mapa: color océano"), "color:ocean"), (self.tr("Mapa: zonas de profundidad"), "zones"),
                          (self.tr("Mapa: normales (OpenGL)"), "normals:opengl"),
                          (self.tr("Mapa: normales (DirectX)"), "normals:directx")):
            self.depth_style_combo.addItem(text, key)
        self.depth_style_combo.setToolTip(self.tr(
            "Cómo se muestra el mapa de profundidad. Solo cambia lo que ves: el mapa real no se toca. «Zonas» marca "
            "con un color cada zona del control de profundidad (cerca azul, medio verde, lejos naranja). «Normales» muestra el "
            "mapa de normales que sale del mapa de profundidad (el mismo que crea el Mapa rápido)."))
        saved_style = str(self.settings.value("depthview/style", "gray"))
        self.depth_style_combo.setCurrentIndex(max(0, self.depth_style_combo.findData(saved_style)))
        self.depth_style_combo.currentIndexChanged.connect(self._on_depth_style_changed)
        self.depth_style_combo.hide()
        bar_layout.addWidget(self.depth_style_combo)
        # Vista Comparar: qué va a cada lado del divisor. Por defecto Foto | Efecto, pero cada lado puede ser la foto,
        # el efecto, el mapa en uso o cualquiera de los cálculos de profundidad de la foto (otro modelo, otra calidad).
        self.ab_a_combo = QComboBox()
        self.ab_b_combo = QComboBox()
        for combo, side in ((self.ab_a_combo, "A"), (self.ab_b_combo, "B")):
            combo.setMaximumWidth(300)
            combo.setToolTip(self.tr("Qué se ve en el lado {0} del divisor: la foto, el efecto, el mapa en uso o "
                                     "cualquier cálculo de profundidad de esta foto").format(side))
            combo.currentIndexChanged.connect(self._on_ab_changed)
            combo.hide()
            bar_layout.addWidget(combo)
        self._ab_updating = False
        self._refresh_ab()
        # Interruptor de los contornos y manejadores de las áreas sobre la imagen (atajo H).
        self.areas_button = QPushButton()
        set_icon(self.areas_button, "eye")
        self.areas_button.setCheckable(True)
        self.areas_button.setChecked(True)
        self.areas_button.setFixedWidth(34)
        self.areas_button.setToolTip(self.tr("Mostrar u ocultar los contornos y manejadores (H). Solo se ven los de lo "
                                             "que estás editando: áreas o bandas, según el último panel que toques. "
                                             "No cambia el efecto ni lo que se exporta."))
        self.areas_button.toggled.connect(lambda _on: self._update_overlays())
        bar_layout.addStretch()                      # el interruptor de áreas, a la derecha
        bar_layout.addWidget(self.areas_button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(step_bar)
        # El visor 2D y el 3D comparten el sitio: solo se ve uno.
        self.view3d = View3D()
        self.view3d.set_quality(self.quality3d_combo.currentData())     # la guardada en los ajustes
        self.view3d.areaGesture.connect(self.editor.gesture3d)
        self.view3d.bandGesture.connect(self.editor.band_gesture3d)
        self.view3d.toolGesture.connect(self.editor.gesture)       # Dibujar y Colocar sobre el relieve: el mismo gesto que en 2D
        self.view3d.pickGesture.connect(lambda g: self.editor.pick_focus_gesture({**g, "kind": "pick_focus"}))
        self.editor.lens_panel.toolChanged.connect(self._on_lens_tool_changed)
        self.editor.lens_panel.showFocusChanged.connect(lambda _on: self._refresh_focus_overlay())
        self.view3d.setToolTip(self.tr("Relieve en 3D. Arrastrar: girar · Mayús+arrastrar o botón central: mover · "
                                       "Rueda: zoom · Doble clic: restablecer · Teclas 1, 2 y 7: vistas"))
        self.viewer_stack = QStackedWidget()
        self.viewer_stack.addWidget(self.canvas)
        self.viewer_stack.addWidget(self.view3d)
        layout.addWidget(self.viewer_stack, 1)
        self.setCentralWidget(central)

    def _build_library_dock(self) -> None:
        self.library = LibraryList()
        self.library.setViewMode(QListWidget.IconMode)
        self.library.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.library.setGridSize(QSize(THUMB_SIZE + 16, THUMB_SIZE + 26))
        self.library.setResizeMode(QListWidget.Adjust)
        self.library.setMovement(QListWidget.Static)
        self.library.setWordWrap(True)
        self.library.setToolTip(self.tr("Fotografías abiertas. Doble clic en el espacio vacío: importar. "
                                        "Supr: quitar la seleccionada de la lista (no borra el archivo)."))
        self.library.currentItemChanged.connect(self._on_library_selection)
        self.library.importRequested.connect(self.import_photos)
        self.library.removeRequested.connect(self.remove_photos)

        add = QPushButton()
        set_icon(add, "plus")
        add.setToolTip(self.tr("Importar fotografías…"))
        add.clicked.connect(self.import_photos)
        self.library_remove = QPushButton()
        set_icon(self.library_remove, "trash")
        self.library_remove.setToolTip(self.tr("Quitar de la biblioteca las fotos seleccionadas "
                                               "(no borra los archivos)"))
        self.library_remove.clicked.connect(lambda: self.remove_photos(self.library.selected_paths()))
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 4)
        row.setSpacing(4)
        row.addWidget(add)
        row.addWidget(self.library_remove)
        row.addStretch()
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self.library, 1)
        column.addLayout(row)

        self.library_dock = self._make_dock(self.tr("Biblioteca"), "LibraryDock", box)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.library_dock)

    def _build_settings_dock(self) -> None:
        """Panel de ajustes con secciones plegables; cada paso muestra las suyas.

        En la fase 0 son marcadores: los controles reales llegan en las fases 1-5.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        def section(title: str, *widgets: QWidget, key: str = "", open_: bool = False) -> CollapsibleSection:
            """Una sección del panel. `key` identifica su estado guardado (abierta o cerrada); `open_` es el de la primera
            vez: por defecto cerradas, para que el panel no obligue a desplazarse tanto."""
            saved = self.settings.value(f"sections/{key}", None) if key else None
            expanded = open_ if saved is None else str(saved).lower() in ("true", "1")
            s = CollapsibleSection(title, expanded=expanded)
            s.key = key
            if key:
                s.header.toggled.connect(lambda on, k=key: self.settings.setValue(f"sections/{k}", on))
            for w in widgets:
                s.add_widget(w)
            layout.addWidget(s)
            return s

        def hint(text: str) -> QLabel:
            label = QLabel(text)
            label.setWordWrap(True)
            return label

        self.photo_panel = PhotoInfoPanel()
        self.photo_panel.reframeRequested.connect(self.reframe_photo)
        self.photo_panel.cropCleared.connect(lambda: self._set_crop(None))
        # El botón de importar solo sale con la biblioteca vacía: con fotos, el «+» de la biblioteca ya lo hace.
        self.import_button = QPushButton(self.tr("Importar fotografía…"))
        self.import_button.setToolTip(self.tr("Abre una o varias fotografías (también puedes arrastrarlas a la ventana)"))
        self.import_button.clicked.connect(self.import_photos)

        # --- Profundidad: calcular dentro de la app o importar un mapa hecho fuera ---
        self.model_combo = QComboBox()
        self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(10)
        self.model_combo.setToolTip(self.tr("Modelo con el que se calcula la profundidad"))
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        manage_button = QPushButton("…")
        manage_button.setFixedWidth(28)
        manage_button.setStyleSheet("padding: 0;")  # sin relleno: si no, los puntos se recortan
        manage_button.setToolTip(self.tr("Gestor de modelos y módulos: descargar, instalar y borrar"))
        manage_button.clicked.connect(lambda _checked=False: self.open_model_manager())
        model_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(4)
        model_row_layout.addWidget(self.model_combo, 1)
        model_row_layout.addWidget(manage_button)
        self._fill_model_combo()

        # Calidad: resolución de proceso y nº de pasadas (presets medidos en depth.py).
        self.quality_combo = QComboBox()
        for preset in QUALITY_PRESETS.values():
            self.quality_combo.addItem(QCoreApplication.translate("Depth", preset.label), preset.id)
            self.quality_combo.setItemData(self.quality_combo.count() - 1, QCoreApplication.translate("Depth", preset.hint),
                                           Qt.ToolTipRole)
        saved_quality = self.settings.value("depth/quality", DEFAULT_QUALITY)
        self.quality_combo.setCurrentIndex(max(0, self.quality_combo.findData(saved_quality)))
        self.quality_combo.setToolTip(self.tr("Calidad del mapa: más resolución y pasadas = más detalle, "
                                              "pero más tiempo y memoria de vídeo"))
        self.quality_hint = hint("")
        self.quality_hint.setProperty("dim", True)
        self.quality_combo.currentIndexChanged.connect(self._on_depth_settings_changed)

        # Resolución máxima: sustituye la del preset (lado mayor con el que trabaja el modelo).
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem(self.tr("Resolución: la de la calidad"), 0)
        for px in RESOLUTION_CHOICES:
            self.resolution_combo.addItem(self.tr("Resolución: {0} px").format(px), px)
        saved_res = self.settings.value("depth/resolution", 0, type=int)
        self.resolution_combo.setCurrentIndex(max(0, self.resolution_combo.findData(saved_res)))
        self.resolution_combo.setToolTip(self.tr(
            "Lado mayor con el que el modelo calcula el mapa. Más alto = más detalle, más tiempo y más memoria de vídeo; "
            "más bajo = más rápido. «La de la calidad» usa la del preset."))
        self.resolution_combo.currentIndexChanged.connect(self._on_depth_settings_changed)

        # Refinado: ajusta los bordes del mapa a los de la foto (deepshot/refine.py).
        self.refine_check = QCheckBox(self.tr("Refinar bordes con la foto"))
        self.refine_check.setToolTip(self.tr(
            "Alinea los bordes del mapa con los de la foto a resolución completa (guided filter). "
            "Tarda unos segundos más. Funciona mejor con fotos nítidas."))
        self.refine_check.setChecked(self.settings.value("depth/refine", False, type=bool))
        self.refine_check.toggled.connect(self._on_depth_settings_changed)
        self._update_quality_hint()

        self.depth_button = QPushButton(self.tr("Calcular profundidad"))
        self.depth_button.setObjectName("Primary")
        self.depth_button.setToolTip(self.tr("Calcula el mapa de la foto seleccionada (Ctrl+D)"))
        self.depth_button.clicked.connect(self.compute_depth)

        self.depth_progress = QProgressBar()
        self.depth_progress.setTextVisible(False)
        self.depth_progress.setFixedHeight(6)
        self.depth_progress.hide()
        self.depth_status = hint("")
        # Cada segundo se refresca el tiempo transcurrido y estimado junto al mensaje.
        self._depth_timer.changed.connect(
            lambda _t: self.depth_status.setText(with_time(self._depth_message, self._depth_timer)))
        self.depth_status.hide()

        # Cálculos de esta foto: tiempo real de cada modelo / calidad, para compararlos y elegir con cuál seguir.
        self.depth_result_label = hint("")
        self.depth_result_label.setProperty("tone", "ok")
        self.depth_result_label.hide()
        self.runs_list = QListWidget()
        self.runs_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.runs_list.setFixedHeight(120)
        self.runs_list.setToolTip(self.tr(
            "Cálculos de profundidad de esta foto con lo que tardó cada uno. ● = el que se está usando. "
            "Selecciona uno para verlo; selecciona dos (Ctrl+clic) para compararlos con el divisor. "
            "En la vista Comparar: clic izquierdo = lo pone a la izquierda (A); clic derecho = a la derecha (B)."))
        self.runs_list.itemSelectionChanged.connect(self._on_runs_selection_changed)
        self._runs_clicks = _RunsClickFilter(self)                 # en Comparar: clic izquierdo = A, derecho = B
        self.runs_list.viewport().installEventFilter(self._runs_clicks)
        self.runs_hint = hint("")
        self.runs_hint.setProperty("dim", True)
        self.runs_use_button = QPushButton(self.tr("Usar este mapa"))
        self.runs_use_button.setToolTip(self.tr("Sigue el proceso (rampa, áreas, desenfoque, exportar…) con el mapa "
                                                "seleccionado"))
        self.runs_use_button.clicked.connect(self._use_selected_run)
        self.runs_remove_button = QPushButton(self.tr("Quitar"))
        self.runs_remove_button.setToolTip(self.tr("Quita de la lista los cálculos seleccionados (no se puede quitar el "
                                                   "que se está usando)"))
        self.runs_remove_button.clicked.connect(self._remove_selected_runs)
        self.runs_keep_button = QPushButton(self.tr("Quedarme solo con este"))
        self.runs_keep_button.setToolTip(self.tr("Usa el cálculo seleccionado y quita todos los demás de la lista"))
        self.runs_keep_button.clicked.connect(self._keep_only_selected_run)
        runs_buttons = QWidget()
        runs_buttons_layout = QHBoxLayout(runs_buttons)
        runs_buttons_layout.setContentsMargins(0, 0, 0, 0)
        runs_buttons_layout.setSpacing(4)
        for button in (self.runs_use_button, self.runs_remove_button):
            runs_buttons_layout.addWidget(button)
        self.runs_box = QWidget()
        runs_box_layout = QVBoxLayout(self.runs_box)
        runs_box_layout.setContentsMargins(0, 0, 0, 0)
        runs_box_layout.setSpacing(4)
        for widget in (hint(self.tr("Cálculos de esta foto")), self.runs_list, self.runs_hint, runs_buttons,
                       self.runs_keep_button):
            runs_box_layout.addWidget(widget)
        self.runs_box.hide()

        self.import_depth_button = QPushButton(self.tr("Cargar mapa de profundidad…"))
        self.import_depth_button.setToolTip(self.tr("Usa un mapa hecho en otro programa (PNG o TIFF de 16 bits)"))
        self.import_depth_button.clicked.connect(self.import_depth_map)
        self.near_bright_check = QCheckBox(self.tr("Mapa importado: cerca = claro"))
        self.near_bright_check.setToolTip(self.tr(
            "Márcalo si en el mapa que vas a cargar lo cercano es claro y lo lejano oscuro "
            "(lo más habitual). Si no, se invierte."))
        self.near_bright_check.setChecked(True)

        self.edit_hint = hint(self.tr("Calcula o carga el mapa de profundidad (paso 2) para editar."))
        self.edit_hint.setProperty("tone", "warn")

        # Qué secciones se ven en cada paso (índice de STEPS).
        self.sections_by_step: dict[int, list[CollapsibleSection]] = {
            0: [section(self.tr("Fotografía"), self.import_button, self.photo_panel, key="photo", open_=True)],
            1: [
                section(self.tr("Mapa de profundidad"), hint(self.tr("Modelo")), model_row, hint(self.tr("Calidad")),
                        self.quality_combo, self.quality_hint, self.resolution_combo, self.refine_check, self.depth_button,
                        self.depth_progress, self.depth_status, self.depth_result_label, self.runs_box, hint(self.tr("o bien")),
                        self.import_depth_button, self.near_bright_check, key="depth_map", open_=True),
            ],
            2: [
                # Relieve va aquí, donde se ve su efecto (antes estaba en el paso Profundidad).
                section(self.tr("Estilo"), hint(self.tr("Un aspecto completo de partida: rampa, bandas y mezcla. "
                                                       "Después puedes ajustarlo y guardar el tuyo. Las áreas que "
                                                       "trae un estilo se quitan al elegir otro (si no las has "
                                                       "tocado); las que dibujas tú se conservan.")),
                        self.editor.style_bar, key="style"),
                section(self.tr("Relieve"), self.editor.relief_panel, key="relief"),
                section(self.tr("Control de profundidad"), self.editor.depth_panel, key="depth_control"),
                section(self.tr("Rampa de color"), self.edit_hint, self.editor.ramp_editor, self.editor.base_panel,
                        key="ramp", open_=True),
                section(self.tr("Bandas"), self.editor.bands_panel, key="bands"),
                section(self.tr("Áreas"), self.editor.areas_panel, key="areas"),
                section(self.tr("Mezcla y máscaras"), self.editor.mix_panel, key="mix"),
                section(self.tr("Desenfoque de lente"), self.editor.lens_panel, key="lens"),
            ],
            3: [section(self.tr("Efectos"), self.editor.look_panel, key="look", open_=True)],
            4: [section(self.tr("Exportar"), self.export_panel, key="export", open_=True)],
        }
        for sections in self.sections_by_step.values():       # Ctrl+clic = solo esta; clic derecho = menú
            for sec in sections:
                sec.soloRequested.connect(lambda s=sec, group=sections: self._solo_section(s, group))
                sec.menuRequested.connect(lambda pos, s=sec, group=sections: self._section_menu(pos, s, group))
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # solo desplazamiento vertical
        scroll.setWidget(container)
        scroll.setMinimumWidth(self._settings_min_width(container, scroll))
        self.settings_dock = self._make_dock(self.tr("Ajustes"), "SettingsDock", scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)

    def _settings_min_width(self, container: QWidget, scroll: QScrollArea) -> int:
        """Ancho mínimo del panel de ajustes: el del paso con el contenido más ancho.

        Se mide mostrando cada paso; así el panel nunca puede estrecharse hasta recortar
        botones o textos (la barra horizontal está desactivada a propósito).
        """
        widest = 260
        states = [(sec, sec.is_expanded()) for secs in self.sections_by_step.values() for sec in secs]
        for sec, _state in states:                         # se mide con todo desplegado: si no, el panel saldría estrecho
            sec.set_expanded(True, silent=True)
        for step in self.sections_by_step:
            for other, sections in self.sections_by_step.items():
                for sec in sections:
                    sec.setVisible(other == step)
            container.adjustSize()
            widest = max(widest, container.minimumSizeHint().width())
        for sec, state in states:
            sec.set_expanded(state, silent=True)
        return widest + scroll.verticalScrollBar().sizeHint().width() + 2

    def _solo_section(self, section: CollapsibleSection, group: list) -> None:
        """Ctrl+clic en una cabecera: se abre esa sección y se cierran las demás del mismo paso (como en Blender)."""
        for sec in group:
            sec.set_expanded(sec is section)

    def _section_menu(self, pos, section: CollapsibleSection, group: list) -> None:
        menu = QMenu(self)
        menu.addAction(self.tr("Solo esta sección (Ctrl+clic)"), lambda: self._solo_section(section, group))
        menu.addAction(self.tr("Abrir todas"), lambda: [s.set_expanded(True) for s in group])
        menu.addAction(self.tr("Cerrar todas"), lambda: [s.set_expanded(False) for s in group])
        menu.exec(pos)

    def _build_log_dock(self) -> None:
        self.log_panel = LogPanel()
        self.log_dock = self._make_dock(self.tr("Registro"), "LogDock", self.log_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()  # oculto por defecto: no mostrar texto técnico al usuario

    def _make_dock(self, title: str, name: str, widget: QWidget) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name)  # necesario para guardar/restaurar la disposición
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                         | QDockWidget.DockWidgetClosable)
        return dock

    def _build_menus(self) -> None:
        menu = self.menuBar()

        # --- Archivo ---
        file_menu = menu.addMenu(self.tr("&Archivo"))
        self.action_import = self._action(self.tr("Importar fotografía…"), QKeySequence.Open, self.import_photos)
        file_menu.addAction(self.action_import)
        file_menu.addAction(self._action(self.tr("Cargar mapa de profundidad…"), QKeySequence("Ctrl+Shift+O"),
                                         self.import_depth_map))
        file_menu.addSeparator()
        # Proyecto: la biblioteca entera (fotos, mapas, ajustes y máscaras) para retomarla en otro momento.
        self.action_new_project = self._action(self.tr("Proyecto nuevo"), QKeySequence("Ctrl+N"), self.new_project)
        file_menu.addAction(self.action_new_project)
        self.action_open_project = self._action(self.tr("Abrir proyecto…"), QKeySequence("Ctrl+Alt+O"),
                                                self.open_project)
        self.action_save_project = self._action(self.tr("Guardar proyecto"), QKeySequence("Ctrl+S"),
                                                self.save_project)
        self.action_save_project_as = self._action(self.tr("Guardar proyecto como…"), QKeySequence("Ctrl+Shift+S"),
                                                   self.save_project_as)
        for action in (self.action_open_project, self.action_save_project, self.action_save_project_as):
            file_menu.addAction(action)
        self.action_start_screen = self._action(self.tr("Pantalla de inicio…"), None,
                                                lambda: self.show_start_dialog(show_checkbox=True))
        file_menu.addAction(self.action_start_screen)
        self.action_simple_depth = self._action(self.tr("Mapa de profundidad rápido…"), None, self.open_simple_depth)
        self.action_simple_depth.setToolTip(self.tr("Modo sencillo: solo el mapa de profundidad o de normales de unas "
                                                    "fotos, sin abrir el editor"))
        file_menu.addAction(self.action_simple_depth)
        self.action_preferences = self._action(self.tr("Preferencias…"), QKeySequence("Ctrl+,"), self.show_preferences)
        file_menu.addAction(self.action_preferences)
        for action in (self.action_save_project, self.action_save_project_as):
            action.setEnabled(False)         # hasta que haya fotos (ver _update_project_actions)
            action.setToolTip(self.tr("Disponible cuando hay fotos en la biblioteca."))
        file_menu.addSeparator()
        # Estilo: el aspecto artístico (rampa, bandas, áreas, mezcla) como archivo suelto, para otras fotos.
        self.action_save_style = self._action(self.tr("Guardar estilo…"), None, self.save_style_file)
        self.action_load_style = self._action(self.tr("Cargar estilo…"), QKeySequence("Ctrl+L"),
                                              self.load_style_file)
        file_menu.addAction(self.action_save_style)
        file_menu.addAction(self.action_load_style)
        file_menu.addSeparator()
        self.action_export = self._action(self.tr("Exportar…"), QKeySequence("Ctrl+E"),
                                          lambda: self.set_step(STEP_EXPORT))
        file_menu.addAction(self.action_export)
        # Estas necesitan una foto con mapa de profundidad: sin él se ven apagadas (ver _on_editable).
        self._needs_depth_actions = (self.action_save_style, self.action_load_style, self.action_export)
        for action in self._needs_depth_actions:
            action.setEnabled(False)
            action.setToolTip(self.tr("Disponible cuando la foto tiene mapa de profundidad."))
        for signal in (self.library.model().rowsInserted, self.library.model().rowsRemoved):
            signal.connect(lambda *_: self._update_project_actions())
            signal.connect(lambda *_: self._dirty_timer.start())
        self.recent_menu = file_menu.addMenu(self.tr("Proyectos recientes"))
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.recent_menu.setToolTip(self.tr("Los últimos proyectos abiertos o guardados."))
        file_menu.addSeparator()
        file_menu.addAction(self._action(self.tr("Salir"), QKeySequence("Ctrl+Q"), self.close))

        # --- Edición: deshacer/rehacer desde la pila común ---
        edit_menu = menu.addMenu(self.tr("&Edición"))
        undo = self.undo_stack.createUndoAction(self, self.tr("Deshacer"))
        undo.setShortcut(QKeySequence.Undo)
        redo = self.undo_stack.createRedoAction(self, self.tr("Rehacer"))
        redo.setShortcuts([QKeySequence.Redo, QKeySequence("Ctrl+Shift+Z")])
        edit_menu.addActions([undo, redo])

        # --- Ver ---
        view_menu = menu.addMenu(self.tr("&Ver"))
        view_menu.addAction(self._action(self.tr("Ajustar a la ventana"), QKeySequence("Ctrl+0"), self.canvas.fit_to_window))
        view_menu.addAction(self._action(self.tr("Tamaño real (100 %)"), QKeySequence("Ctrl+1"), self.canvas.zoom_actual_size))
        view_menu.addAction(self._action(self.tr("Acercar"), QKeySequence.ZoomIn, lambda: self.canvas.zoom(1.25)))
        view_menu.addAction(self._action(self.tr("Alejar"), QKeySequence.ZoomOut, lambda: self.canvas.zoom(0.8)))
        view_menu.addAction(self._action(self.tr("Cambiar vista (foto, profundidad, efecto)"),
                                         QKeySequence("V"), self.toggle_view))
        view_menu.addAction(self._action(self.tr("Comparar con la foto"), QKeySequence("C"), self.toggle_compare))
        view_menu.addAction(self._action(self.tr("Alternar 2D y 3D"), QKeySequence("3"), self.toggle_3d))
        view_menu.addAction(self._action(self.tr("Ver u ocultar la zona en foco"), QKeySequence("Z"),
                                         lambda: self.editor.lens_panel.show_focus.toggle()))
        view_menu.addAction(self._action(self.tr("Dibujar área arrastrando"), QKeySequence("Shift+A"),
                                         lambda: self.editor.areas_panel.draw_button.toggle()))
        view_menu.addAction(self._action(self.tr("Colocar el área seleccionada"), QKeySequence("P"),
                                         lambda: self.editor.areas_panel.place_button.toggle()))
        view_menu.addAction(self._action(self.tr("Mostrar u ocultar los contornos de las áreas"), QKeySequence("H"),
                                         lambda: self.areas_button.toggle()))
        view_menu.addAction(self._action(self.tr("Girar el divisor"), QKeySequence("Shift+C"),
                                         self.toggle_split_orientation))
        view_menu.addSeparator()
        for dock in (self.library_dock, self.settings_dock, self.log_dock):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        self.theme_menu = view_menu.addMenu(self.tr("Tema de color"))
        self.theme_menu.aboutToShow.connect(self._fill_theme_menu)
        self.layout_menu = view_menu.addMenu(self.tr("Disposición de los paneles"))
        self.layout_menu.aboutToShow.connect(self._fill_layout_menu)

        # --- Máscara: pincel de la máscara pintada (como en un programa de retoque) ---
        mask_menu = menu.addMenu(self.tr("&Máscara"))
        mask_menu.addAction(self._action(self.tr("Pincel"), QKeySequence("B"),
                                         lambda: self.editor.mix_panel.activate_paint("paint")))
        mask_menu.addAction(self._action(self.tr("Borrador"), QKeySequence("E"),
                                         lambda: self.editor.mix_panel.activate_paint("erase")))
        mask_menu.addAction(self._action(self.tr("Pincel más pequeño"), QKeySequence("["),
                                         lambda: self._change_brush_size(0.8)))
        mask_menu.addAction(self._action(self.tr("Pincel más grande"), QKeySequence("]"),
                                         lambda: self._change_brush_size(1.25)))
        mask_menu.addSeparator()
        for key, text in (("fill", self.tr("Rellenar máscara")), ("clear", self.tr("Vaciar máscara")),
                          ("invert", self.tr("Invertir máscara"))):
            mask_menu.addAction(self._action(text, None, lambda _c=False, k=key: self.editor.edit_mask(k)))

        # --- Pasos: Alt+1..5 para cambiar de paso ---
        depth_menu = menu.addMenu(self.tr("Pr&ofundidad"))
        depth_menu.addAction(self._action(self.tr("Calcular profundidad"), QKeySequence("Ctrl+D"), self.compute_depth))
        depth_menu.addSeparator()
        depth_menu.addAction(self._action(self.tr("Gestor de modelos y módulos…"), QKeySequence("Ctrl+M"),
                                          self.open_model_manager))
        depth_menu.addAction(self._action(self.tr("Añadir modelo desde Hugging Face…"), None,
                                          lambda: self.open_model_manager(then="open_search")))
        depth_menu.addAction(self._action(self.tr("Instalar modelo desde carpeta…"), None,
                                          lambda: self.open_model_manager(then="install_from_folder")))
        depth_menu.addSeparator()
        depth_menu.addAction(self._action(self.tr("Liberar memoria"), QKeySequence("Ctrl+Shift+M"), self.free_memory))

        step_menu = menu.addMenu(self.tr("&Pasos"))
        for i, name in enumerate(STEPS):
            step_menu.addAction(self._action(self.tr(name), QKeySequence(f"Alt+{i + 1}"),
                                             lambda checked=False, i=i: self.set_step(i)))

        # --- Ayuda ---
        help_menu = menu.addMenu(self.tr("Ay&uda"))
        help_menu.addAction(self._action(self.tr("Guía rápida"), QKeySequence("F1"),
                                         lambda: GuideDialog(self).exec()))
        help_menu.addAction(self._action(self.tr("Atajos de teclado"), QKeySequence("Ctrl+/"),
                                         lambda: ShortcutsDialog(self).exec()))
        help_menu.addSeparator()
        help_menu.addAction(self._action(self.tr("Visitar gnomalab.es"), None,
                                         lambda: QDesktopServices.openUrl(QUrl(WEBSITE))))
        help_menu.addAction(self._action(self.tr("Acerca de {0}").format(APP_NAME), None, self.show_about))

    def _action(self, text: str, shortcut=None, slot=None) -> QAction:
        action = QAction(text, self)
        action.setStatusTip(text.replace("&", "").rstrip("…"))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if slot is not None:
            # Qt pasa `checked` (bool) a los slots: se descarta para que no caiga en un parámetro opcional.
            action.triggered.connect(lambda _checked=False, s=slot: s())
        return action

    def _build_status_bar(self) -> None:
        # Parámetros con los que se calculará la profundidad (se actualizan al cambiarlos).
        self.params_label = QLabel()
        self.params_label.setObjectName("FooterParams")
        self.statusBar().addPermanentWidget(self.params_label)
        self.render_label = QLabel()
        self.render_label.setObjectName("FooterParams")
        self.render_label.setToolTip(self.tr("Tiempo de la última vista previa"))
        self.statusBar().addPermanentWidget(self.render_label)
        # Semáforo de memoria (VRAM y RAM libres del equipo): al pulsarlo se libera memoria.
        self.memory_button = MemoryIndicator()
        self.memory_button.clicked.connect(lambda _checked=False: self.free_memory())
        self.statusBar().addPermanentWidget(self.memory_button)
        self.device_label = QLabel(self.tr("Detectando motor de inferencia…"))
        self.statusBar().addPermanentWidget(self.device_label)
        self.statusBar().showMessage(self.tr("Listo"))

    # =====================================================================
    # Pasos
    # =====================================================================
    def set_step(self, index: int) -> None:
        """Activa un paso: marca su pestaña, muestra solo sus ajustes y la vista que le corresponde.

        Importar → foto · Profundidad → mapa · Enfoque y expresión / Look / Exportar → efecto.
        Si aún no existe lo que toca mostrar (mapa o efecto), se queda la vista actual.
        """
        self.step_group.button(index).setChecked(True)
        for step, sections in self.sections_by_step.items():
            for s in sections:
                s.setVisible(step == index)
        if not hasattr(self, "editor"):
            return
        if index != STEP_EDIT:
            # Dibujar, colocar y pincel solo tienen sentido sobre el efecto del paso 3: se sueltan al salir.
            self.editor.areas_panel.set_tool("")
            self.editor.mix_panel.set_paint_tool("")
            self.editor.lens_panel.set_pick(False)
        wanted = {STEP_IMPORT: VIEW_PHOTO, STEP_DEPTH: VIEW_DEPTH, STEP_EDIT: VIEW_RESULT,
                  STEP_LOOK: VIEW_RESULT, STEP_EXPORT: VIEW_RESULT}[index]
        has = {VIEW_PHOTO: True,
               VIEW_DEPTH: self.current_path is not None and str(self.current_path) in self.depth_maps,
               VIEW_RESULT: self.editor.result is not None}
        # Si aún no hay efecto, la vista cambiará sola cuando llegue el primer render.
        self._want_result_view = index in (STEP_EDIT, STEP_LOOK, STEP_EXPORT) and not has[VIEW_RESULT]
        current = self.view_switch.current_index()
        if has[wanted] and self.current_path is not None and current != wanted \
                and not (current == VIEW_COMPARE and wanted == VIEW_RESULT):
            self.view_switch.set_current_index(wanted)
            self._show_current()
        self._update_overlays()

    # =====================================================================
    # Importación y biblioteca
    # =====================================================================
    def import_photos(self) -> None:
        start_dir = self.settings.value("last_dir", str(Path.home()))
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("Importar fotografía"), start_dir, IMAGE_FILTER)
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_photos([Path(f) for f in files])

    def add_photos(self, paths: list[Path]) -> None:
        """Añade fotos a la biblioteca (sin duplicados) y selecciona la última."""
        existing = {self.library.item(i).data(Qt.UserRole) for i in range(self.library.count())}
        last_item = None
        for path in paths:
            if path.suffix.lower() not in IMAGE_EXTENSIONS or str(path) in existing:
                continue
            image = load_image(path)
            if image.isNull():
                log.warning("No se pudo leer la imagen: %s", path)
                QMessageBox.warning(self, "Importar", f"No se pudo abrir «{path.name}».")
                continue
            thumb = QPixmap.fromImage(image.scaled(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio,
                                                   Qt.SmoothTransformation))
            item = QListWidgetItem(QIcon(thumb), path.stem)
            item.setData(Qt.UserRole, str(path))
            item.setToolTip(str(path))
            self.library.addItem(item)
            last_item = item
            log.info("Importada: %s (%d×%d)", path.name, image.width(), image.height())
        if last_item is not None:
            self.library.setCurrentItem(last_item)

    def remove_photos(self, paths: list[str]) -> None:
        """Quita fotos de la biblioteca y olvida su mapa y sus ajustes (no toca los archivos)."""
        if not paths:
            return
        wanted = set(paths)
        for row in reversed(range(self.library.count())):
            if self.library.item(row).data(Qt.UserRole) in wanted:
                self.library.takeItem(row)     # si era la actual, cambia la selección sola
        for path in wanted:
            self.depth_maps.pop(path, None)
            self.depth_sources.pop(path, None)
            self.depth_runs.pop(path, None)
            self.editor.forget(path)
        log.info("Quitadas de la biblioteca: %d foto(s)", len(wanted))
        self.statusBar().showMessage(self.tr("Quitadas {0} foto(s) de la biblioteca (los archivos no se borran).")
                                     .format(len(wanted)), 4000)

    def _on_library_selection(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            self.current_path = None
            self.current_photo = self.full_photo = None
            self.editor.set_photo(None, None)
            self.canvas.set_image(None)
            self.photo_panel.show_photo(None)
            self.import_button.setVisible(True)
            return
        self.current_path = Path(current.data(Qt.UserRole))
        self.full_photo = load_image(self.current_path)
        self.current_photo = self._framed(self.full_photo, self.crops.get(str(self.current_path)))
        self._load_cached_depth(self.current_path)
        self._discover_runs(self.current_path)
        self.depth_result_label.hide()
        self._clear_run_selection()
        self._refresh_runs_list()
        self.import_button.setVisible(False)
        self.photo_panel.show_photo(self.current_path)
        self.photo_panel.set_crop(self.crops.get(str(self.current_path)), self._full_size())
        self.statusBar().showMessage(self.current_path.name)
        self.export_panel.refresh_for_photo()
        self._sync_editor()
        self._show_current(keep_view=False)

    # =====================================================================
    # Reencuadre previo (framing.py): se trabaja solo con una zona de la foto, sin tocar el archivo
    # =====================================================================
    def _full_size(self) -> tuple[int, int] | None:
        return (self.full_photo.width(), self.full_photo.height()) if self.full_photo is not None else None

    @staticmethod
    def _framed(image: QImage, crop: tuple | None) -> QImage:
        """La foto con el encuadre aplicado (la misma si no hay)."""
        if crop is None:
            return image
        left, top, right, bottom = framing.to_box(crop, (image.width(), image.height()))
        return image.copy(left, top, right - left, bottom - top)

    def reframe_photo(self) -> None:
        """Abre el diálogo de reencuadre de la foto elegida."""
        if self.full_photo is None or self.current_path is None:
            return
        dialog = CropDialog(self.full_photo, self.crops.get(str(self.current_path)), self)
        if dialog.exec() == QDialog.Accepted:
            self._set_crop(dialog.crop())

    def _set_crop(self, crop: tuple | None) -> None:
        """Pone (o quita) el encuadre de la foto actual. El mapa de profundidad de antes era de otra zona: se busca el de esta
        zona en la caché y, si no está, la foto queda sin mapa hasta calcularlo."""
        if self.current_path is None or self.full_photo is None:
            return
        key = str(self.current_path)
        crop = framing.normalize(crop)
        if crop == self.crops.get(key):
            return
        if crop is None:
            self.crops.pop(key, None)
        else:
            self.crops[key] = crop
        self.depth_maps.pop(key, None)
        self.depth_sources.pop(key, None)
        self.current_photo = self._framed(self.full_photo, crop)
        self.depth_runs.pop(key, None)                    # los cálculos de la lista eran de otra zona: se vuelve a mirar la caché
        self._load_cached_depth(self.current_path)
        self._discover_runs(self.current_path)
        self._refresh_runs_list()
        self.photo_panel.set_crop(crop, self._full_size())
        self._sync_editor()
        self._show_current(keep_view=False)
        if key not in self.depth_maps:
            self.statusBar().showMessage(self.tr("Encuadre cambiado: calcula la profundidad de esta zona (paso 2)."), 6000)

    # =====================================================================
    # Vista del lienzo
    # =====================================================================
    def _show_current(self, keep_view: bool = True) -> None:
        """Dibuja la foto o su mapa de profundidad según el conmutador de vista."""
        if self.current_photo is None:
            self.canvas.set_image(None)
            self.view3d.set_data(None, None, None)
            return
        depth = self.depth_maps.get(str(self.current_path))
        result = self.editor.result
        mode = self.view_switch.current_index()
        self.split_button.setVisible(mode == VIEW_COMPARE)
        self.mode_switch.setVisible(mode == VIEW_RESULT)          # el 2D / 3D es una opción de la vista Efecto
        in_3d = self._in_3d()
        self.texture_combo.setVisible(in_3d)
        self.quality3d_combo.setVisible(in_3d)
        self.depth_style_combo.setVisible(mode == VIEW_DEPTH)
        self._update_ab_visibility(mode)
        self._tag_runs()                                          # las marcas [A] [B] de la lista solo van en Comparar
        self._update_runs_buttons()
        self.viewer_stack.setCurrentWidget(self.view3d if in_3d else self.canvas)
        if in_3d:
            self._refresh_3d()
            self._update_overlays()
            return
        size = self.current_photo.size()
        if self._show_selected_runs(mode, keep_view):
            pass                                  # en la vista Profundidad manda lo seleccionado en la lista de cálculos
        elif mode != VIEW_PHOTO and depth is None:
            self.statusBar().showMessage(self.tr("Esta foto aún no tiene mapa de profundidad."), 4000)
            self.canvas.set_image(self.current_photo, keep_view=keep_view)
        elif mode == VIEW_DEPTH:
            self.canvas.set_image(self._depth_view_image(depth), keep_view=keep_view)
        elif mode == VIEW_RESULT and result is not None:
            self.canvas.set_image(result, keep_view=keep_view, scene_size=size)
        elif mode == VIEW_COMPARE:
            # Por defecto Foto | Efecto; cada lado se puede cambiar por otra cosa (otro cálculo de profundidad…).
            # Sin resultado todavía, el efecto se sustituye por la profundidad.
            left, right = self._ab_source(str(self.ab_a_combo.currentData())), self._ab_source(str(self.ab_b_combo.currentData()))
            if right is None and self.ab_b_combo.currentData() == "effect":
                right = (depth_to_qimage(depth), self.tr("Profundidad"))
            if left is None or right is None:
                self.statusBar().showMessage(self.tr("Lo elegido en A o B no está disponible para esta foto."), 4000)
                self.canvas.set_image(self.current_photo, keep_view=keep_view)
            else:
                self.canvas.set_compare(left[0], right[0], (left[1], right[1]), keep_view=keep_view)
        else:
            self.canvas.set_image(self.current_photo, keep_view=keep_view)
        self._update_overlays()

    def _refresh_combo_icons(self) -> None:
        """Iconos de las opciones de la textura 3D (foto, profundidad, efecto), con los colores del tema actual."""
        for i in range(self.texture_combo.count()):
            self.texture_combo.setItemIcon(i, icon(str(self.texture_combo.itemData(i))))

    def _depth_view_image(self, depth: np.ndarray) -> QImage:
        """El mapa de la vista Profundidad según el estilo elegido (gris, color o zonas). No cambia el mapa."""
        style = str(self.depth_style_combo.currentData())
        if style == "gray":
            return depth_to_qimage(depth)
        if style == "zones" and self.editor.project is not None:
            rgb = depth_view.zones_image(depth, self.editor.project.depth_control)
        elif style.startswith("color:"):
            rgb = depth_view.colorize(depth, style.split(":", 1)[1])
        elif style.startswith("normals:"):
            rgb = depth_view.normals_image(depth, style.split(":", 1)[1])
        else:
            return depth_to_qimage(depth)
        rgb = np.ascontiguousarray(rgb)
        return QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()

    def _on_depth_style_changed(self, _index: int) -> None:
        self.settings.setValue("depthview/style", self.depth_style_combo.currentData())
        if self.view_switch.current_index() == VIEW_DEPTH:
            self._show_current()

    def _zones_view_active(self) -> bool:
        """True si se está viendo el mapa con las zonas marcadas: ahí tocar el control no cambia de vista."""
        return self.view_switch.current_index() == VIEW_DEPTH and self.depth_style_combo.currentData() == "zones"

    def _on_quality3d_changed(self, _index: int) -> None:
        key = self.quality3d_combo.currentData()
        self.settings.setValue("view3d/quality", key)
        self.view3d.set_quality(key)

    def _in_3d(self) -> bool:
        """True si ahora se ve el relieve en 3D: vista Efecto con el modo 3D elegido."""
        return self.view_switch.current_index() == VIEW_RESULT and self.mode_switch.current_index() == MODE_3D

    def toggle_3d(self) -> None:
        """Alterna 2D / 3D (tecla 3).Si se está en otra vista, pasa antes al efecto."""
        if self.editor.project is None:
            self.statusBar().showMessage(self.tr("Calcula o carga primero el mapa de profundidad."), 4000)
            return
        if self.view_switch.current_index() != VIEW_RESULT:
            self.view_switch.set_current_index(VIEW_RESULT)
        self.mode_switch.set_current_index(MODE_2D if self.mode_switch.current_index() == MODE_3D else MODE_3D)
        self._show_current()

    def _refresh_3d(self) -> None:
        """Pasa al modo 3D el mapa, el relieve del proyecto y la textura elegida."""
        if not self._in_3d() or self.current_path is None:
            return
        depth = self.depth_maps.get(str(self.current_path))
        project = self.editor.project
        if project is not None and depth is not None and self.editor.path == str(self.current_path):
            # El relieve se ve con el control de profundidad aplicado. Se usa el mapa reducido de la vista previa
            # (con caché): la malla 3D es de baja resolución y así mover un control no recalcula el mapa entero.
            small = self.editor.render.preview_depth()
            depth = self.editor.warped(small if small is not None else depth)
        kind = self.texture_combo.currentData()
        if kind == "effect" and self.editor.result is not None:
            texture = self.editor.result
        elif kind == "depth" and depth is not None:
            texture = depth_to_qimage(depth)
        else:
            texture = self.current_photo
        self.view3d.set_data(depth, project.scene if project is not None else None, texture)

    def toggle_view(self) -> None:
        """Recorre foto → profundidad → efecto (desde Comparar, vuelve a la foto)."""
        mode = self.view_switch.current_index()
        nxt = {VIEW_PHOTO: VIEW_DEPTH, VIEW_DEPTH: VIEW_RESULT}.get(mode, VIEW_PHOTO)
        if nxt == VIEW_RESULT and self.editor.result is None:
            nxt = VIEW_PHOTO
        self.view_switch.set_current_index(nxt)
        self._show_current()

    # =====================================================================
    # Edición (ui/editor.py)
    # =====================================================================
    def _export_context(self) -> dict | None:
        """Qué se puede exportar ahora: la foto actual con su mapa (y sus ajustes, si los hay)."""
        if self.current_path is None:
            return None
        depth = self.depth_maps.get(str(self.current_path))
        if depth is None:
            return None
        return {"path": self.current_path, "depth": depth, "project": self.editor.project,
                "mask": self.editor._mask_array(), "crop": self.crops.get(str(self.current_path))}

    def _on_exported(self, paths: list) -> None:
        names = ", ".join(Path(p).name for p in paths)
        log.info("Exportado: %s", names)
        self.statusBar().showMessage(self.tr("Exportado: {0}").format(names), 8000)

    def show_unexpected_error(self, summary: str, trace: str) -> None:
        """Aviso de un error que nadie esperaba (ver errors.py): la traza ya está en el registro."""
        self.statusBar().showMessage(self.tr("Error inesperado. Consulta el panel Registro."), 8000)
        box = QMessageBox(QMessageBox.Warning, self.tr("Error inesperado"),
                          self.tr("Algo no ha salido como se esperaba. Puedes seguir trabajando; si se repite, "
                                  "copia el registro (Ver › Registro › Copiar) y envíalo a quien mantiene la app."),
                          QMessageBox.Ok, self)
        box.setInformativeText(summary)
        box.setDetailedText(trace)
        box.exec()

    def _on_export_failed(self, message: str, detail: str) -> None:
        self.statusBar().showMessage(self.tr("No se pudo exportar."), 5000)
        box = QMessageBox(QMessageBox.Warning, self.tr("Exportar"), message, QMessageBox.Ok, self)
        if detail:
            box.setDetailedText(detail)
        box.exec()

    def _sync_editor(self) -> None:
        """Pasa a la sesión de edición la foto actual y su mapa (si cambiaron)."""
        if hasattr(self, "export_panel"):
            self.export_panel.refresh()
        if self.current_path is None:
            return
        depth = self.depth_maps.get(str(self.current_path))
        if self.editor.path != str(self.current_path) or self.editor.depth is not depth:
            self.editor.set_photo(self.current_path, depth)

    def _on_result(self, image: QImage, full: bool, ms: float) -> None:
        self.render_label.setText(self.tr("{0}: {1:.0f} ms   |").format(
            self.tr("Completo") if full else self.tr("Vista previa"), ms))
        if getattr(self, "_want_result_view", False) and self.step_group.checkedId() in (STEP_EDIT, STEP_LOOK):
            self._want_result_view = False
            self.view_switch.set_current_index(VIEW_RESULT)
        if self.view_switch.current_index() in (VIEW_RESULT, VIEW_COMPARE) or self._zones_view_active():
            self._show_current()                 # en el modo 3D también refresca el relieve; en «zonas», sus colores

    def _on_user_edit(self) -> None:
        """Si se toca un ajuste del efecto estando en la foto o el mapa, se pasa a ver el efecto:
        si no, el cambio no se vería y parecería que el control no hace nada."""
        if (self.step_group.checkedId() == STEP_EDIT and self.view_switch.current_index() in (VIEW_PHOTO, VIEW_DEPTH)
                and not self._zones_view_active()):      # con las zonas a la vista, se está ajustando el mapa: no saltar
            self.view_switch.set_current_index(VIEW_RESULT)
            self._show_current()

    def _on_editable(self, editable: bool) -> None:
        for action in getattr(self, "_needs_depth_actions", ()):
            action.setEnabled(editable)
        self.edit_hint.setVisible(not editable)

    def _update_overlays(self) -> None:
        """Contornos de las áreas: solo en el paso de edición y sobre el efecto (o al comparar)."""
        context = self.editor.visible_context()          # áreas o bandas: solo se ven los tiradores de lo que se toca
        show = (self.step_group.checkedId() == STEP_EDIT and self.areas_button.isChecked() and context == "areas"
                and self.view_switch.current_index() in (VIEW_RESULT, VIEW_COMPARE) and not self._in_3d())
        self.canvas.set_overlays(self.editor.overlays() if show else [])
        # Las mismas áreas, con sus tiradores, en la vista 3D (solo en el paso de edición y con el ojo activado).
        show3d = (self.step_group.checkedId() == STEP_EDIT and self.areas_button.isChecked()
                  and self.editor.project is not None)
        self.view3d.set_areas(self.editor.visible_areas() if show3d and context == "areas" else [],
                              self.editor.selected_area)
        self.view3d.set_bands(self._band_planes() if show3d and context == "bands" else [],   # o los planos de las bandas
                              self.editor.bands_panel.list.currentRow())
        self.canvas.set_brush(self.editor.brush(self.canvas.tool()))
        self._refresh_mask_overlay()
        self._refresh_focus_overlay()

    def _band_planes(self) -> list[tuple]:
        """Plano de cada banda activa en la escena 3D: [(esquinas (4, 3), color, BandFrame, índice)]."""
        project, size = self.editor.project, self.editor.photo_size
        if project is None or size is None:
            return []
        aspect = size[0] / size[1]
        r = project.ramp
        out = []
        for i, b in enumerate(project.bands):
            if b.enabled and b.strength > 0 and not self.editor.bands_panel.hidden_by_solo(i):
                frame = area3d.BandFrame(b.position, b.width, b.pan, b.tilt, r.range_min, r.range_max,
                                         project.scene.relief, aspect)
                out.append((view3d_math.band_plane(b.position, b.pan, b.tilt, r.range_min, r.range_max,
                                                   project.scene.relief, aspect), tuple(b.color), frame, i))
        return out

    def _ensure_effect_view(self, allow_3d: bool = False) -> bool:
        """Las herramientas del lienzo trabajan sobre el efecto: pasa a ese paso y vista si hace falta.
        Devuelve False (con un aviso) si todavía no hay mapa de profundidad."""
        if self.editor.project is None:
            self.statusBar().showMessage(self.tr("Calcula o carga primero el mapa de profundidad."), 4000)
            return False
        if self.step_group.checkedId() != STEP_EDIT:
            self.set_step(STEP_EDIT)
        if self.view_switch.current_index() not in (VIEW_RESULT, VIEW_COMPARE):
            self.view_switch.set_current_index(VIEW_RESULT)
            self._show_current()
        if self._in_3d() and not allow_3d:                 # dibujar, colocar y pincel son del lienzo 2D (el cuentagotas también va en 3D)
            self.mode_switch.set_current_index(MODE_2D)
            self._show_current()
        self.areas_button.setChecked(True)        # una herramienta con las áreas ocultas no se vería
        return True

    def _on_paint_tool_changed(self, tool: str) -> None:
        """Pincel o borrador de la máscara (o ninguno)."""
        if tool and not self._ensure_effect_view():
            self.editor.mix_panel.set_paint_tool("", emit=False)
            return
        if tool:
            self.editor.areas_panel.set_tool("", emit=False)      # una sola herramienta a la vez
            self.editor.lens_panel.set_pick(False, emit=False)
        self.canvas.set_tool(tool)
        self.view3d.set_tool("")                              # el pincel es del lienzo 2D
        self.canvas.set_brush(self.editor.brush(tool))
        self._refresh_mask_overlay()
        if tool:
            self.statusBar().showMessage(self.tr("Pincel: arrastra sobre la imagen. [ y ] cambian el tamaño · "
                                                 "Esc para dejar de pintar."))

    def _refresh_brush(self) -> None:
        self.canvas.set_brush(self.editor.brush(self.canvas.tool()))

    def _refresh_mask_overlay(self) -> None:
        """Tinte rojo de la máscara pintada: solo con la máscara pintada elegida, en el paso del efecto
        y con «Ver máscara» activado."""
        layer = self.editor.mask_layer(create=False) if self.editor.project is not None else None
        show = (layer is not None and self.editor.project.composite.mask == "painted"
                and self.editor.mix_panel.show_mask.isChecked()
                and self.step_group.checkedId() == STEP_EDIT
                and self.view_switch.current_index() in (VIEW_RESULT, VIEW_COMPARE))
        if show:
            self.canvas.set_mask_overlay(layer.data, layer.photo_size)
        else:
            self.canvas.set_mask_overlay(None)

    def _on_tool_changed(self, tool: str) -> None:
        """«Dibujar» o «Colocar» activadas (o ninguna): el lienzo (2D o 3D) cambia de herramienta."""
        if tool and not self._ensure_effect_view(allow_3d=True):
            self.editor.areas_panel.set_tool("", emit=False)
            return
        if tool:
            self.editor.mix_panel.set_paint_tool("", emit=False)     # una sola herramienta a la vez
            self.editor.lens_panel.set_pick(False, emit=False)
        self.canvas.set_tool(tool)
        self.view3d.set_tool(tool)                            # en 3D, sobre la superficie del relieve
        self.canvas.set_brush(self.editor.brush(tool))
        self._refresh_mask_overlay()
        messages = {
            "draw": self.tr("Dibujar: pulsa en el centro del área y arrastra para darle tamaño "
                            "(Mayús: rectangular). Esc para cancelar."),
            "place": self.tr("Colocar: pulsa y arrastra; el área sigue al puntero hasta que sueltes. "
                             "Esc para cancelar."),
        }
        if tool:
            self.statusBar().showMessage(messages[tool])

    def _change_brush_size(self, factor: float) -> None:
        slider = self.editor.mix_panel.brush_size
        slider.setValue(slider.value() * factor, emit=True)

    def _on_lens_tool_changed(self, tool: str) -> None:
        """Cuentagotas del foco (o ninguna herramienta)."""
        if tool and not self._ensure_effect_view(allow_3d=True):
            self.editor.lens_panel.set_pick(False, emit=False)
            return
        if tool:
            self.editor.areas_panel.set_tool("", emit=False)      # una sola herramienta a la vez
            self.editor.mix_panel.set_paint_tool("", emit=False)
            self.editor.lens_panel.show_focus.setChecked(True)   # para ver qué queda nítido al elegir
        self.canvas.set_tool(tool)
        self.view3d.set_tool(tool)                            # en 3D el cuentagotas elige un punto de la superficie
        self.canvas.set_brush(None)
        self._refresh_mask_overlay()
        if tool:
            self.statusBar().showMessage(self.tr("Cuentagotas: pulsa en lo que quieres nítido (puedes arrastrar). "
                                                 "Esc para cancelar."))

    def _refresh_focus_overlay(self) -> None:
        """Tinte verde de la zona en foco: con «Ver zona en foco» marcado, en el paso del efecto y sobre el efecto."""
        project, depth = self.editor.project, self.editor.depth
        show = (project is not None and depth is not None and self.editor.lens_panel.show_focus.isChecked()
                and self.step_group.checkedId() == STEP_EDIT
                and self.view_switch.current_index() in (VIEW_RESULT, VIEW_COMPARE))
        if show and self.editor.photo_size is not None:
            mask = focus_map(self.editor.warped(depth), project.lens)
            self.canvas.set_focus_overlay(mask, self.editor.photo_size)
            self.view3d.set_focus_overlay(mask)                  # la misma capa verde sobre el relieve en 3D
        else:
            self.canvas.set_focus_overlay(None)
            self.view3d.set_focus_overlay(None)

    def _on_gesture_finished(self, kind: str) -> None:
        if kind == "pick_focus":
            self.editor.lens_panel.set_pick(False)               # el cuentagotas es de un solo uso
            self.statusBar().showMessage(self.tr("Foco elegido."), 3000)
            return
        if kind in ("draw", "place"):             # herramientas de un solo uso
            self.editor.areas_panel.set_tool("")
            self.statusBar().showMessage(self.tr("Listo. Ajusta el área con los tiradores o con los valores."), 4000)

    def _on_canvas_hint(self, text: str) -> None:
        if text:
            self.statusBar().showMessage(text)
        else:
            self.statusBar().clearMessage()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            if self.editor.areas_panel.tool():
                self.editor.areas_panel.set_tool("")
                return
            if self.editor.mix_panel.paint_tool():
                self.editor.mix_panel.set_paint_tool("")
                return
            if self.editor.lens_panel.pick_button.isChecked():
                self.editor.lens_panel.set_pick(False)
                return
        super().keyPressEvent(event)

    # =====================================================================
    # Proyecto (.deepshot): la biblioteca entera para retomarla en otro momento
    # =====================================================================
    def _warn(self, title: str, text: str) -> None:
        """Aviso modal (Aceptar). Se crea la caja y se llama a exec() para que sea fácil de sustituir en tests."""
        QMessageBox(QMessageBox.Warning, title, text, QMessageBox.Ok, self).exec()

    def _project_filter(self) -> str:
        return self.tr("Proyectos de {0} (*{1})").format(APP_NAME, project_file.EXTENSION)

    def _update_project_actions(self) -> None:
        """Guardar proyecto solo tiene sentido con fotos en la biblioteca."""
        has_photos = self.library.count() > 0
        for action in (self.action_save_project, self.action_save_project_as):
            action.setEnabled(has_photos)

    def _set_project_path(self, path: Path | None) -> None:
        self.project_path = path
        self._refresh_title()

    # --- Cambios sin guardar -----------------------------------------------------------------
    def _fingerprint(self) -> str:
        """Huella del contenido que guarda el proyecto: fotos, ajustes, mapas y máscaras.

        Se compara con la de la última vez que se guardó o abrió para saber si hay cambios sin guardar.
        Los ajustes que siguen siendo los de fábrica cuentan como «sin editar»: seleccionar una foto crea
        sus ajustes por defecto y eso no es un cambio."""
        default = Project().to_dict()
        parts = []
        for row in range(self.library.count()):
            path = self.library.item(row).data(Qt.UserRole)
            settings = self.editor.project_dict(path)
            if settings is not None and self._is_untouched(settings, default):
                settings = None
            mask = (settings or {}).get("composite", {}).get("painted_mask") or ""
            try:
                stat = Path(mask).stat() if mask else None
            except OSError:
                stat = None
            depth = self.depth_maps.get(path)
            depth_part = None if depth is None else (
                depth.shape, hash(np.ascontiguousarray(depth).ravel()[::4093].tobytes()), self.depth_sources.get(path))
            parts.append((path, json.dumps(settings, sort_keys=True), depth_part,
                          (stat.st_size, stat.st_mtime_ns) if stat else None))
        return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _is_untouched(settings: dict, default: dict) -> bool:
        """¿Son los ajustes de fábrica? El rango de la rampa se ajusta solo al mapa de cada foto al seleccionarla
        (no lo toca el usuario), así que no cuenta para decidirlo."""
        neutral = json.loads(json.dumps(settings))
        ramp = neutral.get("ramp")
        if isinstance(ramp, dict):
            for key in ("range_min", "range_max"):
                ramp[key] = default.get("ramp", {}).get(key)
        return neutral == json.loads(json.dumps(default))

    def is_dirty(self) -> bool:
        """True si hay trabajo que se perdería al cerrar: cambios desde el último guardado, o, en un proyecto
        que aún no tiene archivo, cualquier mapa de profundidad (una foto sin tocar no es trabajo)."""
        if self.library.count() == 0:
            return False
        if self.project_path is None:
            return any(self.library.item(r).data(Qt.UserRole) in self.depth_maps for r in range(self.library.count()))
        return self._fingerprint() != self._saved_fingerprint

    def _mark_saved(self) -> None:
        self._saved_fingerprint = self._fingerprint()
        self._refresh_title()

    def _refresh_title(self) -> None:
        name = self.project_path.stem if self.project_path is not None else None
        if self.is_dirty():
            name = f"{name or self.tr('Sin título')}*"
        self.setWindowTitle(f"{name} — {APP_NAME}" if name else APP_NAME)

    def _confirm_discard(self, title: str) -> bool:
        """Antes de cerrar o sustituir el proyecto: si hay cambios sin guardar, pregunta. True = se puede seguir."""
        if not self.is_dirty():
            return True
        name = self.project_path.name if self.project_path is not None else self.tr("el proyecto sin título")
        box = QMessageBox(QMessageBox.Question, title,
                          self.tr("Hay cambios sin guardar en {0}. ¿Quieres guardarlos?").format(name),
                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, self)
        box.button(QMessageBox.Save).setText(self.tr("Guardar"))
        box.button(QMessageBox.Discard).setText(self.tr("No guardar"))
        box.button(QMessageBox.Cancel).setText(self.tr("Cancelar"))
        box.setDefaultButton(QMessageBox.Save)
        answer = box.exec()
        if answer == QMessageBox.Save:
            return self.save_project()           # si se cancela el cuadro de guardar, no se sigue
        return answer == QMessageBox.Discard

    def new_project(self) -> bool:
        """Empieza en blanco: pregunta si hay cambios sin guardar y vacía la biblioteca. False si se cancela."""
        if not self._confirm_discard(self.tr("Proyecto nuevo")):
            return False
        self.remove_photos([self.library.item(i).data(Qt.UserRole) for i in range(self.library.count())])
        self._set_project_path(None)
        self._saved_fingerprint = ""
        self.statusBar().showMessage(self.tr("Proyecto nuevo: importa fotografías para empezar."), 5000)
        return True

    def startup_dialogs(self) -> None:
        """Lo que se pregunta al arrancar: la pantalla de inicio (si no se desactivó) y, si no hay ningún modelo instalado,
        el aviso del primer arranque."""
        if self._start_enabled():
            self.show_start_dialog()
        self.show_first_run_if_needed()

    def show_first_run_if_needed(self) -> str:
        """Si no hay ningún modelo de profundidad instalado (y no se pidió no volver a preguntar), explica qué hace falta y
        ofrece abrir el gestor o instalar desde una carpeta. Devuelve «manager», «folder», «later» o «» si no se mostró."""
        if self.settings.value("firstrun/dismissed", False, type=bool) or not needs_first_run(self.models):
            return ""
        dialog = FirstRunDialog(self.models, self)
        dialog.exec()
        if dialog.dont_ask.isChecked():
            self.settings.setValue("firstrun/dismissed", True)
        if dialog.action == "manager":
            self.open_model_manager()
        elif dialog.action == "folder":
            self.open_model_manager(then="install_from_folder")
        return dialog.action

    def show_start_dialog(self, show_checkbox: bool = True) -> str:
        """Pantalla de inicio: proyecto nuevo, abrir uno o retomar uno reciente. Devuelve lo que se eligió
        («new», «open» o «recent»). La casilla «No volver a mostrar» se guarda en `startup/show_start`."""
        dialog = StartDialog(self.recent_projects(), self, show_checkbox)
        dialog.exec()
        for path in dialog.removed:
            self.settings.setValue("recent_projects", [p for p in self.recent_projects() if p != path])
        if show_checkbox and dialog.dont_show.isChecked():
            self.settings.setValue("startup/show_start", False)
        if dialog.action == "recent" and dialog.chosen_path:
            self.open_recent(dialog.chosen_path)
        elif dialog.action == "open":
            self.open_project()
        elif dialog.action == "quick":
            self.open_simple_depth()
        elif dialog.action == "new" and self.library.count():
            self.new_project()
        return dialog.action

    def open_simple_depth(self) -> SimpleDepthDialog:
        """Modo sencillo: solo el mapa de profundidad (o de normales) de unas fotos, sin proyecto ni editor."""
        dialog = SimpleDepthDialog(self)
        if self.library.count():
            dialog.add_library_photos()
        dialog.show()
        self._simple_dialog = dialog                  # para que no lo recoja el recolector mientras está abierto
        return dialog

    # --- Apariencia y disposición --------------------------------------------------------------------
    def show_preferences(self) -> None:
        PreferencesDialog(self).exec()

    def apply_appearance(self, theme_key: str | None = None, appearance: themes.Appearance | None = None,
                         save: bool = True) -> None:
        """Cambia el tema de color y el aspecto de los controles al momento (sin reiniciar) y lo guarda."""
        key = theme_key if theme_key is not None else theme.current_theme().key
        look = (appearance or theme.current_appearance()).validated()
        theme.apply_theme(QApplication.instance(), key, look)
        if save:
            theme.save_choice(theme.current_theme().key, look)
        refresh_icons(self)                       # los iconos se dibujan con los colores del tema
        self._refresh_combo_icons()
        for widget in (self.canvas, self.canvas.viewport(), self.view3d):
            widget.update()
        self.update()

    def _fill_theme_menu(self) -> None:
        self.theme_menu.clear()
        current = theme.current_theme().key
        for t in themes.all_themes():
            label = QCoreApplication.translate("Themes", t.label) if t.factory else t.label
            action = self.theme_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(t.key == current)
            action.triggered.connect(lambda _c=False, k=t.key: self.apply_appearance(k))

    def _fill_layout_menu(self) -> None:
        self.layout_menu.clear()
        for key, label in layouts.factory_names():
            self.layout_menu.addAction(QCoreApplication.translate("Layouts", label)).triggered.connect(
                lambda _c=False, k=key: self.apply_layout("factory", k))
        user = layouts.list_user()
        if user:
            self.layout_menu.addSeparator()
            for name in user:
                self.layout_menu.addAction(name).triggered.connect(lambda _c=False, n=name: self.apply_layout("user", n))
        self.layout_menu.addSeparator()
        save = self.layout_menu.addAction(self.tr("Guardar la disposición actual…"))
        save.triggered.connect(self._ask_save_layout)
        self.layout_menu.addAction(self.tr("Restablecer disposición")).triggered.connect(lambda _c=False: self.reset_layout())

    def apply_layout(self, source: str, key: str) -> bool:
        """Aplica una disposición de los paneles («factory» o «user»). False si no se pudo."""
        try:
            if source == "factory":
                layouts.apply_factory(self, key)
                return True
            return layouts.apply_user(self, key)
        except KeyError:
            return False

    def reset_layout(self) -> None:
        layouts.reset(self)
        self.statusBar().showMessage(self.tr("Disposición restablecida."), 4000)

    def save_layout(self, name: str) -> bool:
        try:
            layouts.save_user(self, name)
        except (OSError, ValueError) as exc:
            self._warn(self.tr("Guardar disposición"), self.tr("No se pudo guardar: {0}").format(exc))
            return False
        self.statusBar().showMessage(self.tr("Disposición guardada: {0}").format(name.strip()), 4000)
        return True

    def _ask_save_layout(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, self.tr("Guardar disposición"), self.tr("Nombre de la disposición:"))
        if ok and name.strip():
            self.save_layout(name)

    def _start_enabled(self) -> bool:
        return self.settings.value("startup/show_start", True, type=bool)

    # --- Proyectos recientes ----------------------------------------------------------------------
    def recent_projects(self) -> list[str]:
        value = self.settings.value("recent_projects", [])
        if isinstance(value, str):               # QSettings devuelve un texto si la lista tiene un solo elemento
            value = [value]
        return [str(v) for v in (value or [])]

    def _add_recent(self, path: Path) -> None:
        items = [str(path), *[p for p in self.recent_projects() if p != str(path)]]
        self.settings.setValue("recent_projects", items[:RECENT_MAX])

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self.recent_projects()
        for file in recent:
            action = self.recent_menu.addAction(Path(file).name)
            action.setToolTip(file)
            action.setStatusTip(file)
            action.triggered.connect(lambda _checked=False, f=file: self.open_recent(f))
        if not recent:
            self.recent_menu.addAction(self.tr("(ninguno)")).setEnabled(False)
        else:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction(self.tr("Vaciar la lista")).triggered.connect(
                lambda _checked=False: self.settings.setValue("recent_projects", []))

    def open_recent(self, file: str) -> bool:
        """Abre un proyecto de la lista de recientes; si el archivo ya no está, avisa y lo quita de la lista."""
        if not Path(file).is_file():
            self.settings.setValue("recent_projects", [p for p in self.recent_projects() if p != file])
            self._warn(self.tr("Abrir proyecto"), self.tr("«{0}» ya no está en su sitio. Se quita de la lista.")
                       .format(Path(file).name))
            return False
        return self.open_project(file)

    def _project_entries(self) -> list:
        """Datos de todas las fotos de la biblioteca (ajustes, mapa y máscara) para guardarlos."""
        entries = []
        for row in range(self.library.count()):
            path = self.library.item(row).data(Qt.UserRole)
            settings = self.editor.project_dict(path)
            mask = settings["composite"].get("painted_mask") if settings else ""
            entries.append(project_file.entry_for_photo(
                Path(path), settings, self.depth_maps.get(path), self.depth_sources.get(path, ""),
                Path(mask) if mask else None, self.crops.get(path)))
        return entries

    def save_project(self) -> bool:
        """Guarda en el proyecto abierto; si todavía no tiene archivo, pregunta dónde."""
        if self.project_path is None:
            return self.save_project_as()
        return self._write_project(self.project_path)

    def save_project_as(self) -> bool:
        if self.library.count() == 0:
            self.statusBar().showMessage(self.tr("No hay nada que guardar: importa alguna fotografía."), 4000)
            return False
        start = self.project_path or Path(self.settings.value("last_project_dir", str(Path.home()))) / \
            f"{self.current_path.stem if self.current_path else 'proyecto'}{project_file.EXTENSION}"
        file, _ = QFileDialog.getSaveFileName(self, self.tr("Guardar proyecto"), str(start), self._project_filter())
        if not file:
            return False
        path = Path(file)
        if path.suffix.lower() != project_file.EXTENSION:
            path = path.with_name(path.name + project_file.EXTENSION)
        self.settings.setValue("last_project_dir", str(path.parent))
        return self._write_project(path)

    def _write_project(self, path: Path) -> bool:
        QApplication.setOverrideCursor(Qt.WaitCursor)       # calcular las huellas de fotos grandes tarda un poco
        try:
            current = self.library.currentRow()
            project_file.save_project(path, self._project_entries(), max(current, 0), __version__)
        except OSError as exc:
            log.warning("No se pudo guardar el proyecto %s: %s", path, exc)
            QApplication.restoreOverrideCursor()
            self._warn(self.tr("Guardar proyecto"),
                                self.tr("No se pudo guardar «{0}»: {1}").format(path.name, exc.strerror or exc))
            return False
        QApplication.restoreOverrideCursor()
        self._set_project_path(path)
        self._add_recent(path)
        self._mark_saved()
        self.statusBar().showMessage(self.tr("Proyecto guardado: {0}").format(path.name), 4000)
        return True

    def open_project(self, file: str | None = None) -> bool:
        """Abre un proyecto: sustituye la biblioteca actual por la del archivo."""
        if not file:
            start = self.settings.value("last_project_dir", str(Path.home()))
            file, _ = QFileDialog.getOpenFileName(self, self.tr("Abrir proyecto"), start, self._project_filter())
        if not file:
            return False
        path = Path(file)
        try:
            loaded = project_file.load_project(path)
        except project_file.ProjectFileError as exc:
            log.warning("Proyecto no válido %s: %s", path, exc)
            self._warn(self.tr("Abrir proyecto"),
                                self.tr("No se pudo abrir «{0}»: {1}").format(path.name, exc))
            return False
        if not self._confirm_discard(self.tr("Abrir proyecto")):
            return False
        self.settings.setValue("last_project_dir", str(path.parent))
        self.remove_photos([self.library.item(i).data(Qt.UserRole) for i in range(self.library.count())])

        found, missing, changed = [], [], []
        for entry in loaded.photos:
            photo, was_changed = project_file.locate_photo(entry, path)
            if photo is None:       # movida o borrada: se le pide al usuario que la localice
                photo = self._ask_photo_location(entry)
            if photo is None:
                missing.append(entry.name or Path(entry.path).name)
                found.append(None)
                continue
            found.append(photo)
            if self._register_project_photo(entry, photo, was_changed):
                changed.append(photo.name)
        self.add_photos([p for p in found if p is not None])
        wanted = found[loaded.current] if loaded.current < len(found) else None
        for row in range(self.library.count()):
            if wanted is not None and self.library.item(row).data(Qt.UserRole) == str(wanted):
                self.library.setCurrentRow(row)
        self._set_project_path(path)
        self._add_recent(path)
        self._mark_saved()
        self._report_open(path, missing, changed)
        return True

    def _ask_photo_location(self, entry) -> Path | None:
        name = entry.name or Path(entry.path).name
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Localizar «{0}»").format(name),
                                              str(Path(entry.path).parent), IMAGE_FILTER)
        return Path(file) if file else None

    def _register_project_photo(self, entry, photo: Path, was_changed: bool) -> bool:
        """Deja listos el mapa, los ajustes y la máscara de una foto del proyecto.

        Devuelve True si hubo que descartar su trabajo porque la foto ya no coincide con el mapa."""
        key = str(photo)
        size = project_file.photo_size(photo)
        crop = framing.normalize(entry.crop)
        if crop is not None:
            self.crops[key] = crop
            if size is not None:
                size = framing.cropped_size(crop, size)          # el mapa guardado es el de la zona encuadrada
        if entry.depth is not None and size is not None and (entry.depth.shape[1], entry.depth.shape[0]) != size:
            log.warning("La foto %s no coincide con su mapa guardado: se descartan mapa y ajustes", photo.name)
            return True
        if entry.depth is not None:
            self.depth_maps[key] = entry.depth
            self.depth_sources[key] = entry.source or "external"
        settings = entry.project
        if settings:
            composite = settings.setdefault("composite", {})
            if entry.mask:
                target = mask_path_for(key)          # la máscara vuelve a la carpeta de datos de este equipo
                try:
                    target.write_bytes(entry.mask)
                    composite["painted_mask"] = str(target)
                except OSError as exc:
                    log.warning("No se pudo restaurar la máscara de %s: %s", photo.name, exc)
                    composite["painted_mask"] = ""
            else:
                composite["painted_mask"] = ""
            try:
                self.editor.set_project_dict(key, settings)
            except (ValueError, KeyError, TypeError) as exc:
                log.warning("Ajustes no válidos para %s: %s", photo.name, exc)
        return was_changed

    def _report_open(self, path: Path, missing: list[str], changed: list[str]) -> None:
        text = self.tr("Proyecto abierto: {0}").format(path.name)
        if missing or changed:
            lines = [self.tr("El proyecto se abrió, pero:")]
            if missing:
                lines.append(self.tr("• No se encontraron estas fotos y quedan fuera: {0}").format(", ".join(missing)))
            if changed:
                lines.append(self.tr("• Estas fotos han cambiado desde que se guardó el proyecto "
                                     "(revisa su mapa y su efecto): {0}").format(", ".join(changed)))
            self._warn(self.tr("Abrir proyecto"), "\n".join(lines))
        self.statusBar().showMessage(text, 4000)

    # =====================================================================
    # Estilo (.deepshotstyle): el aspecto artístico como archivo suelto
    # =====================================================================
    def _style_filter(self) -> str:
        return self.tr("Estilos de {0} (*{1} *.json)").format(APP_NAME, STYLE_EXTENSION)

    def save_style_file(self) -> None:
        if self.editor.project is None:
            self.statusBar().showMessage(self.tr("No hay estilo que guardar: falta el mapa de profundidad."), 4000)
            return
        start = Path(self.settings.value("last_style_dir", str(self.current_path.parent))) / \
            f"{self.current_path.stem}{STYLE_EXTENSION}"
        file, _ = QFileDialog.getSaveFileName(self, self.tr("Guardar estilo"), str(start), self._style_filter())
        if not file:
            return
        path = Path(file)
        if path.suffix.lower() != STYLE_EXTENSION:
            path = path.with_name(path.name + STYLE_EXTENSION)
        try:
            presets.save_file(path, "style", path.stem, presets.from_project("style", self.editor.project))
        except OSError as exc:
            self._warn(self.tr("Guardar estilo"),
                                self.tr("No se pudo guardar «{0}»: {1}").format(path.name, exc.strerror or exc))
            return
        self.settings.setValue("last_style_dir", str(path.parent))
        self.statusBar().showMessage(self.tr("Estilo guardado: {0}").format(path.name), 4000)

    def load_style_file(self) -> None:
        if self.editor.project is None:
            self.statusBar().showMessage(self.tr("Selecciona una foto con mapa de profundidad."), 4000)
            return
        start = self.settings.value("last_style_dir", str(self.current_path.parent))
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Cargar estilo"), start, self._style_filter())
        if not file:
            return
        try:
            try:
                preset = presets.load_file(Path(file), "style")
            except ValueError:
                # También vale el .json de ajustes que acompaña a una exportación: se toma su aspecto.
                preset = presets.from_project("style", Project.load(Path(file)))
            self.editor.apply_style(preset)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Estilo no válido %s: %s", file, exc)
            self._warn(self.tr("Cargar estilo"),
                                self.tr("El archivo no contiene un estilo válido de {0}.").format(APP_NAME))
            return
        self.settings.setValue("last_style_dir", str(Path(file).parent))
        self.statusBar().showMessage(self.tr("Estilo aplicado: {0}").format(Path(file).stem), 4000)

    def toggle_compare(self) -> None:
        mode = self.view_switch.current_index()
        self.view_switch.set_current_index(VIEW_PHOTO if mode == VIEW_COMPARE else VIEW_COMPARE)
        self._show_current()

    def toggle_split_orientation(self) -> None:
        vertical_line = self.canvas.split_orientation() == Qt.Horizontal
        self.canvas.set_split_orientation(Qt.Vertical if vertical_line else Qt.Horizontal)
        # El icono indica hacia dónde se mueve el divisor.
        self.split_button.setText("⇅" if vertical_line else "⇆")

    # =====================================================================
    # Profundidad
    # =====================================================================
    def _load_cached_depth(self, path: Path) -> None:
        """Si el mapa ya se calculó antes (caché en disco), lo carga sin usar el modelo."""
        if str(path) in self.depth_maps:
            return
        try:
            cached = cache_path(path, self._depth_settings())
            if cached.is_file():
                self.depth_maps[str(path)] = load_depth_png(cached)
                self.depth_sources[str(path)] = self._depth_settings().key()
                self._register_run(str(path), self._depth_settings(), self.depth_maps[str(path)], None, cached)
        except OSError as exc:
            log.warning("No se pudo leer la caché de profundidad: %s", exc)

    # --- Cálculos de profundidad de la foto (tiempos y comparación) ---------------------------------
    def _runs_for(self, path: str) -> depth_runs.RunList:
        return self.depth_runs.setdefault(path, depth_runs.RunList())

    def _register_run(self, path: str, settings: DepthSettings, depth: np.ndarray | None, run,
                      cache_file: Path | None = None) -> None:
        """Apunta un cálculo de esta foto en la lista (sustituye al que tenga los mismos ajustes)."""
        if run is None and cache_file is not None:              # sale de la caché: los tiempos van en su .json
            t = read_timing(cache_file)
            run = DepthRun(settings.model, settings.processing_resolution, settings.ensemble_size, settings.refine,
                           infer_seconds=t.get("infer_seconds"), refine_seconds=float(t.get("refine_seconds", 0.0)),
                           from_cache=True)
        if cache_file is None:
            try:
                cache_file = cache_path(Path(path), settings)
            except OSError:
                cache_file = None
        runs = self._runs_for(path)
        old = runs.get(settings.key())
        if old is not None and run is not None and run.from_cache and old.run is not None and not old.run.from_cache:
            run = old.run                                        # se conserva el tiempo medido en esta sesión
        runs.add(depth_runs.RunEntry(settings, run, cache_file, depth), keep_key=self.depth_sources.get(path, ""))

    def _discover_runs(self, path: Path) -> None:
        """Recupera de la caché de disco los cálculos anteriores de esta foto (con sus tiempos), sin leer los mapas."""
        runs = self._runs_for(str(path))
        try:
            found = depth_runs.discover(path, [m.id for m in self.models.all(include_hidden=True)],
                                        self.crops.get(str(path)))
        except OSError as exc:
            log.warning("No se pudo mirar la caché de profundidad: %s", exc)
            return
        for entry in found:
            if runs.get(entry.key) is None:
                runs.add(entry, keep_key=self.depth_sources.get(str(path), ""))

    def _run_label(self, entry: depth_runs.RunEntry, mark: bool = True) -> str:
        try:
            model = self.models.get(entry.settings.model).display_label
        except ModelError:
            model = entry.settings.model
        parts = [model, depth_runs.quality_label(entry.settings)]
        if entry.settings.refine > 0:
            parts.append(self.tr("refinado"))
        current = self.depth_sources.get(str(self.current_path), "") if self.current_path is not None else ""
        dot = "● " if mark and entry.key == current else ("    " if mark else "")
        return f"{dot}{' · '.join(parts)}  —  {depth_runs.format_seconds(entry.seconds)}"

    def _run_tooltip(self, entry: depth_runs.RunEntry) -> str:
        run = entry.run
        if run is None:
            return self.tr("Sin datos de tiempo.")
        lines = [self.tr("Modelo + refinado: {0}").format(depth_runs.format_seconds(run.seconds))]
        if run.refine_seconds:
            lines.append(self.tr("  de ellos, refinar bordes: {0}").format(depth_runs.format_seconds(run.refine_seconds)))
        if run.load_seconds:
            lines.append(self.tr("Cargar el modelo: {0} (solo la primera vez)").format(
                depth_runs.format_seconds(run.load_seconds)))
        if run.from_cache:
            lines.append(self.tr("Recuperado de la caché: el tiempo es el del cálculo original."))
        if run.device:
            lines.append(run.device)
        return "\n".join(lines)

    def _selected_run_entries(self) -> list[depth_runs.RunEntry]:
        if self.current_path is None:
            return []
        runs = self.depth_runs.get(str(self.current_path))
        if runs is None:
            return []
        keys = {item.data(Qt.UserRole) for item in self.runs_list.selectedItems()}
        return [e for e in runs.entries if e.key in keys]

    def _refresh_runs_list(self) -> None:
        """Rehace la lista de cálculos de la foto actual (conservando lo seleccionado)."""
        keep = {item.data(Qt.UserRole) for item in self.runs_list.selectedItems()}
        runs = self.depth_runs.get(str(self.current_path)) if self.current_path is not None else None
        self._runs_updating = True
        self.runs_list.clear()
        if runs is not None:
            for entry in runs.entries:
                item = QListWidgetItem(self._run_label(entry))
                item.setData(Qt.UserRole, entry.key)
                item.setToolTip(self._run_tooltip(entry))
                self.runs_list.addItem(item)
                item.setSelected(entry.key in keep)
        self._runs_updating = False
        self.runs_box.setVisible(self.runs_list.count() > 0)
        self._update_runs_buttons()
        self._refresh_ab()
        self._tag_runs()

    # --- Vista Comparar: qué va a cada lado del divisor -----------------------------------------------
    def _ab_run_entries(self) -> list[depth_runs.RunEntry]:
        path = getattr(self, "current_path", None)             # aún no existe cuando se crea la barra
        runs = getattr(self, "depth_runs", {}).get(str(path)) if path is not None else None
        return list(runs.entries) if runs is not None else []

    def _refresh_ab(self) -> None:
        """Rehace las opciones de A y B (foto, efecto, mapa en uso y cada cálculo de la foto) conservando lo elegido."""
        entries = self._ab_run_entries()
        keep = (self.ab_a_combo.currentData() or "photo", self.ab_b_combo.currentData() or "effect")
        self._ab_updating = True
        for combo, wanted in ((self.ab_a_combo, keep[0]), (self.ab_b_combo, keep[1])):
            combo.clear()
            combo.addItem(self.tr("Foto"), "photo")
            combo.addItem(self.tr("Efecto"), "effect")
            combo.addItem(self.tr("Mapa en uso"), "depth")
            for entry in entries:
                combo.addItem(self.tr("Profundidad: {0}").format(self._run_label(entry, mark=False)), "run:" + entry.key)
                combo.setItemData(combo.count() - 1, self._run_tooltip(entry), Qt.ToolTipRole)
            index = combo.findData(wanted)
            combo.setCurrentIndex(index if index >= 0 else (0 if combo is self.ab_a_combo else 1))
        self._ab_updating = False

    def _update_ab_visibility(self, mode: int) -> None:
        """Los desplegables A y B solo se ven en la vista Comparar (2D)."""
        for combo in (self.ab_a_combo, self.ab_b_combo):
            combo.setVisible(mode == VIEW_COMPARE and not self._in_3d())

    def _ab_source(self, key: str) -> tuple[QImage, str] | None:
        """Imagen y rótulo de lo elegido en un lado del divisor; None si no hay de dónde sacarlo."""
        depth = self.depth_maps.get(str(self.current_path))
        if key == "photo":
            return self.current_photo, self.tr("Foto")
        if key == "effect":
            result = self.editor.result
            return (result, self.tr("Efecto")) if result is not None else None
        if key == "depth":
            return (self._depth_view_image(depth), self.tr("Mapa en uso")) if depth is not None else None
        entry = next((e for e in self._ab_run_entries() if "run:" + e.key == key), None)
        run_depth = entry.get_depth() if entry is not None else None
        if entry is None or run_depth is None:
            return None
        return self._depth_view_image(run_depth), self._run_label(entry, mark=False)

    def _on_ab_changed(self, _index: int) -> None:
        if not self._ab_updating and self.view_switch.current_index() == VIEW_COMPARE:
            self._show_current()
            self._tag_runs()

    def _clear_run_selection(self) -> None:
        """Deselecciona la lista sin que eso cambie lo que se está viendo."""
        self._runs_updating = True
        self.runs_list.clearSelection()
        self._runs_updating = False

    def _update_runs_buttons(self) -> None:
        selected = self._selected_run_entries()
        current = self.depth_sources.get(str(self.current_path), "") if self.current_path is not None else ""
        self.runs_use_button.setEnabled(len(selected) == 1 and selected[0].key != current)
        self.runs_remove_button.setEnabled(bool(selected) and all(e.key != current for e in selected))
        self.runs_keep_button.setEnabled(len(selected) == 1)
        if self._comparing_view():
            self.runs_hint.setText(self.tr("Comparando: clic izquierdo en un cálculo lo pone a la izquierda (A); "
                                           "clic derecho, a la derecha (B)."))
        elif len(selected) >= 2:
            self.runs_hint.setText(self.tr("Comparando los dos primeros con el divisor (arrástralo)."))
        elif len(selected) == 1 and selected[0].key != current:
            self.runs_hint.setText(self.tr("Estás viendo un cálculo que NO es el que se usa. Pulsa «Usar este mapa» "
                                           "para seguir con él."))
        else:
            self.runs_hint.setText(self.tr("Selecciona un cálculo para verlo, o dos para compararlos."))

    def _on_runs_selection_changed(self) -> None:
        if self._runs_updating:
            return
        self._update_runs_buttons()
        if self.current_path is None:
            return
        # En la vista Comparar la lista no la abandona: lo que se ve a cada lado lo deciden A y B (clic izquierdo / derecho).
        if self.view_switch.current_index() != VIEW_DEPTH and not self._comparing_view():
            self.view_switch.set_current_index(VIEW_DEPTH)
        self._show_current()

    # --- Vista Comparar: la lista de cálculos reparte los lados A y B --------------------------------
    def _comparing_view(self) -> bool:
        return self.view_switch.current_index() == VIEW_COMPARE and not self._in_3d()

    def _assign_run_side(self, side: str, item) -> None:
        """Pone el cálculo de la lista en el lado A (izquierda) o B (derecha) del divisor."""
        combo = self.ab_a_combo if side == "A" else self.ab_b_combo
        index = combo.findData("run:" + str(item.data(Qt.UserRole)))
        if index >= 0 and index != combo.currentIndex():
            combo.setCurrentIndex(index)          # _on_ab_changed redibuja y rehace las marcas

    def _tag_runs(self) -> None:
        """En la vista Comparar, marca en la lista qué cálculo va a la izquierda [A] y cuál a la derecha [B]."""
        comparing = self._comparing_view()
        a, b = self.ab_a_combo.currentData(), self.ab_b_combo.currentData()
        entries = {e.key: e for e in self._ab_run_entries()}
        for i in range(self.runs_list.count()):
            item = self.runs_list.item(i)
            key = str(item.data(Qt.UserRole))
            entry = entries.get(key)
            if entry is None:
                continue
            tag = ("A" if a == "run:" + key else "") + ("B" if b == "run:" + key else "")
            prefix = (f"[{tag}]".ljust(5) + " ") if comparing else ""
            item.setText(prefix + self._run_label(entry))

    def _show_selected_runs(self, mode: int, keep_view: bool) -> bool:
        """En la vista Profundidad enseña el cálculo seleccionado, o dos a la vez con el divisor. True si lo hizo."""
        if mode != VIEW_DEPTH:
            return False
        selected = self._selected_run_entries()
        maps = []
        for entry in selected[:2]:
            depth = entry.get_depth()
            if depth is not None:
                maps.append((entry, depth))
        if not maps:
            return False
        images = [self._depth_view_image(d) for _e, d in maps]
        if len(maps) == 2:
            self.canvas.set_compare(images[0], images[1], (self._run_label(maps[0][0], mark=False),
                                                          self._run_label(maps[1][0], mark=False)), keep_view=keep_view)
        else:
            self.canvas.set_image(images[0], keep_view=keep_view)
        return True

    def _show_result_time(self, run, settings: DepthSettings) -> None:
        if run is None:
            return
        try:
            model = self.models.get(settings.model).display_label
        except ModelError:
            model = settings.model
        text = self.tr("Listo en {0} · {1} · {2}").format(
            depth_runs.format_seconds(run.seconds), model, depth_runs.quality_label(settings))
        if run.from_cache:
            text += self.tr("  (de la caché)")
        elif run.load_seconds:
            text += self.tr("  (+{0} cargando el modelo)").format(depth_runs.format_seconds(run.load_seconds))
        self.depth_result_label.setText(text)
        self.depth_result_label.show()

    def _apply_run_to_combos(self, settings: DepthSettings) -> None:
        """Pone el modelo, la calidad y el refinado del panel como los del cálculo elegido (sin recalcular)."""
        # Con una resolución propia no hay preset exacto: se elige el primero con las mismas pasadas.
        quality = depth_runs.quality_id(settings) or next(
            (p.id for p in QUALITY_PRESETS.values() if p.ensemble_size == settings.ensemble_size), None)
        for widget, value in ((self.model_combo, settings.model), (self.quality_combo, quality)):
            index = widget.findData(value) if value is not None else -1
            if index >= 0:
                widget.blockSignals(True)
                widget.setCurrentIndex(index)
                widget.blockSignals(False)
        # Resolución: la del preset (Automática) o, si el cálculo usó otra, esa.
        preset_px = QUALITY_PRESETS[self.quality_combo.currentData() or DEFAULT_QUALITY].processing_resolution
        res = 0 if settings.processing_resolution in (None, preset_px) else settings.processing_resolution
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.setCurrentIndex(max(0, self.resolution_combo.findData(res)))
        self.resolution_combo.blockSignals(False)
        self.settings.setValue("depth/resolution", self.resolution_combo.currentData() or 0)
        self.refine_check.blockSignals(True)
        self.refine_check.setChecked(settings.refine > 0)
        self.refine_check.blockSignals(False)
        self.settings.setValue("depth/active_model", settings.model)
        self.settings.setValue("depth/quality", self.quality_combo.currentData())
        self.settings.setValue("depth/refine", self.refine_check.isChecked())
        self._update_quality_hint()
        self._update_params_label()

    def _use_selected_run(self) -> None:
        selected = self._selected_run_entries()
        if len(selected) != 1 or self.current_path is None:
            return
        entry = selected[0]
        depth = entry.get_depth()
        if depth is None:
            self._warn(self.tr("Cálculos"), self.tr("No se pudo leer este mapa (¿se borró la caché?). "
                                                    "Quítalo de la lista y vuelve a calcularlo."))
            return
        path = str(self.current_path)
        self.depth_maps[path] = depth
        self.depth_sources[path] = entry.key
        self._apply_run_to_combos(entry.settings)
        self._sync_editor()
        self._clear_run_selection()
        self._refresh_runs_list()
        self._show_current()
        self.statusBar().showMessage(self.tr("Se usa el mapa: {0}").format(self._run_label(entry, mark=False)), 6000)

    def _remove_selected_runs(self) -> None:
        if self.current_path is None:
            return
        runs = self._runs_for(str(self.current_path))
        current = self.depth_sources.get(str(self.current_path), "")
        for entry in self._selected_run_entries():
            if entry.key != current:
                runs.remove(entry.key)
        self._refresh_runs_list()
        self._show_current()

    def _keep_only_selected_run(self) -> None:
        selected = self._selected_run_entries()
        if len(selected) != 1:
            return
        if selected[0].key != self.depth_sources.get(str(self.current_path), ""):
            self._use_selected_run()
        self._runs_for(str(self.current_path)).keep_only(selected[0].key)
        self._refresh_runs_list()
        self._show_current()

    def _release_run_maps(self) -> None:
        """«Liberar memoria»: los mapas de la lista que se pueden volver a leer de la caché dejan de estar en memoria."""
        for path, runs in self.depth_runs.items():
            in_use = self.depth_sources.get(path, "")
            for entry in runs.entries:
                if entry.key != in_use and entry.cache_file is not None and entry.cache_file.is_file():
                    entry.depth = None

    def _depth_settings(self) -> DepthSettings:
        """Ajustes actuales del panel: modelo, calidad y refinado."""
        settings = settings_for(model=self.model_combo.currentData() or DEFAULT_MODEL,
                                quality=self.quality_combo.currentData() or DEFAULT_QUALITY,
                                refine=1.0 if self.refine_check.isChecked() else 0.0,
                                resolution=self.resolution_combo.currentData() or None)
        crop = self.crops.get(str(self.current_path)) if self.current_path is not None else None
        return replace(settings, crop=crop) if crop else settings          # el encuadre entra en la clave de la caché

    def _update_params_label(self) -> None:
        """Pie de ventana: modelo · calidad · refinado."""
        if not hasattr(self, "params_label"):
            return
        preset = QUALITY_PRESETS.get(self.quality_combo.currentData())
        parts = [self.model_combo.currentText().split("  (")[0]]
        if preset:
            px = self.resolution_combo.currentData() or preset.processing_resolution
            parts.append(f"{QCoreApplication.translate('Depth', preset.label)} {px} px"
                         + (f" ×{preset.ensemble_size}" if preset.ensemble_size > 1 else ""))
        parts.append(self.tr("refinado") if self.refine_check.isChecked() else self.tr("sin refinar"))
        self.params_label.setText("  ·  ".join(parts) + "   |")

    def _enable_status_tips(self) -> None:
        """Ayuda contextual: al pasar el ratón por un control, su descripción sale en el pie."""
        for widget in self.findChildren(QWidget):
            if widget.toolTip() and not widget.statusTip():
                widget.setStatusTip(widget.toolTip().replace("\n", " · "))

    def _update_quality_hint(self) -> None:
        preset = QUALITY_PRESETS.get(self.quality_combo.currentData())
        self.quality_hint.setText(QCoreApplication.translate("Depth", preset.hint) if preset else "")

    # --- Modelos -------------------------------------------------------------
    def active_model(self) -> str:
        """Modelo activo guardado en la configuración (o el de por defecto si ya no existe)."""
        model_id = self.settings.value("depth/active_model", DEFAULT_MODEL)
        try:
            self.models.get(model_id)
            return model_id
        except ModelError:
            return DEFAULT_MODEL

    def _fill_model_combo(self) -> None:
        """Rellena el selector con la lista del gestor y marca el activo."""
        active = self.active_model()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        entries = [e for e in self.models.all() if self.models.is_enabled(e.id) or e.id == active]   # los desmarcados no salen
        if active and all(e.id != active for e in entries):      # el modelo activo es de prueba y está oculto: se mantiene
            try:
                entries.append(self.models.get(active))
            except ModelError:
                pass
        for entry in entries:
            label = entry.display_label
            if not can_compute(entry):
                label += self.tr("  (sin cálculo aún)")
            elif not self.models.is_installed(entry.id):
                if entry.source in ("local", "folder"):
                    label += self.tr("  (no disponible)")
                elif entry.warning or entry.dev_only:            # estos nunca se descargan solos
                    label += self.tr("  (no descargado)")
                else:
                    label += self.tr("  (se descargará)")
            self.model_combo.addItem(label, entry.id)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(active)))
        self.model_combo.blockSignals(False)

    def _on_model_combo_changed(self, _index: int) -> None:
        model_id = self.model_combo.currentData()
        if model_id:
            self.settings.setValue("depth/active_model", model_id)
            log.info("Modelo de profundidad activo: %s", model_id)
            self._on_depth_settings_changed()

    def _on_depth_settings_changed(self, *_args) -> None:
        """Modelo, calidad o refinado cambiados: se guardan y se muestra el mapa que corresponda."""
        if not hasattr(self, "refine_check"):  # aún construyendo el panel
            return
        self.settings.setValue("depth/quality", self.quality_combo.currentData())
        self.settings.setValue("depth/resolution", self.resolution_combo.currentData() or 0)
        self.settings.setValue("depth/refine", self.refine_check.isChecked())
        self._update_quality_hint()
        self._update_params_label()
        current =self._depth_settings().key()
        # Los mapas calculados con otros ajustes dejan de valer; los importados se conservan.
        for path in [p for p, src in self.depth_sources.items() if src not in ("external", current)]:
            self.depth_maps.pop(path, None)
            self.depth_sources.pop(path, None)
        if self.current_path is not None:
            self._load_cached_depth(self.current_path)
            self._sync_editor()
            self._show_current()

    def _set_active_model(self, model_id: str) -> None:
        self.settings.setValue("depth/active_model", model_id)
        self._fill_model_combo()
        self._on_model_combo_changed(self.model_combo.currentIndex())

    def memory_levels(self) -> tuple[str, str]:
        """(nivel de la VRAM, nivel de la RAM) del equipo: "ok" / "warn" / "bad"."""
        return self.memory_button.levels()

    def refresh_memory_indicator(self) -> None:
        self.memory_button.refresh()

    def free_memory(self) -> bool:
        """Descarga el modelo de profundidad y suelta las cachés que se rehacen solas (render, relieve 3D).

        No toca las fotos, los mapas ni los ajustes: el proyecto queda igual. El modelo se vuelve a cargar solo
        la próxima vez que se calcule una profundidad (unos segundos). Devuelve False si hay un cálculo en marcha."""
        if self.depth_worker.is_busy() or self.export_panel._worker.is_busy():
            self.statusBar().showMessage(self.tr("No se puede liberar memoria mientras se calcula o se exporta."), 5000)
            return False
        before_ram, before_gpu = memory.process_memory_mb(), memory.gpu_memory_mb()
        self.estimator.unload()
        self.editor.release_caches()
        self.view3d.release_caches()
        self._release_run_maps()
        if self._in_3d():
            self.view3d.update()
        memory.release_python_memory()
        after_ram, after_gpu = memory.process_memory_mb(), memory.gpu_memory_mb()
        parts = [self.tr("Memoria liberada.")]
        if before_ram is not None and after_ram is not None:
            parts.append(self.tr("RAM: {0:,.0f} → {1:,.0f} MB.").format(before_ram, after_ram))
        if before_gpu is not None and after_gpu is not None:
            parts.append(self.tr("Vídeo: {0:,.0f} → {1:,.0f} MB.").format(before_gpu, after_gpu))
        message = " ".join(parts)
        log.info(message)
        self.statusBar().showMessage(message, 8000)
        self.refresh_memory_indicator()
        return True

    def open_model_manager(self, then: str | None = None) -> None:
        """Abre el gestor. `then`: método del gestor a lanzar nada más abrirlo
        ("open_search" = añadir desde Hugging Face, "install_from_folder")."""
        dialog = ModelManagerDialog(self.models, self.estimator, self.active_model(),
                                    self.depth_worker.is_busy, self)
        dialog.active_changed.connect(self._set_active_model)
        dialog.models_changed.connect(self._fill_model_combo)
        if then:
            QTimer.singleShot(0, getattr(dialog, then))
        dialog.exec()
        self._fill_model_combo()  # refleja descargas hechas desde el diálogo

    def _confirm_memory_for_depth(self) -> bool:
        """Preaviso en rojo: si la memoria de vídeo o la RAM están casi llenas, ofrece liberarla antes de calcular.
        Devuelve False si el usuario cancela."""
        if "bad" not in self.memory_levels():
            return True
        box = QMessageBox(QMessageBox.Warning, self.tr("Memoria casi llena"),
                          self.tr("Queda muy poca memoria libre (VRAM o RAM). Calcular ahora puede fallar o ir muy lento."),
                          QMessageBox.NoButton, self)
        box.setInformativeText(self.tr("Puedes liberar memoria (descarga el modelo y suelta cachés; el proyecto no cambia) "
                                       "y continuar."))
        free = box.addButton(self.tr("Liberar memoria y continuar"), QMessageBox.AcceptRole)
        go = box.addButton(self.tr("Continuar igualmente"), QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is free:
            self.free_memory()
            return True
        return box.clickedButton() is go

    def compute_depth(self) -> None:
        if self.current_path is None:
            self.statusBar().showMessage(self.tr("Selecciona primero una fotografía."), 4000)
            return
        if not self._confirm_memory_for_depth():
            return
        if not self.depth_worker.start(self.current_path, self._depth_settings()):
            self.statusBar().showMessage(self.tr("Ya hay un cálculo en marcha."), 4000)
            return
        settings = self._depth_settings()
        self._depth_job_settings = settings
        cold = self.estimator.loaded_model != settings.model        # cargar el modelo añade varios segundos
        self._depth_timer_key = (f"depth/{settings.model}/{settings.processing_resolution}x{settings.ensemble_size}/"
                                 f"{'cold' if cold else 'warm'}")
        self._depth_message = self.tr("Preparando…")
        self._depth_timer.start(self.durations.expected(self._depth_timer_key))
        self.depth_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.depth_progress.setRange(0, 0)  # indeterminada hasta que haya un porcentaje
        self.depth_progress.show()
        self.depth_status.setText(self.tr("Preparando…"))
        self.depth_status.show()

    def _on_depth_progress(self, message: str, fraction: float) -> None:
        if fraction < 0:
            self.depth_progress.setRange(0, 0)
        else:
            self.depth_progress.setRange(0, 1000)
            self.depth_progress.setValue(int(fraction * 1000))
        self._depth_timer.update(fraction)
        self._depth_message = message
        self.depth_status.setText(with_time(message, self._depth_timer))
        self.statusBar().showMessage(message)

    def _end_depth_job(self, success: bool = False) -> None:
        seconds = self._depth_timer.stop()
        if success and self._depth_timer_key:
            self.durations.record(self._depth_timer_key, seconds)     # para estimar la próxima vez
        self.depth_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.depth_progress.hide()
        self.depth_status.hide()

    def _on_depth_finished(self, path: str, depth: np.ndarray) -> None:
        self._end_depth_job(success=True)
        if not self._in_library(path):
            return       # la foto se quitó de la biblioteca mientras se calculaba
        settings = self._depth_job_settings or self._depth_settings()
        previous_key = self.depth_sources.get(path, "")              # el cálculo que se usaba antes de este
        self.depth_maps[path] = depth
        self.depth_sources[path] = settings.key()
        run = self.estimator.last_run
        self._register_run(path, settings, depth, run)
        if self.current_path is not None and str(self.current_path) == path:
            self._sync_editor()
        took = f" · {depth_runs.format_seconds(run.seconds)}" if run is not None and not run.from_cache else ""
        self.statusBar().showMessage(self.tr("Profundidad lista: {0}{1}").format(Path(path).name, took), 8000)
        self._show_result_time(run, settings)
        self._refresh_runs_list()
        if self.current_path is not None and str(self.current_path) == path:
            # Con dos o más cálculos de la foto se queda en la vista Comparar: A = el que se usaba antes, B = el nuevo.
            # Así se ve enseguida la diferencia entre modelos o calidades, sin activarlo a mano.
            keys = [e.key for e in self._ab_run_entries()]
            if len(keys) >= 2:
                # A: el cálculo que se usaba antes; si no se sabe cuál era, el más reciente de los otros.
                before = previous_key if previous_key in keys and previous_key != settings.key() else \
                    next((k for k in reversed(keys) if k != settings.key()), "")
                if before:
                    self._ab_updating = True
                    self.ab_a_combo.setCurrentIndex(max(0, self.ab_a_combo.findData("run:" + before)))
                    self.ab_b_combo.setCurrentIndex(max(0, self.ab_b_combo.findData("run:" + settings.key())))
                    self._ab_updating = False
                self.view_switch.set_current_index(VIEW_COMPARE)
            else:
                self.view_switch.set_current_index(VIEW_DEPTH)
            self._show_current()

    def _on_depth_failed(self, path: str, message: str, detail: str = "") -> None:
        self._end_depth_job()
        self.statusBar().showMessage(self.tr("No se pudo calcular la profundidad."), 5000)
        box = QMessageBox(QMessageBox.Warning, self.tr("Profundidad"), message, QMessageBox.Ok, self)
        box.setInformativeText(self.tr("Foto: {0}").format(Path(path).name))
        if detail:
            box.setDetailedText(detail)   # traza técnica: botón «Mostrar detalles…»
        retry = None
        if "out of memory" in detail.lower():
            retry = box.addButton(self.tr("Liberar memoria y reintentar"), QMessageBox.AcceptRole)
        box.exec()
        if retry is not None and box.clickedButton() is retry:
            if self.free_memory():
                self.compute_depth()

    def _in_library(self, path: str) -> bool:
        return any(self.library.item(i).data(Qt.UserRole) == path for i in range(self.library.count()))

    def import_depth_map(self) -> None:
        """Usa un mapa hecho fuera de la app para la foto seleccionada."""
        if self.current_path is None or self.current_photo is None:
            self.statusBar().showMessage(self.tr("Selecciona primero la fotografía a la que corresponde el mapa."), 4000)
            return
        start_dir = self.settings.value("last_depth_dir", str(self.current_path.parent))
        file, _ = QFileDialog.getOpenFileName(self, self.tr("Cargar mapa de profundidad"), start_dir, DEPTH_FILTER)
        if not file:
            return
        self.settings.setValue("last_depth_dir", str(Path(file).parent))
        try:
            depth = load_external_depth(Path(file), near_is_bright=self.near_bright_check.isChecked())
        except OSError as exc:
            log.warning("No se pudo abrir el mapa %s: %s", file, exc)
            QMessageBox.warning(self, "Cargar mapa", f"No se pudo abrir «{Path(file).name}».")
            return
        crop = self.crops.get(str(self.current_path))
        if crop is not None:                    # el mapa es de la foto entera: se encuadra igual que la foto
            full = self._full_size()
            if (depth.shape[1], depth.shape[0]) != full:
                depth = np.clip(np.asarray(Image.fromarray(depth, mode="F").resize(full, Image.BICUBIC), dtype=np.float32), 0.0, 1.0)
            left, top, right, bottom = framing.to_box(crop, full)
            depth = np.ascontiguousarray(depth[top:bottom, left:right])
        # Si el mapa no tiene el tamaño de la foto, se reescala para que coincidan píxel a píxel.
        size = (self.current_photo.width(), self.current_photo.height())
        if (depth.shape[1], depth.shape[0]) != size:
            log.info("Mapa reescalado de %d×%d a %d×%d", depth.shape[1], depth.shape[0], *size)
            depth = np.asarray(Image.fromarray(depth, mode="F").resize(size, Image.BICUBIC), dtype=np.float32)
            depth = np.clip(depth, 0.0, 1.0)
        self.depth_maps[str(self.current_path)] = depth
        self.depth_sources[str(self.current_path)] = "external"
        self._sync_editor()
        self._refresh_runs_list()
        log.info("Mapa importado para %s: %s", self.current_path.name, file)
        self.view_switch.set_current_index(VIEW_DEPTH)
        self._show_current()

    # --- Arrastrar y soltar archivos sobre la ventana ---
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        self.add_photos(paths)

    # =====================================================================
    # Otros
    # =====================================================================
    def show_about(self) -> None:
        from deepshot.ui.about import AboutDialog

        AboutDialog({"app": APP_NAME, "tagline": self.tr("the art of depthing"), "ver": __version__,
                     "url": WEBSITE, "org": ORG_NAME, "author": AUTHOR, "date": RELEASE_DATE}, self).exec()

    def _restore_state(self) -> None:
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        else:
            self.resize(1400, 860)
        if isinstance(state, QByteArray):
            self.restoreState(state)
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Asegura que la ventana cabe en la pantalla disponible (sin barra de tareas).

        Con el escalado de Windows (125-150 % en portátiles) 1400×860 puede no caber,
        y una posición guardada puede quedar fuera si cambió la pantalla.
        """
        if self.isMaximized() or self.isFullScreen():
            return
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        frame_extra = self.frameGeometry().size() - self.geometry().size()  # barra de título y bordes
        width = min(self.width(), area.width() - frame_extra.width())
        height = min(self.height(), area.height() - frame_extra.height())
        self.resize(max(width, self.minimumWidth()), max(height, self.minimumHeight()))
        frame = self.frameGeometry()
        if not area.contains(frame):
            frame.moveCenter(area.center())
            frame.moveLeft(max(frame.left(), area.left()))
            frame.moveTop(max(frame.top(), area.top()))
            self.move(frame.topLeft())

    def closeEvent(self, event) -> None:
        if self.ask_before_closing and not self._confirm_discard(self.tr("Salir")):
            event.ignore()                       # Cancelar (o se canceló el guardado): la ventana sigue abierta
            return
        QApplication.instance().removeEventFilter(self._hover_help)
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.log_panel.detach()
        self._preloader.wait(30)                 # no cerrar con un modelo a medio cargar
        self.editor.shutdown()
        super().closeEvent(event)
