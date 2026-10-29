"""Mapa de profundidad rápido: elegir fotos, calcular y exportar solo el mapa (de profundidad y/o de normales).

Es el modo sencillo de la aplicación: sin proyecto, sin rampas ni áreas. Usa el mismo modelo, calidad y resolución
que el modo completo (se copian los selectores de la ventana principal) y su estimador, así que los mapas ya
calculados salen de la caché. El trabajo va en un hilo; los resultados se guardan en la carpeta elegida.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QSettings, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deepshot import simple_depth
from deepshot.depth import DEFAULT_MODEL, DEFAULT_QUALITY, settings_for
from deepshot.ui.depth_errors import friendly_error
from deepshot import depth_runs
from deepshot.ui.icons import set_icon
from deepshot.ui.memory_indicator import MemoryIndicator
from deepshot.ui.task_timer import TaskTimer, with_time
from deepshot.ui.quick_viewer import IMAGE_SUFFIXES, QuickViewer
from deepshot.ui.widgets import HoverHelp

log = logging.getLogger(__name__)

PHOTO_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"


class _SimpleWorker(QObject):
    """Calcula y guarda, una foto tras otra, en un hilo. Las señales llegan al hilo principal."""

    progress = Signal(str, float, int, int)      # mensaje, fracción (-1 = sin medir), foto actual, total
    saved = Signal(str)                          # ruta escrita
    photo_done = Signal(str, str)                # foto, resumen del cálculo («Listo en 4,5 s · modelo · calidad»)
    failed = Signal(str, str)                    # foto, mensaje
    finished = Signal(int, int)                  # fotos hechas, total

    def __init__(self, estimator, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.estimator = estimator
        self._cancel = threading.Event()
        self.busy = False

    def start(self, photos: list[Path], depth_settings, options: simple_depth.SimpleOptions) -> None:
        self._cancel.clear()
        self.busy = True
        threading.Thread(target=self._run, args=(photos, depth_settings, options), daemon=True).start()

    def cancel(self) -> None:
        self._cancel.set()

    def _summary(self, settings) -> str:
        """«Listo en 4,5 s · Modelo · Máxima 1280 px ×3» del último cálculo (con «(de la caché)» si se leyó de ella)."""
        run = getattr(self.estimator, "last_run", None)
        try:
            model = self.estimator.manager.get(settings.model).display_label
        except Exception:
            model = settings.model
        seconds = depth_runs.format_seconds(run.seconds) if run is not None else "—"
        text = QCoreApplication.translate("SimpleDepth", "Listo en {0} · {1} · {2}").format(
            seconds, model, depth_runs.quality_label(settings))
        if run is not None and run.from_cache:
            text += QCoreApplication.translate("SimpleDepth", "  (de la caché)")
        elif run is not None and run.load_seconds:
            text += QCoreApplication.translate("SimpleDepth", "  (+{0} cargando el modelo)").format(
                depth_runs.format_seconds(run.load_seconds))
        return text

    def _run(self, photos: list[Path], depth_settings, options: simple_depth.SimpleOptions) -> None:
        done = 0
        try:
            for i, photo in enumerate(photos, start=1):
                if self._cancel.is_set():
                    break
                try:
                    depth = self.estimator.estimate(
                        photo, depth_settings,
                        lambda message, fraction, i=i: self.progress.emit(
                            message, -1.0 if fraction is None else float(fraction), i, len(photos)))
                    self.photo_done.emit(photo.name, self._summary(depth_settings))
                    self.progress.emit(QCoreApplication.translate("SimpleDepth", "Guardando…"), 1.0, i, len(photos))
                    for path in simple_depth.export_simple(depth, photo.name, options):
                        self.saved.emit(str(path))
                    done += 1
                except Exception as exc:                                  # una foto que falla no detiene las demás
                    log.exception("Fallo en el mapa rápido de %s", photo)
                    self.failed.emit(photo.name, friendly_error(exc) + "\n\n" + traceback.format_exc(limit=0))
        finally:
            self.busy = False
            self.finished.emit(done, len(photos))


class _HintBar:
    """Hace de «barra de estado» de la ventana para `HoverHelp`: la ayuda de cada control sale en una etiqueta."""

    def __init__(self, label: QLabel) -> None:
        self._label = label

    def showMessage(self, text: str, _timeout: int = 0) -> None:
        self._label.setText(text)

    def clearMessage(self) -> None:
        self._label.clear()

    def currentMessage(self) -> str:
        return self._label.text()


class QuickHost:
    """Lo que el Mapa rápido necesita de quien lo abre. Así no depende de la ventana principal y puede ir también
    en la aplicación básica por separado (`deepshot/quick_app.py`).

    estimator            calcula la profundidad (`DepthEstimator`)
    model_combo, quality_combo, resolution_combo   selectores de los que se copian las opciones y la elección
    library_paths()      fotos que ya hay importadas (lista vacía si no hay biblioteca)
    is_busy()            True si el anfitrión está calculando otra profundidad (no se calcula a la vez)
    show_message(texto)  aviso breve al terminar (la barra de estado, si la hay)
    """

    def __init__(self, estimator, model_combo: QComboBox, quality_combo: QComboBox, resolution_combo: QComboBox,
                 library_paths=lambda: [], is_busy=lambda: False, show_message=lambda _text: None,
                 free_memory=None, open_model_manager=None) -> None:
        self.estimator = estimator
        self._free_memory = free_memory
        self.open_model_manager = open_model_manager    # abre el gestor de modelos (None: el anfitrión no lo ofrece)
        self.model_combo, self.quality_combo, self.resolution_combo = model_combo, quality_combo, resolution_combo
        self.library_paths, self.is_busy, self.show_message = library_paths, is_busy, show_message

    def free_memory(self) -> None:
        """Libera la memoria: la del anfitrión si la trae, y si no, descarga el modelo y suelta lo que Python retiene."""
        if self._free_memory is not None:
            self._free_memory()
            return
        from deepshot import memory

        self.estimator.unload()
        memory.release_python_memory()

    @classmethod
    def from_window(cls, window) -> "QuickHost":
        """El anfitrión de la ventana principal de Deepshot Studio."""
        library = window.library
        return cls(window.estimator, window.model_combo, window.quality_combo, window.resolution_combo,
                   library_paths=lambda: [library.item(i).data(Qt.UserRole) for i in range(library.count())],
                   is_busy=window.depth_worker.is_busy,
                   show_message=lambda text: window.statusBar().showMessage(text, 8000),
                   free_memory=window.free_memory, open_model_manager=window.open_model_manager)


class SimpleDepthDialog(QDialog):
    """Ventana del modo sencillo. `host` es un `QuickHost` o la ventana principal (de la que se crea uno)."""

    def __init__(self, host, parent: QWidget | None = None) -> None:
        super().__init__(parent or (host if isinstance(host, QWidget) else None))
        host = host if isinstance(host, QuickHost) else QuickHost.from_window(host)
        self.host = host
        self.settings = QSettings()
        self.setWindowTitle(self.tr("Mapa de profundidad rápido"))
        self.setMinimumWidth(560)
        self._worker = _SimpleWorker(host.estimator, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.saved.connect(self._on_saved)
        self._worker.photo_done.connect(self._on_photo_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._written: list[str] = []
        self._errors: list[str] = []

        intro = QLabel(self.tr("Elige una o varias fotos y obtén solo el mapa de profundidad (o de normales), "
                               "sin abrir el editor."))
        intro.setWordWrap(True)
        intro.setProperty("dim", True)

        # --- Fotos ---
        self.photos = QListWidget()
        self.photos.setSelectionMode(QListWidget.ExtendedSelection)
        self.photos.setFixedHeight(110)
        self.photos.setToolTip(self.tr("Fotos de las que se hará el mapa"))
        self.add_button = QPushButton(self.tr("Añadir fotos…"))
        self.add_button.setToolTip(self.tr("Busca fotografías en tu equipo"))
        set_icon(self.add_button, "import")
        self.add_button.clicked.connect(self.add_photos_dialog)
        self.library_button = QPushButton(self.tr("Usar las de la biblioteca"))
        set_icon(self.library_button, "photo")
        self.library_button.setToolTip(self.tr("Añade las fotos que ya están importadas en la ventana principal"))
        self.library_button.clicked.connect(self.add_library_photos)
        self.remove_button = QPushButton(self.tr("Quitar"))
        set_icon(self.remove_button, "minus")
        self.remove_button.setToolTip(self.tr("Quita de la lista las fotos seleccionadas"))
        self.remove_button.clicked.connect(self.remove_selected)
        photo_buttons = QHBoxLayout()
        for b in (self.add_button, self.library_button, self.remove_button):
            photo_buttons.addWidget(b)
        photo_buttons.addStretch()

        # --- Cálculo: se copian los selectores de la ventana principal ---
        self.model_combo = self._copy_combo(host.model_combo)
        self.model_combo.setToolTip(self.tr("Modelo que calcula la profundidad"))
        self.quality_combo = self._copy_combo(host.quality_combo)
        self.quality_combo.setToolTip(self.tr("Calidad: más resolución y pasadas = más detalle, pero más tiempo"))
        self.resolution_combo = self._copy_combo(host.resolution_combo)
        self.resolution_combo.setToolTip(host.resolution_combo.toolTip())
        self.models_button = QPushButton("…")           # el gestor de modelos, sin entrar en la aplicación completa
        self.models_button.setFixedWidth(28)
        self.models_button.setStyleSheet("padding: 0;")
        self.models_button.setToolTip(self.tr("Gestor de modelos: descargar, instalar, borrar y elegir el modelo"))
        self.models_button.clicked.connect(lambda _checked=False: self.manage_models())
        self.models_button.setVisible(host.open_model_manager is not None)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.models_button)
        calc = QFormLayout()
        calc.addRow(self.tr("Modelo"), model_row)
        calc.addRow(self.tr("Calidad"), self.quality_combo)
        calc.addRow(self.tr("Resolución"), self.resolution_combo)

        # --- Resultado ---
        self.depth_check = QCheckBox(self.tr("Mapa de profundidad"))
        self.depth_check.setChecked(self.settings.value("simple/depth", True, type=bool))
        self.depth_check.setToolTip(self.tr("Imagen en gris con la distancia de cada punto"))
        self.normalize_check = QCheckBox(self.tr("Normalizar (usar todo el rango)"))
        self.normalize_check.setChecked(self.settings.value("simple/normalize", True, type=bool))
        self.normalize_check.setToolTip(self.tr("Estira el mapa para que use del negro al blanco completos"))
        self.near_check = QCheckBox(self.tr("Cerca = claro"))
        self.near_check.setChecked(self.settings.value("simple/near_bright", True, type=bool))
        self.near_check.setToolTip(self.tr("Marcado: lo cercano sale claro y lo lejano oscuro. Desmarcado: al revés "
                                           "(mapa invertido)"))
        self.normal_check = QCheckBox(self.tr("Mapa de normales"))
        self.normal_check.setChecked(self.settings.value("simple/normal", False, type=bool))
        self.normal_check.setToolTip(self.tr("Mapa de normales (color) calculado a partir de la profundidad, para "
                                             "motores 3D y materiales"))
        self.strength = QDoubleSpinBox()
        self.strength.setRange(simple_depth.MIN_STRENGTH, simple_depth.MAX_STRENGTH)
        self.strength.setDecimals(2)
        self.strength.setSingleStep(0.1)
        self.strength.setValue(self.settings.value("simple/strength", simple_depth.DEFAULT_STRENGTH, type=float))
        self.strength.setToolTip(self.tr("Intensidad del relieve de las normales: más alta = más inclinadas"))
        self.convention = QComboBox()
        self.convention.addItem(self.tr("OpenGL (Y+ arriba: Blender, Unity)"), "opengl")
        self.convention.addItem(self.tr("DirectX (Y+ abajo: Unreal)"), "directx")
        self.convention.setCurrentIndex(max(0, self.convention.findData(self.settings.value("simple/convention", "opengl"))))
        self.convention.setToolTip(self.tr("Hacia dónde apunta el canal verde; elige el de tu programa"))
        results = QGroupBox(self.tr("Qué crear"))
        grid = QFormLayout(results)
        grid.addRow(self.depth_check)
        depth_options = QHBoxLayout()
        depth_options.setContentsMargins(18, 0, 0, 0)
        depth_options.addWidget(self.normalize_check)
        depth_options.addWidget(self.near_check)
        grid.addRow(depth_options)
        grid.addRow(self.normal_check)
        grid.addRow(self.tr("Intensidad"), self.strength)
        grid.addRow(self.tr("Convención"), self.convention)

        # --- Salida ---
        self.format_combo = QComboBox()
        for key, label in (("png16", self.tr("PNG 16 bits")), ("png8", self.tr("PNG 8 bits")),
                           ("exr32", self.tr("EXR 32 bits"))):
            self.format_combo.addItem(label, key)
        self.format_combo.setCurrentIndex(max(0, self.format_combo.findData(self.settings.value("simple/format", "png16"))))
        self.format_combo.setToolTip(self.tr("Formato del archivo. 16 bits conserva los degradados suaves."))
        self.folder_edit = QLineEdit(str(self.settings.value("simple/folder", str(Path.home() / "Pictures"))))
        self.folder_edit.setToolTip(self.tr("Carpeta donde se guardan los mapas"))
        browse = QPushButton()                                    # el mismo icono de carpeta que en la exportación
        set_icon(browse, "folder")
        browse.setIconSize(QSize(16, 16))
        browse.setFixedWidth(30)
        browse.setToolTip(self.tr("Elegir la carpeta de salida…"))
        browse.clicked.connect(self._choose_folder)
        self.open_button = QPushButton(self.tr("Abrir carpeta"))
        self.open_button.setToolTip(self.tr("Abre la carpeta de salida en el explorador de archivos"))
        self.open_button.clicked.connect(self.open_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse)
        folder_row.addWidget(self.open_button)
        self.json_check = QCheckBox(self.tr("Guardar también los datos (.json)"))
        self.json_check.setChecked(self.settings.value("simple/json", False, type=bool))
        self.json_check.setToolTip(self.tr("Guarda junto a cada imagen un archivo .json con el modelo, la calidad y "
                                           "las opciones con que se hizo"))
        output = QFormLayout()
        output.addRow(self.tr("Formato"), self.format_combo)
        output.addRow(self.tr("Carpeta"), folder_row)
        output.addRow(self.json_check)

        # --- Ejecutar ---
        self.run_button = QPushButton(self.tr("Crear y exportar"))
        self.run_button.setObjectName("Primary")
        set_icon(self.run_button, "export")
        self.run_button.setIconSize(QSize(16, 16))
        self.run_button.setToolTip(self.tr("Calcula la profundidad de cada foto y guarda los mapas en la carpeta"))
        self.run_button.clicked.connect(self.run)
        self.review_button = QPushButton(self.tr("Revisar"))
        set_icon(self.review_button, "eye")
        self.review_button.setEnabled(False)
        self.review_button.setToolTip(self.tr("Abre los mapas recién creados en un visor rápido, con zoom y sus datos"))
        self.review_button.clicked.connect(self.review)
        self.close_button = QPushButton(self.tr("Cerrar"))
        self.close_button.setToolTip(self.tr("Cierra esta ventana (si hay un cálculo en marcha, lo cancela al terminar la foto)"))
        self.close_button.clicked.connect(self.reject)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("dim", True)
        self._timer = TaskTimer(self)                # tiempo transcurrido y estimado de cada foto
        self._timer.changed.connect(lambda _t: self.status.setText(with_time(self._message, self._timer)))
        self._message = ""
        self._current_index = 0
        self.result_label = QLabel("")               # lo último que se hizo, en verde (como en el panel de la ventana principal)
        self.result_label.setProperty("tone", "ok")
        self.result_label.setWordWrap(True)
        self.result_label.hide()
        self.results = QListWidget()                 # un renglón por foto: cuánto tardó, con qué modelo y calidad
        self.results.setFixedHeight(90)
        self.results.setToolTip(self.tr("Lo que se ha hecho en esta sesión: foto, tiempo, modelo y calidad"))
        self.results.hide()
        self.memory_button = MemoryIndicator()
        self.memory_button.clicked.connect(lambda _checked=False: self.free_memory())
        self.hint = QLabel("")                    # ayuda del control bajo el ratón (como el pie de la ventana principal)
        self.hint.setWordWrap(True)
        self.hint.setProperty("dim", True)
        # Alto fijo (dos líneas) y ancho que no empuja: si la ayuda cambiara el tamaño de la etiqueta, la ventana entera se
        # recolocaría cada vez que el ratón pasa por un control (parpadeo).
        self.hint.setFixedHeight(self.hint.fontMetrics().lineSpacing() * 2 + 4)
        self.hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._hover_help = HoverHelp(self)
        QApplication.instance().installEventFilter(self._hover_help)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.review_button)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.run_button)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        for item in (intro, self.photos):
            layout.addWidget(item)
        layout.addLayout(photo_buttons)
        layout.addLayout(calc)
        layout.addWidget(results)
        layout.addLayout(output)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.result_label)
        layout.addWidget(self.results)
        layout.addLayout(buttons)
        footer = QHBoxLayout()                       # el mismo indicador de memoria que el pie de la ventana principal
        footer.addWidget(self.hint, 1)
        footer.addWidget(self.memory_button)
        layout.addLayout(footer)
        self.normal_check.toggled.connect(self._update_enabled)
        self.depth_check.toggled.connect(self._update_enabled)
        self._update_enabled()

    def statusBar(self) -> _HintBar:
        """Lo que `HoverHelp` espera de una ventana: aquí la ayuda sale en la etiqueta de abajo."""
        return _HintBar(self.hint)

    def done(self, result: int) -> None:
        QApplication.instance().removeEventFilter(self._hover_help)
        super().done(result)

    def review(self) -> None:
        """Abre los mapas recién creados en el visor de revisión rápida."""
        images = [p for p in self._written if Path(p).suffix.lower() in IMAGE_SUFFIXES]
        if images:
            QuickViewer(images, self).exec()

    def manage_models(self) -> None:
        """Abre el gestor de modelos y, al cerrarlo, rehace el selector de modelo con lo que haya cambiado."""
        if self._worker.busy or self.host.open_model_manager is None:
            return
        chosen = self.model_combo.currentData()
        self.host.open_model_manager()
        self.reload_models(prefer=self.host.model_combo.currentData() or chosen)

    def reload_models(self, prefer: str | None = None) -> None:
        """Vuelve a copiar la lista de modelos del anfitrión (tras instalar, borrar o elegir uno en el gestor)."""
        keep = prefer or self.model_combo.currentData()
        self.model_combo.clear()
        source = self.host.model_combo
        for i in range(source.count()):
            self.model_combo.addItem(source.itemText(i), source.itemData(i))
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(keep)))

    def free_memory(self) -> None:
        """Libera la memoria del modelo (lo que pide el indicador). El anfitrión puede traer su propia versión."""
        if self._worker.busy:
            self.status.setText(self.tr("No se puede liberar memoria mientras se calcula."))
            return
        self.host.free_memory()
        self.memory_button.refresh()
        self.status.setText(self.tr("Memoria liberada."))

    def open_folder(self) -> None:
        folder = Path(self.folder_edit.text().strip() or ".")
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.status.setText(self.tr("La carpeta todavía no existe: se creará al crear los mapas."))

    # --- Utilidades de interfaz -------------------------------------------------------------------
    @staticmethod
    def _copy_combo(source: QComboBox) -> QComboBox:
        """Un selector con las mismas opciones y la misma elección que el de la ventana principal."""
        combo = QComboBox()
        for i in range(source.count()):
            combo.addItem(source.itemText(i), source.itemData(i))
        combo.setCurrentIndex(source.currentIndex())
        return combo

    def _update_enabled(self) -> None:
        depth_on, normal_on = self.depth_check.isChecked(), self.normal_check.isChecked()
        self.normalize_check.setEnabled(depth_on or normal_on)
        self.near_check.setEnabled(depth_on)
        self.strength.setEnabled(normal_on)
        self.convention.setEnabled(normal_on)
        self.run_button.setEnabled(not self._worker.busy and (depth_on or normal_on) and self.photos.count() > 0)

    def photo_paths(self) -> list[Path]:
        return [Path(self.photos.item(i).data(Qt.UserRole)) for i in range(self.photos.count())]

    def add_photos(self, paths: list[str | Path]) -> None:
        """Añade fotos a la lista sin repetir."""
        known = {str(p) for p in self.photo_paths()}
        for p in paths:
            if str(p) not in known:
                self.photos.addItem(Path(p).name)
                item = self.photos.item(self.photos.count() - 1)
                item.setData(Qt.UserRole, str(p))
                item.setToolTip(str(p))
                known.add(str(p))
        self._update_enabled()

    def add_photos_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("Fotografías"), str(Path.home() / "Pictures"), PHOTO_FILTER)
        self.add_photos(files)

    def add_library_photos(self) -> None:
        self.add_photos(self.host.library_paths())

    def remove_selected(self) -> None:
        for item in self.photos.selectedItems():
            self.photos.takeItem(self.photos.row(item))
        self._update_enabled()

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.tr("Carpeta de salida"), self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    # --- Opciones y ejecución -----------------------------------------------------------------------
    def options(self) -> simple_depth.SimpleOptions:
        settings, quality = self.depth_settings(), self.quality_combo.currentData() or DEFAULT_QUALITY
        return simple_depth.SimpleOptions(
            depth_map=self.depth_check.isChecked(), normal=self.normal_check.isChecked(),
            normalize=self.normalize_check.isChecked(), near_bright=self.near_check.isChecked(),
            strength=self.strength.value(), convention=self.convention.currentData(),
            format=self.format_combo.currentData(), folder=Path(self.folder_edit.text().strip() or "."),
            tag=simple_depth.file_tag(settings.model, quality, self.resolution_combo.currentData() or None),
            save_json=self.json_check.isChecked(),
            info={"model": settings.model, "quality": quality, "processing_resolution": settings.processing_resolution,
                  "ensemble_size": settings.ensemble_size, "seed": settings.seed, "app": "Deepshot Studio"})

    def depth_settings(self):
        return settings_for(model=self.model_combo.currentData() or DEFAULT_MODEL,
                            quality=self.quality_combo.currentData() or DEFAULT_QUALITY,
                            resolution=self.resolution_combo.currentData() or None)

    def _remember(self) -> None:
        o = self.options()
        for key, value in (("depth", o.depth_map), ("normalize", o.normalize), ("near_bright", o.near_bright),
                           ("normal", o.normal), ("strength", o.strength), ("convention", o.convention),
                           ("format", o.format), ("folder", str(o.folder)), ("json", o.save_json)):
            self.settings.setValue(f"simple/{key}", value)

    def run(self) -> None:
        if self._worker.busy or not self.photo_paths():
            return
        if self.host.is_busy():
            QMessageBox.information(self, self.windowTitle(),
                                    self.tr("Espera a que termine el cálculo de profundidad de la ventana principal."))
            return
        self._remember()
        self._written, self._errors = [], []
        self.results.clear()
        self.results.hide()
        self.result_label.hide()
        self._current_index = 0
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status.setText(self.tr("Preparando…"))
        self.run_button.setEnabled(False)
        self._worker.start(self.photo_paths(), self.depth_settings(), self.options())

    def _on_progress(self, message: str, fraction: float, index: int, total: int) -> None:
        if fraction < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
        if index != self._current_index:                # foto nueva: el cronómetro empieza de cero
            self._current_index = index
            self._timer.stop()
            self._timer.start()
        self._timer.update(None if fraction < 0 else fraction)
        prefix = f"[{index}/{total}] " if total > 1 else ""
        self._message = prefix + message
        self.status.setText(with_time(self._message, self._timer))
        self.memory_button.refresh()

    def _on_photo_done(self, name: str, summary: str) -> None:
        self.result_label.setText(f"{name}: {summary}")
        self.result_label.show()
        self.results.addItem(f"{name} — {summary}")
        self.results.show()
        self.results.scrollToBottom()

    def _on_saved(self, path: str) -> None:
        self._written.append(path)

    def _on_failed(self, name: str, message: str) -> None:
        self._errors.append(f"{name}: {message}")

    def _on_finished(self, done: int, total: int) -> None:
        self._timer.stop()
        self.progress.hide()
        self._update_enabled()
        self.review_button.setEnabled(any(Path(p).suffix.lower() in IMAGE_SUFFIXES for p in self._written))
        if self._errors:
            QMessageBox.warning(self, self.windowTitle(), "\n\n".join(self._errors))
        self.status.setText(self.tr("Listo: {0} de {1} foto(s). Archivos guardados: {2}.").format(
            done, total, len(self._written)))
        self.host.show_message(self.status.text())

    def reject(self) -> None:
        if self._worker.busy:
            self._worker.cancel()          # termina la foto en curso y no sigue con las demás
            self.status.setText(self.tr("Cancelando al terminar la foto actual…"))
            return
        super().reject()
