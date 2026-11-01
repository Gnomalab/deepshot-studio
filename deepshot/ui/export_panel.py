"""Panel del paso «Exportar»: qué guardar (efecto, mapa de profundidad o imagen final), formato, carpeta y nombre.

El trabajo pesado (procesar a resolución completa y guardar) va en un hilo con barra de progreso;
la lógica de archivos está en `deepshot/export.py`.
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QSize, QUrl, Qt, Signal, QCoreApplication
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from deepshot import export as ex
from deepshot.project import Project
from deepshot.ui.icons import icon, set_icon
from deepshot import presets
from deepshot.ui.preset_bar import PresetBar
from deepshot.ui.quick_viewer import IMAGE_SUFFIXES, QuickViewer
from deepshot.ui.task_timer import TaskTimer, with_time
from deepshot.ui.value_slider import ValueSlider
from deepshot.ui.widgets import SegmentedControl

log = logging.getLogger(__name__)

FORMAT_KEYS = ["png8", "png16", "exr32"]
CONTENT_KEYS = ["effect", "depth", "final"]
FINAL_FORMAT_KEYS = ["jpg", "tiff8", "tiff16", "png8", "png16", "webp"]      # formatos de la imagen final
SIZE_MODE_KEYS = ["original", "percent", "width", "height", "long_side"]
DEPTH_SOURCE_KEYS = ["original", "zones"]       # mapa original o con las zonas movidas del control de profundidad


def name_suffix(options: ex.ExportOptions) -> str:
    """Sufijo en inglés que describe cómo está hecho el archivo, para reconocerlo sin abrirlo: «_effect», «_final»,
    «_depth_gray», «_depth_gray_inverted», «_depth_thermal», «_depth_rainbow_zones»…"""
    if options.content == "effect":
        return "_effect"
    if options.content == "final":
        return "_final"
    parts = ["_depth", options.depth_palette or "gray"]
    if not options.depth_palette and not options.near_bright:
        parts.append("inverted")                  # en gris, lo lejano claro
    if options.depth_zones:
        parts.append("zones")                     # con las zonas del control de profundidad movidas
    return "_".join(parts)


class _Worker(QObject):
    """Ejecuta una exportación en un hilo y avisa con señales (se reciben en el hilo de la interfaz)."""

    progress = Signal(str, float)
    finished = Signal(list)
    failed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._busy = False

    def is_busy(self) -> bool:
        return self._busy

    def start(self, job: Callable[[ex.Progress], list[Path]]) -> bool:
        if self._busy:
            return False
        self._busy = True
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return True

    def _run(self, job) -> None:
        try:
            paths = job(lambda message, fraction: self._emit(self.progress, message, float(fraction)))
            self._emit(self.finished, [str(p) for p in paths])
        except Exception as exc:  # cualquier fallo se muestra al usuario; el detalle va al registro
            log.exception("Fallo al exportar")
            self._emit(self.failed, self._friendly(exc), traceback.format_exc())
        finally:
            self._busy = False

    @staticmethod
    def _emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            pass        # la ventana se cerró durante la exportación

    @staticmethod
    def _friendly(exc: Exception) -> str:
        if isinstance(exc, MemoryError):
            return QCoreApplication.translate("ExportPanel", "No hay memoria suficiente para procesar la imagen a resolución completa.")
        if isinstance(exc, PermissionError):
            return QCoreApplication.translate("ExportPanel", "No se puede escribir en esa carpeta. Elige otra.")
        if isinstance(exc, OSError):
            return QCoreApplication.translate("ExportPanel", "No se pudo guardar el archivo: {0}").format(exc.strerror or exc)
        if isinstance(exc, ImportError):
            return QCoreApplication.translate("ExportPanel", "Falta un componente para este formato (OpenEXR). Reinstala las dependencias.")
        return QCoreApplication.translate("ExportPanel", "No se pudo exportar. Consulta el panel Registro para ver el detalle.")


class ExportPanel(QWidget):
    """Controles de exportación. `context` devuelve lo que se va a exportar (o None si no hay nada):
    {"path": Path de la foto, "depth": mapa, "project": Project, "mask": máscara pintada o None}."""

    exported = Signal(list)              # rutas escritas
    failed = Signal(str, str)            # mensaje, detalle técnico
    previewChanged = Signal()            # cambió lo que se exporta: la vista previa debe enseñarlo

    PRESET_FIELDS = tuple(presets.EXPORT_DEFAULTS)

    def __init__(self, context: Callable[[], dict | None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._settings = QSettings()
        self._name_dirty = False
        self._last_dir: Path | None = None
        self._worker = _Worker(self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        def hint(text: str) -> QLabel:
            label = QLabel(text)
            label.setWordWrap(True)
            label.setProperty("dim", True)
            return label

        self.info = hint("")
        # Presets de exportación: de fábrica por tipo y los propios (misma barra que rampas y estilos).
        self.preset_bar = PresetBar("export", self.tr("Cargar preset de exportación…"), self.current_preset)
        self.preset_bar.combo.setToolTip(self.tr("Ajustes de exportación listos: elige uno y se rellenan el tipo, el "
                                                 "formato, el tamaño y la calidad (no cambian el nombre ni la carpeta)."))
        self.preset_bar.chosen.connect(self.apply_preset)
        self.content = SegmentedControl([self.tr("Efecto"), self.tr("Mapa de profundidad"), self.tr("Imagen final")])
        self.content.set_tooltips([
            self.tr("La imagen con la rampa, las bandas, las áreas y la mezcla aplicadas (sin el paso Look)"),
            self.tr("Solo el mapa de profundidad en gris (cerca = claro), sin ningún efecto"),
            self.tr("El efecto con las capas del paso Look aplicadas, en un formato de fotografía (JPG, TIFF…) y al "
                    "tamaño que elijas"),
        ])
        self.depth_source = SegmentedControl([self.tr("Original"), self.tr("Con zonas movidas")])
        self.depth_source.set_tooltips([
            self.tr("El mapa tal como lo calculó el modelo (o como lo cargaste)"),
            self.tr("El mapa con las zonas cerca / medio / lejos movidas, como lo leen la rampa, las bandas, "
                    "el desenfoque y las áreas (Control de profundidad)"),
        ])
        self.depth_hint = hint("")
        # Mapa en gris o coloreado con una paleta (la misma que la vista Profundidad).
        self.depth_color = QComboBox()
        self.depth_color.addItem(self.tr("Gris"), "")
        for text, key in ((self.tr("Color rojo / azul"), "red_blue"), (self.tr("Color arcoíris"), "rainbow"),
                          (self.tr("Color térmico"), "thermal"), (self.tr("Color océano"), "ocean")):
            self.depth_color.addItem(text, key)
        self.depth_color.setToolTip(self.tr("El mapa en gris o coloreado con una paleta (cerca → lejos). "
                                            "Solo cambia la imagen guardada, no el mapa."))
        self.format = SegmentedControl([self.tr("PNG 8 bits"), self.tr("PNG 16 bits"), self.tr("EXR 32 bits")])
        self.format.set_tooltips([
            self.tr("PNG de 8 bits: ligero, para publicar o ver en cualquier programa"),
            self.tr("PNG de 16 bits: mucha más precisión de tonos; ideal para retocar después"),
            self.tr("EXR de 32 bits en coma flotante: sin pérdida, para flujos profesionales"),
        ])
        self.format_hint = hint("")
        # --- Imagen final: formato de fotografía, tamaño, resolución y metadatos ---
        self.final_format = QComboBox()
        for text, key, tip in (
                (self.tr("JPG"), "jpg", self.tr("JPG: ligero, el más compatible; pierde algo de calidad según el ajuste")),
                (self.tr("TIFF 8 bits"), "tiff8", self.tr("TIFF de 8 bits comprimido sin pérdida: para imprimir o retocar")),
                (self.tr("TIFF 16 bits"), "tiff16", self.tr("TIFF de 16 bits sin comprimir: máxima precisión de tonos "
                                                          "para seguir retocando (pesa mucho)")),
                (self.tr("PNG 8 bits"), "png8", self.tr("PNG de 8 bits sin pérdida, para web y publicar")),
                (self.tr("PNG 16 bits"), "png16", self.tr("PNG de 16 bits sin pérdida, mucha precisión de tonos")),
                (self.tr("WebP"), "webp", self.tr("WebP: muy ligero para la web, con la calidad que elijas"))):
            self.final_format.addItem(text, key)
            self.final_format.setItemData(self.final_format.count() - 1, tip, Qt.ToolTipRole)
        self.final_format.setToolTip(self.tr("Formato del archivo de la imagen final"))
        self.quality = ValueSlider(self.tr("Calidad"), 92, 1, 100, decimals=0, step=1)
        self.quality.setToolTip(self.tr("Calidad de la compresión de JPG y WebP: más alto = mejor imagen y archivo más "
                                        "grande (92 es un buen punto de partida)"))
        self.size_mode = QComboBox()
        for text, key in ((self.tr("Tamaño original"), "original"), (self.tr("Porcentaje"), "percent"),
                          (self.tr("Ancho"), "width"), (self.tr("Alto"), "height"), (self.tr("Lado largo"), "long_side")):
            self.size_mode.addItem(text, key)
        self.size_mode.setToolTip(self.tr("Tamaño de la imagen exportada: el original o uno que elijas, siempre "
                                          "conservando la proporción"))
        self.size_value = QSpinBox()
        self.size_value.setRange(1, 30000)
        self.size_value.setValue(100)
        self.size_value.setToolTip(self.tr("Porcentaje o píxeles, según el modo elegido"))
        self.size_result = hint("")
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        self.dpi.setSuffix(self.tr(" ppp"))
        self.dpi.setToolTip(self.tr("Resolución que se anota en el archivo (para imprimir). No cambia los píxeles."))
        self.keep_metadata = QCheckBox(self.tr("Conservar metadatos de la foto"))
        self.keep_metadata.setToolTip(self.tr("Copia a la imagen exportada los datos EXIF de la foto original (cámara, "
                                              "fecha…). Desmarca para quitarlos."))
        self.final_box = QWidget()
        final_layout = QVBoxLayout(self.final_box)
        final_layout.setContentsMargins(0, 0, 0, 0)
        final_layout.setSpacing(4)
        size_row = QHBoxLayout()
        size_row.setSpacing(4)
        size_row.addWidget(self.size_mode, 1)
        size_row.addWidget(self.size_value)
        dpi_row = QHBoxLayout()
        dpi_row.setSpacing(4)
        dpi_row.addWidget(QLabel(self.tr("Resolución")))
        dpi_row.addWidget(self.dpi, 1)
        for widget in (QLabel(self.tr("Formato de la imagen")), self.final_format, self.quality):
            final_layout.addWidget(widget)
        final_layout.addWidget(QLabel(self.tr("Tamaño")))
        final_layout.addLayout(size_row)
        final_layout.addWidget(self.size_result)
        final_layout.addLayout(dpi_row)
        final_layout.addWidget(self.keep_metadata)
        self.near_bright = QCheckBox(self.tr("Cerca = claro"))
        self.near_bright.setToolTip(self.tr("Marcado: lo cercano sale claro y lo lejano oscuro (como se ve en la "
                                            "app). Desmarcado: al revés."))
        self.save_settings = QCheckBox(self.tr("Guardar también los ajustes (.json)"))
        self.save_settings.setToolTip(self.tr("Guarda junto a la imagen un archivo con todos los ajustes, para "
                                              "repetir el resultado o aplicarlo a otra foto (Archivo › Cargar estilo, Ctrl+L)"))

        self.name = QLineEdit()
        self.name.setToolTip(self.tr("Nombre del archivo, sin la extensión. Si ya existe, se añade un número: "
                                     "nunca se sobrescribe."))
        self.name.textEdited.connect(lambda _t: setattr(self, "_name_dirty", True))
        self.folder = QLineEdit()
        self.folder.setReadOnly(True)
        self.folder.setPlaceholderText(self.tr("Carpeta de salida"))
        self.folder.setToolTip(self.tr("Carpeta donde se guarda el archivo"))
        choose = QPushButton()
        set_icon(choose, "folder")
        choose.setIconSize(QSize(16, 16))
        choose.setFixedWidth(30)
        choose.setStyleSheet("padding: 0;")
        choose.setToolTip(self.tr("Elegir la carpeta de salida…"))
        choose.clicked.connect(self.choose_folder)
        self.open_button = QPushButton(self.tr("Abrir carpeta"))
        self.open_button.setToolTip(self.tr("Abre la carpeta de salida en el explorador de archivos"))
        self.open_button.clicked.connect(self.open_folder)

        self.review_button = QPushButton(self.tr("Revisar"))
        set_icon(self.review_button, "eye")
        self.review_button.setEnabled(False)
        self.review_button.setToolTip(self.tr("Abre lo último que se exportó en un visor rápido, con zoom y sus datos"))
        self.review_button.clicked.connect(self.review)
        self._last_paths: list[str] = []
        self.last_photo = ""                      # la foto de la última exportación (para anotar lo exportado)
        self.button = QPushButton(self.tr("Exportar"))
        self.button.setObjectName("Primary")
        set_icon(self.button, "export")
        self.button.setIconSize(QSize(16, 16))
        self.button.setToolTip(self.tr("Procesa la imagen a resolución completa y la guarda (Ctrl+E abre este paso)"))
        self.button.clicked.connect(self.export)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        self.status = hint("")
        self._timer = TaskTimer(self)               # tiempo transcurrido y estimado junto al mensaje
        self._message = ""
        self._timer.changed.connect(lambda _t: self.status.setText(with_time(self._message, self._timer)))

        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        name_row.addWidget(QLabel(self.tr("Nombre")))
        name_row.addWidget(self.name, 1)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(4)
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(choose)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.info)
        layout.addWidget(self.preset_bar)
        layout.addWidget(QLabel(self.tr("Qué exportar")))
        layout.addWidget(self.content)
        self.depth_source_label = QLabel(self.tr("Mapa de profundidad"))
        layout.addWidget(self.depth_source_label)
        layout.addWidget(self.depth_source)
        layout.addWidget(self.depth_hint)
        layout.addWidget(self.depth_color)
        self.format_label = QLabel(self.tr("Formato"))
        layout.addWidget(self.format_label)
        layout.addWidget(self.format)
        layout.addWidget(self.format_hint)
        layout.addWidget(self.final_box)
        layout.addWidget(self.near_bright)
        layout.addWidget(self.save_settings)
        layout.addLayout(name_row)
        layout.addLayout(folder_row)
        layout.addWidget(self.open_button)
        layout.addWidget(self.review_button)
        layout.addSpacing(4)
        layout.addWidget(self.button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self._restore()
        self.content.changed.connect(self._on_content_changed)
        self.depth_source.changed.connect(self._on_content_changed)
        self.depth_color.currentIndexChanged.connect(self._on_content_changed)
        self.format.changed.connect(self._on_format_changed)
        self.final_format.currentIndexChanged.connect(self._on_format_changed)
        self.size_mode.currentIndexChanged.connect(self._on_size_mode)
        self.size_value.valueChanged.connect(self._on_format_changed)
        self.quality.valueChanged.connect(lambda _v: self._on_format_changed(0))
        self.dpi.valueChanged.connect(self._on_format_changed)
        self.keep_metadata.toggled.connect(self._on_format_changed)
        self.near_bright.toggled.connect(self._on_content_changed)
        self._refresh()

    # --- Estado guardado --------------------------------------------------------------------------
    def _restore(self) -> None:
        s = self._settings
        content = s.value("export/content", "effect")
        fmt = s.value("export/format", "png16")
        self.content.set_current_index(CONTENT_KEYS.index(content) if content in CONTENT_KEYS else 0)
        self.format.set_current_index(FORMAT_KEYS.index(fmt) if fmt in FORMAT_KEYS else 1)
        source = s.value("export/depth_source", "original")
        self.depth_source.set_current_index(DEPTH_SOURCE_KEYS.index(source) if source in DEPTH_SOURCE_KEYS else 0)
        self.depth_color.setCurrentIndex(max(0, self.depth_color.findData(str(s.value("export/depth_color", "")))))
        self.final_format.setCurrentIndex(max(0, self.final_format.findData(str(s.value("export/final_format", "jpg")))))
        self.quality.setValue(int(s.value("export/quality", 92)))
        self.size_mode.setCurrentIndex(max(0, self.size_mode.findData(str(s.value("export/size_mode", "original")))))
        self.dpi.setValue(int(s.value("export/dpi", 300)))
        self.size_value.setValue(int(float(s.value("export/size_value", 100))))
        self.keep_metadata.setChecked(s.value("export/keep_metadata", True, type=bool))
        self.near_bright.setChecked(s.value("export/near_bright", True, type=bool))
        self.save_settings.setChecked(s.value("export/save_settings", True, type=bool))
        folder = s.value("export/dir", "")
        self._last_dir = Path(folder) if folder else None

    def _save(self) -> None:
        s = self._settings
        s.setValue("export/content", CONTENT_KEYS[self.content.current_index()])
        s.setValue("export/format", FORMAT_KEYS[self.format.current_index()])
        s.setValue("export/depth_source", DEPTH_SOURCE_KEYS[self.depth_source.current_index()])
        s.setValue("export/depth_color", self.depth_color.currentData())
        s.setValue("export/final_format", self.final_format.currentData())
        s.setValue("export/quality", int(self.quality.value()))
        s.setValue("export/size_mode", self.size_mode.currentData())
        s.setValue("export/dpi", self.dpi.value())
        s.setValue("export/size_value", self.size_value.value())
        s.setValue("export/keep_metadata", self.keep_metadata.isChecked())
        s.setValue("export/near_bright", self.near_bright.isChecked())
        s.setValue("export/save_settings", self.save_settings.isChecked())
        if self._last_dir is not None:
            s.setValue("export/dir", str(self._last_dir))

    # --- Presets -------------------------------------------------------------------------------------------
    def current_preset(self) -> dict:
        """Los ajustes de exportación actuales tal como los guarda un preset (sin nombre ni carpeta)."""
        return {"export": {
            "content": CONTENT_KEYS[self.content.current_index()], "format": FORMAT_KEYS[self.format.current_index()],
            "final_format": str(self.final_format.currentData()), "quality": int(self.quality.value()),
            "size_mode": str(self.size_mode.currentData()), "size_value": int(self.size_value.value()),
            "dpi": int(self.dpi.value()), "keep_metadata": self.keep_metadata.isChecked(),
            "near_bright": self.near_bright.isChecked(), "depth_color": str(self.depth_color.currentData() or ""),
            "depth_source": DEPTH_SOURCE_KEYS[self.depth_source.current_index()],
            "save_settings": self.save_settings.isChecked()}}

    def apply_preset(self, data: dict) -> None:
        """Rellena los controles con un preset (lo que falte toma el valor por defecto)."""
        d = {**presets.EXPORT_DEFAULTS, **{k: v for k, v in data.get("export", {}).items() if k in self.PRESET_FIELDS}}
        self.content.set_current_index(CONTENT_KEYS.index(d["content"]) if d["content"] in CONTENT_KEYS else 0)
        self.format.set_current_index(FORMAT_KEYS.index(d["format"]) if d["format"] in FORMAT_KEYS else 1)
        self.final_format.setCurrentIndex(max(0, self.final_format.findData(str(d["final_format"]))))
        self.quality.setValue(int(d["quality"]))
        self.size_mode.blockSignals(True)           # al cambiar el modo a mano el valor se reinicia: aquí lo manda el preset
        self.size_mode.setCurrentIndex(max(0, self.size_mode.findData(str(d["size_mode"]))))
        self.size_mode.blockSignals(False)
        self.size_value.setValue(int(float(d["size_value"])))
        self.dpi.setValue(int(d["dpi"]))
        self.keep_metadata.setChecked(bool(d["keep_metadata"]))
        self.near_bright.setChecked(bool(d["near_bright"]))
        self.depth_color.setCurrentIndex(max(0, self.depth_color.findData(str(d["depth_color"] or ""))))
        self.depth_source.set_current_index(DEPTH_SOURCE_KEYS.index(d["depth_source"])
                                            if d["depth_source"] in DEPTH_SOURCE_KEYS else 0)
        self.save_settings.setChecked(bool(d["save_settings"]))
        self._on_content_changed(0)

    def content_key(self) -> str:
        """Lo que se va a exportar: «effect», «depth» o «final»."""
        return CONTENT_KEYS[self.content.current_index()]

    # --- Contenido ---------------------------------------------------------------------------------------
    def options(self) -> ex.ExportOptions:
        content = CONTENT_KEYS[self.content.current_index()]
        fmt = str(self.final_format.currentData()) if content == "final" else FORMAT_KEYS[self.format.current_index()]
        return ex.ExportOptions(
            content=content, format=fmt,
            folder=self.output_folder(), name=self.name.text().strip(), near_bright=self.near_bright.isChecked(),
            save_settings=self.save_settings.isChecked(), quality=int(self.quality.value()),
            size_mode=str(self.size_mode.currentData()), size_value=float(self.size_value.value()),
            dpi=self.dpi.value(), keep_metadata=self.keep_metadata.isChecked(),
            depth_zones=DEPTH_SOURCE_KEYS[self.depth_source.current_index()] == "zones",
            depth_palette=str(self.depth_color.currentData() or ""))

    def output_folder(self) -> Path:
        """Carpeta elegida o, si no hay, la de la foto."""
        if self._last_dir is not None:
            return self._last_dir
        ctx = self._context()
        return Path(ctx["path"]).parent if ctx else Path.home()

    def refresh_for_photo(self) -> None:
        """Se llama al cambiar de foto: actualiza el nombre propuesto, la carpeta y la resolución."""
        self._name_dirty = False
        self._refresh()

    def refresh(self) -> None:
        """Vuelve a leer qué hay para exportar (foto, mapa) y actualiza los textos y el botón."""
        self._refresh()

    def _refresh(self) -> None:
        self.preset_bar.sync()                       # el nombre del preset se ve mientras los ajustes sean los suyos
        ctx = self._context()
        content = CONTENT_KEYS[self.content.current_index()]
        fmt = FORMAT_KEYS[self.format.current_index()]
        final = content == "final"
        self.format.setVisible(not final)
        self.format_label.setVisible(not final)
        self.format_hint.setVisible(not final)
        self.final_box.setVisible(final)
        self.quality.setVisible(final and self.final_format.currentData() in ("jpg", "webp"))
        self.size_value.setVisible(self.size_mode.currentData() != "original")
        self.near_bright.setVisible(content == "depth" and not self.depth_color.currentData())   # en color no se aplica
        self.depth_color.setVisible(content == "depth")
        for widget in (self.depth_source_label, self.depth_source, self.depth_hint):
            widget.setVisible(content == "depth")
        self.depth_hint.setText(self._depth_source_hint(ctx))
        self.save_settings.setVisible(content in ("effect", "final"))
        self.format_hint.setText({
            "png8": self.tr("PNG 8 bits: ligero, para publicar."),
            "png16": self.tr("PNG 16 bits: mucha precisión de tonos, ideal para retocar después."),
            "exr32": self.tr("EXR 32 bits: coma flotante sin pérdida, para flujos profesionales."),
        }[fmt])
        if ctx is None:
            self.info.setText(self.tr("Para exportar, selecciona una foto y calcula (o carga) su mapa de profundidad."))
            self.button.setEnabled(False)
            return
        h, w = ctx["depth"].shape
        self.info.setText(self.tr("Se exporta a resolución completa: {0} × {1} px.").format(w, h))
        if final:
            out_w, out_h = ex.target_size(w, h, str(self.size_mode.currentData()), float(self.size_value.value()))
            self.size_result.setText(self.tr("Saldrá a {0} × {1} px.").format(out_w, out_h)
                                     + ("  " + self.tr("Al ampliar no se recalcula el efecto: se estira la imagen.")
                                        if out_w > w or out_h > h else ""))
        self.button.setEnabled(not self._worker.is_busy())
        if not self._name_dirty:
            self.name.setText(Path(ctx["path"]).stem + name_suffix(self.options()))
        folder = self.output_folder()
        self.folder.setText(str(folder))
        self.folder.setToolTip(str(folder))

    def _depth_source_hint(self, ctx: dict | None) -> str:
        """Aclara qué sale según el mapa elegido; avisa si «con zonas movidas» no cambiaría nada."""
        if DEPTH_SOURCE_KEYS[self.depth_source.current_index()] == "original":
            return self.tr("Se guarda el mapa original, sin cambios.")
        project = ctx.get("project") if ctx else None
        if project is None or project.depth_control.is_identity():
            return self.tr("El control de profundidad está apagado o sin cambios: el mapa saldrá igual que el "
                           "original. Actívalo en el paso 3 (Control de profundidad).")
        return self.tr("Se guarda el mapa con las zonas cerca / medio / lejos movidas.")

    def _on_content_changed(self, _i: int) -> None:
        self._save()
        self._refresh()
        self.previewChanged.emit()

    def _on_format_changed(self, _i: int = 0) -> None:
        self._save()
        self._refresh()

    def _on_size_mode(self, _i: int) -> None:
        """Al cambiar el modo de tamaño, el valor parte de algo con sentido: 100 % o el lado actual de la imagen."""
        mode = str(self.size_mode.currentData())
        ctx = self._context()
        value = 100
        if mode != "original" and mode != "percent" and ctx is not None:
            h, w = ctx["depth"].shape
            value = {"width": w, "height": h, "long_side": max(w, h)}[mode]
        self.size_value.blockSignals(True)
        self.size_value.setValue(value)
        self.size_value.blockSignals(False)
        self._on_format_changed()

    # --- Carpeta --------------------------------------------------------------------------------------------------
    def choose_folder(self) -> None:
        start = str(self.output_folder())
        folder = QFileDialog.getExistingDirectory(self, self.tr("Carpeta de salida"), start)
        if folder:
            self._last_dir = Path(folder)
            self._save()
            self._refresh()

    def review(self) -> None:
        """Abre en el visor rápido lo último que se exportó."""
        if self._last_paths:
            QuickViewer(self._last_paths, self).exec()

    def open_folder(self) -> None:
        folder = self.output_folder()
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.status.setText(self.tr("La carpeta todavía no existe: se creará al exportar."))

    # --- Exportar -----------------------------------------------------------------------------------------------------
    def export(self) -> None:
        ctx = self._context()
        if ctx is None:
            return
        options = self.options()
        try:
            options.validate()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._save()
        depth, path = ctx["depth"], Path(ctx["path"])
        self.last_photo = str(path)
        project = Project.from_dict(ctx["project"].to_dict()) if ctx.get("project") is not None else None
        mask = ctx.get("mask")
        crop = ctx.get("crop")                       # reencuadre previo de la foto (None = entera)
        aux_maps = ctx.get("maps")                   # mapas auxiliares ya calculados (para efectos limitados a un mapa)

        if options.content == "depth" or project is None:
            def job(progress):
                control = project.depth_control if project is not None else None
                return [ex.export_depth(depth, options, progress, depth_control=control)]
        else:
            def job(progress):
                return ex.export_effect(project, depth, path, options, painted_mask=mask, progress=progress,
                                        crop=crop, masks=aux_maps)

        if not self._worker.start(job):
            self.status.setText(self.tr("Ya hay una exportación en marcha."))
            return
        self.button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self._message = self.tr("Preparando…")
        self._timer.start()
        self.status.setText(self._message)

    def _on_progress(self, message: str, fraction: float) -> None:
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(fraction * 1000))
        self._timer.update(fraction)
        self._message = message
        self.status.setText(with_time(message, self._timer))

    def _end(self) -> None:
        self._timer.stop()
        self.progress.hide()
        self.button.setEnabled(self._context() is not None)

    def _on_finished(self, paths: list) -> None:
        self._end()
        self._last_paths = [str(p) for p in paths]
        self.review_button.setEnabled(any(Path(p).suffix.lower() in IMAGE_SUFFIXES for p in paths))
        first = Path(paths[0]) if paths else None
        self.status.setText(self.tr("Guardado: {0}").format(first.name) if first else self.tr("Listo"))
        self.status.setToolTip(str(first) if first else "")
        self.exported.emit(paths)

    def _on_failed(self, message: str, detail: str) -> None:
        self._end()
        self.status.setText(message)
        self.failed.emit(message, detail)
