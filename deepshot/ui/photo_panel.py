"""Panel de la primera página: datos de la fotografía (EXIF y archivo) y su reencuadre previo."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from deepshot import framing, photo_info


class PhotoInfoPanel(QWidget):
    """Una tabla de datos de la foto elegida y, debajo, el encuadre con sus botones.

    No importa nada ni guarda estado: la ventana principal le pasa la foto (`show_photo`) y el encuadre (`set_crop`) y
    escucha `reframeRequested` (abrir el diálogo) y `cropCleared`."""

    reframeRequested = Signal()
    cropCleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info = photo_info.PhotoInfo()
        self.empty = QLabel(self.tr("Ninguna fotografía seleccionada."))
        self.empty.setWordWrap(True)

        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(3)
        self._fields: dict[str, QLabel] = {}
        for key, title in (("name", self.tr("Archivo")), ("size", self.tr("Tamaño")), ("dims", self.tr("Dimensiones")),
                           ("taken", self.tr("Fecha")), ("camera", self.tr("Cámara")), ("lens", self.tr("Objetivo")),
                           ("exposure", self.tr("Exposición")), ("color", self.tr("Color")), ("software", self.tr("Software")),
                           ("gps", self.tr("Ubicación"))):
            value = QLabel()
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)               # se puede copiar el dato
            label = QLabel(title)
            label.setProperty("dim", True)
            self._fields[key] = value
            self.form.addRow(label, value)
        self._labels = {key: self.form.labelForField(widget) for key, widget in self._fields.items()}
        self.show_gps = QCheckBox(self.tr("Mostrar la ubicación"))
        self.show_gps.setToolTip(self.tr("La ubicación GPS de la foto está oculta por defecto: es un dato sensible si "
                                         "compartes capturas de la ventana."))
        self.show_gps.toggled.connect(lambda _on: self._fill())

        self.reframe_button = QPushButton(self.tr("Reencuadrar…"))
        self.reframe_button.setToolTip(self.tr("Elige la zona de la foto con la que trabajar. El archivo original no se "
                                               "toca; la profundidad se calcula solo sobre esa zona."))
        self.reframe_button.clicked.connect(self.reframeRequested)
        self.clear_button = QPushButton(self.tr("Foto entera"))
        self.clear_button.setToolTip(self.tr("Quita el encuadre: se trabaja con toda la foto"))
        self.clear_button.clicked.connect(self.cropCleared)
        self.crop_label = QLabel()
        self.crop_label.setWordWrap(True)
        self.crop_label.setProperty("dim", True)
        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        buttons.addWidget(self.reframe_button, 1)
        buttons.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.empty)
        layout.addLayout(self.form)
        layout.addWidget(self.show_gps)
        layout.addLayout(buttons)
        layout.addWidget(self.crop_label)
        self.show_photo(None)

    # --- API -------------------------------------------------------------------------
    def show_photo(self, path: Path | None) -> None:
        self._info = photo_info.read(path) if path is not None else photo_info.PhotoInfo()
        self.empty.setVisible(path is None)
        self.reframe_button.setEnabled(path is not None)
        self._fill()
        if path is None:
            self.set_crop(None, None)

    def set_crop(self, crop: framing.Crop | None, full_size: tuple[int, int] | None) -> None:
        """Muestra el encuadre actual (None = foto entera) y a cuántos píxeles queda."""
        self.clear_button.setEnabled(crop is not None)
        if crop is None or full_size is None:
            self.crop_label.setText(self.tr("Se trabaja con la foto entera.") if full_size else "")
            return
        w, h = framing.cropped_size(crop, full_size)
        self.crop_label.setText(self.tr("Encuadre: {0} × {1} px ({2:.0f} % de la foto).").format(
            w, h, 100.0 * (w * h) / max(full_size[0] * full_size[1], 1)))

    # --- Contenido -------------------------------------------------------------------
    def _fill(self) -> None:
        i = self._info
        exposure = "  ·  ".join(x for x in (i.aperture, i.shutter, i.iso, i.focal_mm) if x)
        values = {
            "name": i.name,
            "size": "{0}  ·  {1}".format(photo_info.human_size(i.file_bytes), i.format) if i.file_bytes else "",
            "dims": "{0} × {1} px  ·  {2} MP".format(i.width, i.height, i.megapixels) if i.width else "",
            "taken": i.taken, "camera": i.camera, "lens": i.lens, "exposure": exposure, "color": i.color_profile,
            "software": i.software,
            "gps": (i.gps if self.show_gps.isChecked() else self.tr("(oculta)")) if i.gps else "",
        }
        for key, text in values.items():
            self._fields[key].setText(text)
            self._fields[key].setVisible(bool(text))
            self._labels[key].setVisible(bool(text))
        self.show_gps.setVisible(bool(i.gps))
