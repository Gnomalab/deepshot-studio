"""Visor de revisión rápida: abre las imágenes que acaba de guardar la app (PNG 8 / 16 bits o EXR 32) para mirarlas con
zoom y ver sus datos (resolución, tamaño, formato y, si hay un .json al lado, el modelo y la calidad con que se hicieron).

Lo usan el Mapa rápido y el panel de Exportar. No depende del editor, del proyecto ni de la ventana principal (solo de
PySide6, numpy y Pillow), para poder llevárselo tal cual a la aplicación básica por separado.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QCoreApplication, QRectF, QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QKeyEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deepshot.ui.icons import set_icon

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".exr")          # lo que sabe abrir (el .json de al lado no es una imagen)
MAX_DISPLAY_PIXELS = 60_000_000           # por encima, se enseña una versión reducida (los datos de la ficha son los reales)


@dataclass
class FileInfo:
    """Datos de una imagen guardada, ya listos para enseñar."""

    path: Path
    width: int
    height: int
    format: str                  # «PNG 8 bits», «PNG 16 bits», «EXR 32 bits»
    size_bytes: int
    model: str = ""              # del .json de al lado, si existe
    quality: str = ""

    @property
    def size_text(self) -> str:
        n = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return ""


def sidecar_json(path: Path) -> dict:
    """Los datos del .json que se guarda junto a la imagen («x.png.json» del Mapa rápido o «x.deepshot.json» de
    Exportar). Vacío si no hay o no se puede leer."""
    for candidate in (path.with_name(path.name + ".json"), path.with_suffix(".deepshot.json")):
        try:
            if candidate.is_file():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            continue
    return {}


def _read_exr(path: Path) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        channels = f.channels()
        key = "RGB" if "RGB" in channels else next(iter(channels))
        return np.asarray(channels[key].pixels, dtype=np.float32)


def load_for_review(path: Path) -> tuple[np.ndarray, FileInfo]:
    """Lee una imagen guardada y devuelve (píxeles de 8 bits para enseñar, ficha con sus datos).

    Los PNG de 16 bits y los EXR se llevan a 8 bits solo para mirarlos; el archivo no se toca."""
    path = Path(path)
    if path.suffix.lower() == ".exr":
        data = _read_exr(path)
        shown = (np.clip(data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        fmt = QCoreApplication.translate("QuickViewer", "EXR 32 bits")
    else:
        with Image.open(path) as img:
            if img.mode in ("I;16", "I;16B", "I;16L", "I"):
                shown = (np.asarray(img, dtype=np.float32) / 257.0 + 0.5).clip(0, 255).astype(np.uint8)
                fmt = QCoreApplication.translate("QuickViewer", "PNG 16 bits")
            else:
                shown = np.asarray(img.convert("RGB" if img.mode not in ("L", "P") else "L"))
                fmt = QCoreApplication.translate("QuickViewer", "PNG 8 bits")
    h, w = shown.shape[:2]
    extra = sidecar_json(path)
    info = FileInfo(path=path, width=w, height=h, format=fmt, size_bytes=path.stat().st_size,
                    model=str(extra.get("model", "")), quality=str(extra.get("quality", "")))
    return shown, info


def to_qimage(pixels: np.ndarray) -> QImage:
    """Matriz uint8 (alto, ancho) o (alto, ancho, 3) -> QImage (copiada: no depende de la matriz)."""
    pixels = np.ascontiguousarray(pixels)
    h, w = pixels.shape[:2]
    if pixels.ndim == 2:
        return QImage(pixels.data, w, h, w, QImage.Format_Grayscale8).copy()
    return QImage(pixels.data, w, h, w * 3, QImage.Format_RGB888).copy()


class _ZoomView(QGraphicsView):
    """Vista con zoom de rueda bajo el puntero, arrastre para desplazar y doble clic para ajustar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setMinimumSize(QSize(520, 360))
        self.fitted = True

    def set_image(self, image: QImage) -> None:
        self.item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(QRectF(self.item.pixmap().rect()))
        self.fit()

    def fit(self) -> None:
        self.fitted = True
        self.resetTransform()
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def actual_size(self) -> None:
        self.fitted = False
        self.resetTransform()

    def zoom_percent(self) -> float:
        return self.transform().m11() * 100.0

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.fitted = False
        self.scale(1.25 if event.angleDelta().y() > 0 else 0.8, 1.25 if event.angleDelta().y() > 0 else 0.8)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fitted:
            self.fit()


class QuickViewer(QDialog):
    """Ventana para revisar una o varias imágenes guardadas: zoom, ficha de datos y paso de una a otra."""

    def __init__(self, files: list[str | Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Revisión rápida"))
        self.files = [Path(f) for f in files if Path(f).suffix.lower() in IMAGE_SUFFIXES and Path(f).is_file()]
        self.index = 0
        self.info: FileInfo | None = None

        self.view = _ZoomView()
        self.view.setToolTip(self.tr("Rueda: zoom · Arrastrar: mover · Doble clic: ajustar a la ventana"))
        self.title = QLabel("")
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.details = QLabel("")
        self.details.setProperty("dim", True)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setToolTip(self.tr("Datos del archivo que se está viendo"))

        self.prev_button = QPushButton("◀")
        self.prev_button.setToolTip(self.tr("Imagen anterior (flecha izquierda)"))
        self.prev_button.clicked.connect(lambda: self.step(-1))
        self.next_button = QPushButton("▶")
        self.next_button.setToolTip(self.tr("Imagen siguiente (flecha derecha)"))
        self.next_button.clicked.connect(lambda: self.step(1))
        self.fit_button = QPushButton(self.tr("Ajustar"))
        self.fit_button.setToolTip(self.tr("Ajusta la imagen a la ventana (tecla F)"))
        self.fit_button.clicked.connect(self._fit)
        self.actual_button = QPushButton("100 %")
        self.actual_button.setToolTip(self.tr("Muestra la imagen a tamaño real, un píxel por píxel (tecla 1)"))
        self.actual_button.clicked.connect(self._actual)
        self.folder_button = QPushButton(self.tr("Abrir carpeta"))
        set_icon(self.folder_button, "folder")
        self.folder_button.setToolTip(self.tr("Abre la carpeta del archivo en el explorador"))
        self.folder_button.clicked.connect(self.open_folder)
        self.close_button = QPushButton(self.tr("Cerrar"))
        self.close_button.setToolTip(self.tr("Cierra el visor (Esc)"))
        self.close_button.clicked.connect(self.accept)

        bar = QHBoxLayout()
        for b in (self.prev_button, self.next_button, self.fit_button, self.actual_button):
            bar.addWidget(b)
        bar.addStretch()
        bar.addWidget(self.folder_button)
        bar.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.details)
        layout.addLayout(bar)
        self.resize(820, 640)
        self.show_index(0)

    # --- Mostrar ---------------------------------------------------------------------------------------------
    def show_index(self, index: int) -> None:
        """Enseña la imagen número `index` (si hay)."""
        many = len(self.files) > 1
        self.prev_button.setVisible(many)
        self.next_button.setVisible(many)
        if not self.files:
            self.title.setText(self.tr("No hay imágenes que revisar."))
            self.details.setText("")
            self.folder_button.setEnabled(False)
            return
        self.index = index % len(self.files)
        path = self.files[self.index]
        try:
            pixels, self.info = load_for_review(path)
        except Exception as exc:                                    # un archivo roto no debe cerrar el visor
            log.warning("No se pudo abrir %s para revisarlo: %s", path, exc)
            self.info = None
            self.title.setText(path.name)
            self.details.setText(self.tr("No se pudo abrir este archivo."))
            return
        h, w = pixels.shape[:2]
        if w * h > MAX_DISPLAY_PIXELS:                              # solo para mirar: se reduce por pasos enteros
            k = int(np.ceil(np.sqrt(w * h / MAX_DISPLAY_PIXELS)))
            pixels = pixels[::k, ::k]
        self.view.set_image(to_qimage(pixels))
        counter = f"  ({self.index + 1}/{len(self.files)})" if many else ""
        self.title.setText(path.name + counter)
        self.details.setText(self.describe(self.info))

    def describe(self, info: FileInfo) -> str:
        parts = [self.tr("{0} × {1} px").format(info.width, info.height), info.format, info.size_text]
        if info.model:
            parts.append(self.tr("Modelo: {0}").format(info.model))
        if info.quality:
            parts.append(self.tr("Calidad: {0}").format(info.quality))
        return "  ·  ".join(parts)

    def step(self, delta: int) -> None:
        if self.files:
            self.show_index(self.index + delta)

    def _fit(self) -> None:
        self.view.fit()

    def _actual(self) -> None:
        self.view.actual_size()

    def open_folder(self) -> None:
        if self.files:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.files[self.index].parent)))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_Left:
            self.step(-1)
        elif key == Qt.Key_Right:
            self.step(1)
        elif key == Qt.Key_F:
            self._fit()
        elif key == Qt.Key_1:
            self._actual()
        else:
            super().keyPressEvent(event)
