"""Diálogo de reencuadre: la foto entera con un rectángulo que se mueve y se redimensiona (el archivo no se toca)."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from deepshot import framing
from deepshot.ui import theme

HANDLE = 6                       # medio lado de los tiradores, en píxeles de pantalla
GRAB = 11                        # a qué distancia de un tirador o borde se agarra


class CropView(QWidget):
    """Pinta la foto ajustada al widget, oscurece lo que queda fuera del rectángulo y lo deja editar con el ratón.

    `crop` (framing.Crop o None) es el rectángulo normalizado. `aspect` (ancho/alto en píxeles de la foto, o None) lo
    mantiene con esa proporción al redimensionar."""

    changed = Signal()

    def __init__(self, image: QImage, crop: framing.Crop | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = image
        self.crop: framing.Crop = crop or (0.0, 0.0, 1.0, 1.0)
        self.aspect: float | None = None
        self._drag: tuple[str, framing.Crop, QPointF] | None = None
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self.setToolTip(QCoreApplication.translate("Framing", "Arrastra el rectángulo para moverlo y sus bordes o esquinas "
                                                            "para cambiar el encuadre. Arrastra fuera para dibujar uno nuevo."))

    # --- Geometría -----------------------------------------------------------------
    def _image_rect(self) -> QRectF:
        w, h = self._image.width(), self._image.height()
        avail = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        scale = min(avail.width() / w, avail.height() / h)
        size = (w * scale, h * scale)
        return QRectF(avail.center().x() - size[0] / 2, avail.center().y() - size[1] / 2, *size)

    def _crop_rect(self) -> QRectF:
        r = self._image_rect()
        x0, y0, x1, y1 = self.crop
        return QRectF(r.left() + x0 * r.width(), r.top() + y0 * r.height(), (x1 - x0) * r.width(), (y1 - y0) * r.height())

    def _to_norm(self, p: QPointF) -> tuple[float, float]:
        r = self._image_rect()
        return (min(max((p.x() - r.left()) / r.width(), 0.0), 1.0), min(max((p.y() - r.top()) / r.height(), 0.0), 1.0))

    def _hit(self, p: QPointF) -> str:
        """Qué agarra el ratón: una esquina o un borde (n, s, e, w, ne…), el interior («move») o nada («new»)."""
        r = self._crop_rect()
        near_l, near_r = abs(p.x() - r.left()) <= GRAB, abs(p.x() - r.right()) <= GRAB
        near_t, near_b = abs(p.y() - r.top()) <= GRAB, abs(p.y() - r.bottom()) <= GRAB
        inside_x, inside_y = r.left() - GRAB <= p.x() <= r.right() + GRAB, r.top() - GRAB <= p.y() <= r.bottom() + GRAB
        v = "n" if near_t and inside_x else "s" if near_b and inside_x else ""
        h = "w" if near_l and inside_y else "e" if near_r and inside_y else ""
        if v or h:
            return v + h
        return "move" if r.contains(p) else "new"

    # --- Pintado ---------------------------------------------------------------------
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), theme.color("base"))
        img_rect = self._image_rect()
        painter.drawImage(img_rect, self._image)
        crop_rect = self._crop_rect()
        shade = QColor(0, 0, 0, 150)
        for part in (QRectF(img_rect.left(), img_rect.top(), img_rect.width(), crop_rect.top() - img_rect.top()),
                     QRectF(img_rect.left(), crop_rect.bottom(), img_rect.width(), img_rect.bottom() - crop_rect.bottom()),
                     QRectF(img_rect.left(), crop_rect.top(), crop_rect.left() - img_rect.left(), crop_rect.height()),
                     QRectF(crop_rect.right(), crop_rect.top(), img_rect.right() - crop_rect.right(), crop_rect.height())):
            painter.fillRect(part, shade)
        accent = theme.color("accent")
        painter.setPen(QPen(accent, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(crop_rect)
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1))                    # regla de tercios
        for i in (1, 2):
            x = crop_rect.left() + crop_rect.width() * i / 3
            y = crop_rect.top() + crop_rect.height() * i / 3
            painter.drawLine(QPointF(x, crop_rect.top()), QPointF(x, crop_rect.bottom()))
            painter.drawLine(QPointF(crop_rect.left(), y), QPointF(crop_rect.right(), y))
        painter.setPen(QPen(theme.color("base"), 1))
        painter.setBrush(accent)
        cx, cy = crop_rect.center().x(), crop_rect.center().y()
        for x in (crop_rect.left(), cx, crop_rect.right()):
            for y in (crop_rect.top(), cy, crop_rect.bottom()):
                if (x, y) != (cx, cy):
                    painter.drawRect(QRectF(x - HANDLE / 2, y - HANDLE / 2, HANDLE, HANDLE))

    # --- Ratón -----------------------------------------------------------------------
    _CURSORS = {"n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor, "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
                "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor, "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
                "move": Qt.SizeAllCursor, "new": Qt.CrossCursor}

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        mode = self._hit(pos)
        if mode == "new":
            x, y = self._to_norm(pos)
            self.crop = (x, y, x, y)
            mode = "se"
        self._drag = (mode, self.crop, pos)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._drag is None:
            self.setCursor(self._CURSORS.get(self._hit(pos), Qt.ArrowCursor))
            return
        mode, start, origin = self._drag
        r = self._image_rect()
        dx, dy = (pos.x() - origin.x()) / r.width(), (pos.y() - origin.y()) / r.height()
        x0, y0, x1, y1 = start
        if mode == "move":
            w, h = x1 - x0, y1 - y0
            nx, ny = min(max(x0 + dx, 0.0), 1.0 - w), min(max(y0 + dy, 0.0), 1.0 - h)
            self.crop = (nx, ny, nx + w, ny + h)
        else:
            if "w" in mode:
                x0 = x0 + dx
            if "e" in mode:
                x1 = x1 + dx
            if "n" in mode:
                y0 = y0 + dy
            if "s" in mode:
                y1 = y1 + dy
            if mode == "se" and start[0] == start[2]:                      # rectángulo nuevo: parte del punto donde se pulsó
                x0, y0 = start[0], start[1]
            self.crop = self._constrain(mode, (x0, y0, x1, y1))
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        if self._drag is not None:
            self._drag = None
            self.crop = framing.normalize(self.crop) or (0.0, 0.0, 1.0, 1.0)
            self.changed.emit()
            self.update()

    def _constrain(self, mode: str, box: framing.Crop) -> framing.Crop:
        """Ordena, mantiene dentro de la foto y, si hay proporción fija, la respeta moviendo el lado libre."""
        x0, y0, x1, y1 = box
        x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
        y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
        if self.aspect:
            w_px, h_px = self._image.width(), self._image.height()
            width, height = (x1 - x0) * w_px, (y1 - y0) * h_px
            if "e" in mode or "w" in mode:                                  # manda el ancho
                height = width / self.aspect
            else:
                width = height * self.aspect
            nw, nh = width / w_px, height / h_px
            if "w" in mode:
                x0 = x1 - nw
            else:
                x1 = x0 + nw
            if "n" in mode:
                y0 = y1 - nh
            elif "s" in mode or mode in ("e", "w"):
                y1 = y0 + nh
            if mode in ("e", "w"):                                          # el borde libre crece desde el centro
                mid = (box[1] + box[3]) / 2
                y0, y1 = mid - nh / 2, mid + nh / 2
            if mode in ("n", "s"):
                mid = (box[0] + box[2]) / 2
                x0, x1 = mid - nw / 2, mid + nw / 2
            x0, x1, y0, y1 = max(x0, 0.0), min(x1, 1.0), max(y0, 0.0), min(y1, 1.0)
            if self.aspect and (x1 - x0) * w_px / max((y1 - y0) * h_px, 1e-6) > self.aspect * 1.001:
                x1 = x0 + (y1 - y0) * h_px * self.aspect / w_px            # tocó el borde de la foto: se reajusta
            elif self.aspect and (x1 - x0) * w_px / max((y1 - y0) * h_px, 1e-6) < self.aspect / 1.001:
                y1 = y0 + (x1 - x0) * w_px / self.aspect / h_px
        return (x0, y0, x1, y1)


class CropDialog(QDialog):
    """Reencuadre previo: elige una proporción o dibuja el rectángulo a mano. El resultado es `crop()` (None = foto entera)."""

    def __init__(self, image: QImage, crop: framing.Crop | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(QCoreApplication.translate("Framing", "Reencuadre previo"))
        self._image = image
        self.view = CropView(image, crop)
        self.view.changed.connect(self._update_label)

        self.aspect_combo = QComboBox()
        for key, label, _ratio in framing.ASPECTS:
            self.aspect_combo.addItem(QCoreApplication.translate("Framing", label), key)
        self.aspect_combo.setToolTip(QCoreApplication.translate("Framing", "Proporción del encuadre. «Libre» deja cualquier forma."))
        self.aspect_combo.currentIndexChanged.connect(self._on_aspect)
        reset = QPushButton(QCoreApplication.translate("Framing", "Foto entera"))
        reset.setToolTip(QCoreApplication.translate("Framing", "Quita el encuadre: se trabaja con toda la foto"))
        reset.clicked.connect(self._reset)
        self.label = QLabel()
        self.label.setProperty("dim", True)
        hint = QLabel(QCoreApplication.translate("Framing", "El archivo original no se modifica: la profundidad y el efecto se calculan solo con esta zona."))
        hint.setProperty("dim", True)
        hint.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(QLabel(QCoreApplication.translate("Framing", "Proporción")))
        top.addWidget(self.aspect_combo)
        top.addWidget(reset)
        top.addStretch()
        top.addWidget(self.label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.view, 1)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self.resize(780, 620)
        self._update_label()

    def crop(self) -> framing.Crop | None:
        return framing.normalize(self.view.crop)

    def _size(self) -> tuple[int, int]:
        return (self._image.width(), self._image.height())

    def _on_aspect(self) -> None:
        key = self.aspect_combo.currentData()
        ratio = dict((k, r) for k, _l, r in framing.ASPECTS).get(key)
        if key == "original":
            ratio = self._size()[0] / self._size()[1]
        self.view.aspect = ratio
        if ratio:
            self.view.crop = framing.with_aspect(self.crop(), ratio, self._size())
        self.view.update()
        self._update_label()

    def _reset(self) -> None:
        self.view.crop = (0.0, 0.0, 1.0, 1.0)
        self.aspect_combo.setCurrentIndex(0)
        self.view.update()
        self._update_label()

    def _update_label(self) -> None:
        w, h = framing.cropped_size(framing.normalize(self.view.crop), self._size())
        self.label.setText("{0} × {1} px".format(w, h))
