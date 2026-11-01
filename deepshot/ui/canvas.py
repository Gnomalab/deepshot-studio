"""Lienzo central: imagen con zoom y desplazamiento, y modo comparativo A/B con divisor."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deepshot.ui import theme
from deepshot.area_edit import axes  # noqa: E402  (módulo sin interfaz)


def depth_to_qimage(depth: np.ndarray) -> QImage:
    """Mapa de profundidad (0 = cerca) -> imagen gris para ver (cerca = claro)."""
    gray = np.ascontiguousarray(np.round((1.0 - depth) * 255.0).astype(np.uint8))
    h, w = gray.shape
    # .copy(): QImage no copia el buffer de numpy; así no depende de que `gray` siga vivo.
    return QImage(gray.data, w, h, w, QImage.Format_Grayscale8).copy()


class CanvasView(QGraphicsView):
    """Vista con zoom (rueda del ratón) y desplazamiento (arrastre).

    Dos modos:
      - Imagen única: `set_image()`.
      - Comparar: `set_compare(a, b)`. A se ve a un lado del divisor y B al otro.
        El divisor se arrastra con el ratón; puede ser vertical (A izquierda | B derecha)
        u horizontal (A arriba / B abajo), con `set_split_orientation()`.
    Mientras no hay imagen, muestra un estado vacío con un botón para importar.

    Edición de áreas (cuando hay contornos que mostrar):
      - clic y arrastre dentro de un área: la mueve (y la selecciona); tiradores del área
        seleccionada: cuadrados = escalar un semieje, círculo superior = girar;
      - herramientas "draw" (dibujar arrastrando) y "place" (colocar siguiendo al puntero);
      - modo Blender con el ratón sobre el lienzo: G mover, S escalar, R girar; X / Y / Z fijan el eje;
        clic o Intro aceptan, Esc o clic derecho cancelan;
      - rueda con Ctrl: tamaño · con Mayús: profundidad · con Alt: suavidad del borde;
      - botón central: desplazar la vista.
    Todo se comunica con señales; el lienzo no modifica ningún dato.
    """

    import_requested = Signal()
    # Gesto de edición de un área. Diccionario con: kind ("move", "scale_x", "scale_y", "rotate",
    # "scale_all", "rotate_free", "pan", "tilt", "draw", "place"), phase ("press", "move", "axis", "release", "cancel"),
    # index (área o -1), x, y (píxeles de la foto), mods (modificadores de teclado) y axis ("", "x", "y", "z").
    gesture = Signal(dict)
    areaWheel = Signal(str, float)   # "scale" | "depth" | "softness", pasos de rueda
    hint = Signal(str)               # ayuda para la barra de estado ("" = quitar)
    ZOOM_STEP = 1.15
    GRAB_DISTANCE = 10  # píxeles de pantalla alrededor del divisor que permiten arrastrarlo
    HANDLE_SIZE = 8     # lado de los tiradores cuadrados, en píxeles de pantalla
    HANDLE_TOLERANCE = 9
    ROTATE_ARM = 30     # distancia del tirador de giro al borde superior del área, en píxeles

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Canvas")
        self.setFrameShape(QFrame.NoFrame)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # Redibujo completo: el divisor se pinta encima y no debe dejar rastro al desplazar.
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMouseTracking(True)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: QGraphicsPixmapItem | None = None     # imagen A (o la única)
        self._clip: QGraphicsRectItem | None = None       # recorta B al lado del divisor
        self._item_b: QGraphicsPixmapItem | None = None   # imagen B (solo al comparar)
        self._mask_item: QGraphicsPixmapItem | None = None  # tinte rojo de la máscara pintada
        self._focus_item: QGraphicsPixmapItem | None = None  # tinte verde de la zona en foco (desenfoque de lente)
        self._labels = ("", "")
        self._badge = ""                  # aviso fijo en la esquina superior izquierda (p. ej. «Pre FX»)
        self._split = 0.5                                 # posición del divisor, 0..1
        self._orientation = Qt.Horizontal                 # Horizontal: divisor vertical (izq | der)
        self._dragging_split = False
        self._overlays: list[dict] = []   # áreas dibujadas encima (coordenadas de la escena)
        self._tool = ""                   # "" (editar), "draw" o "place"
        self._brush: dict | None = None   # vista previa del tamaño al usar una herramienta
        self._cursor_pos = None           # última posición del ratón en el visor (QPoint)
        self._active: dict | None = None  # gesto en curso: {"kind", "index", "modal"}
        self._axis = ""                   # eje fijado en un gesto con teclado (G / S / R)
        self._panning = False
        self._pan_last = None
        self._fit_mode = True  # True: la imagen se ajusta a la ventana al redimensionar
        self.setFocusPolicy(Qt.StrongFocus)  # para G / S / R y los ejes X / Y / Z

        self._empty = self._build_empty_state()

    # --- Estado vacío -------------------------------------------------------
    def _build_empty_state(self) -> QWidget:
        box = QWidget(self.viewport())
        box.setObjectName("EmptyState")
        layout = QVBoxLayout(box)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        title = QLabel(self.tr("Empieza con una fotografía"))
        title.setObjectName("EmptyTitle")
        title.setAlignment(Qt.AlignCenter)
        hint = QLabel(self.tr("Arrastra una imagen aquí o impórtala desde Archivo."))
        hint.setObjectName("EmptyHint")
        hint.setAlignment(Qt.AlignCenter)
        button = QPushButton(self.tr("Importar fotografía…"))
        button.setObjectName("Primary")
        button.clicked.connect(self.import_requested)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addSpacing(6)
        layout.addWidget(button, alignment=Qt.AlignCenter)
        return box

    # --- Imagen única ---------------------------------------------------------
    def set_image(self, image: QImage | None, keep_view: bool = False, scene_size: QSize | None = None) -> None:
        """Muestra una imagen (o vuelve al estado vacío con None).

        keep_view=True conserva el zoom y la posición si la imagen tiene el mismo tamaño
        (útil al alternar entre la foto y su mapa de profundidad).
        scene_size: tamaño con el que se muestra (la vista previa, más pequeña, se amplía
        al tamaño de la foto para que zoom, áreas y comparación coincidan).
        """
        if image is None or image.isNull():
            self._reset_scene()
            self._empty.show()
            return
        self._empty.hide()
        self._set_base(image, keep_view, scene_size)
        self._remove_b()
        self.viewport().update()

    # --- Comparar ---------------------------------------------------------------
    def set_compare(self, image_a: QImage, image_b: QImage, labels: tuple[str, str] = ("", ""),
                    keep_view: bool = False) -> None:
        """Muestra A y B a la vez, separadas por el divisor. B se muestra al tamaño de A."""
        if image_a is None or image_a.isNull() or image_b is None or image_b.isNull():
            self.set_image(image_a, keep_view)
            return
        self._empty.hide()
        self._set_base(image_a, keep_view)
        pix_b = QPixmap.fromImage(image_b)
        if self._item_b is None:
            # El rectángulo recorta a su hijo (B): solo se ve B dentro de él.
            self._clip = QGraphicsRectItem()
            self._clip.setPen(QPen(Qt.NoPen))
            self._clip.setFlag(QGraphicsItem.ItemClipsChildrenToShape, True)
            self._scene.addItem(self._clip)
            self._item_b = QGraphicsPixmapItem(pix_b, self._clip)
            self._item_b.setTransformationMode(Qt.SmoothTransformation)
        else:
            self._item_b.setPixmap(pix_b)
        # Escala de B para cubrir exactamente A (sin reescalar los píxeles: más rápido).
        self._item_b.setScale(image_a.width() / max(image_b.width(), 1))
        self._labels = labels
        self._update_clip()

    def set_badge(self, text: str) -> None:
        """Aviso en la esquina superior izquierda del visor («Pre FX», «Post FX»…); vacío = ninguno."""
        if text != self._badge:
            self._badge = text
            self.viewport().update()

    def is_comparing(self) -> bool:
        return self._item_b is not None

    def set_split_orientation(self, orientation: Qt.Orientation) -> None:
        """Qt.Horizontal: A a la izquierda, B a la derecha. Qt.Vertical: A arriba, B abajo."""
        self._orientation = orientation
        self._update_clip()

    def split_orientation(self) -> Qt.Orientation:
        return self._orientation

    def set_split(self, fraction: float) -> None:
        self._split = min(max(fraction, 0.0), 1.0)
        self._update_clip()

    def split(self) -> float:
        return self._split

    # --- Áreas dibujadas encima y herramientas ---------------------------------------
    def set_overlays(self, overlays: list[dict]) -> None:
        """Áreas a dibujar: dicts con cx, cy, rx, ry (píxeles de la foto), angle (grados), box, selected,
        strength, soft (suavidad del borde 0..1), color (r, g, b en 0..1), name e index."""
        self._overlays = overlays
        # Si ya no hay áreas se cancela el gesto sobre ellas; los de herramientas (pincel, cuentagotas) siguen.
        if not overlays and self._active is not None and self._active["kind"] not in ("pick_focus", "paint", "erase"):
            self._end_gesture(cancel=True)
        self.viewport().update()

    def set_tool(self, tool: str) -> None:
        """"" = editar con el ratón · "draw" = dibujar un área arrastrando · "place" = colocar el área ·
        "paint" / "erase" = pincel de la máscara · "pick_focus" = cuentagotas del foco."""
        self._tool = tool
        self.setDragMode(QGraphicsView.NoDrag if tool else QGraphicsView.ScrollHandDrag)
        self._update_cursor()
        self.viewport().update()

    def tool(self) -> str:
        return self._tool

    def set_mask_overlay(self, mask: np.ndarray | None, scene_size: tuple[int, int] | None = None) -> None:
        """Tiñe de rojo, encima de la imagen, lo que está pintado en la máscara (uint8, 2D).

        scene_size: tamaño (ancho, alto) de la foto: la capa, de menor resolución, se estira a él.
        None quita la capa.
        """
        self._set_tint("_mask_item", 5, mask, scene_size, (255, 60, 60))

    def set_focus_overlay(self, sharpness: np.ndarray | None, scene_size: tuple[int, int] | None = None) -> None:
        """Tiñe de verde, encima de la imagen, la zona que queda en foco con el desenfoque de lente.

        sharpness: (alto, ancho) uint8; 255 = en foco, 0 = totalmente desenfocado. None quita la capa."""
        self._set_tint("_focus_item", 4, sharpness, scene_size, (40, 220, 90))

    def _set_tint(self, attr: str, z: float, mask: np.ndarray | None, scene_size: tuple[int, int] | None,
                  rgb: tuple[int, int, int]) -> None:
        """Capa de color translúcida sobre la imagen: más valor en `mask` = más opaca. None la quita."""
        item = getattr(self, attr)
        if mask is None or self._item is None or scene_size is None:
            if item is not None:
                self._scene.removeItem(item)
                setattr(self, attr, None)
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
        rgba[..., 3] = (mask.astype(np.float32) * 0.55).astype(np.uint8)
        image = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(image)
        if item is None:
            item = self._scene.addPixmap(pix)
            item.setZValue(z)
            item.setTransformationMode(Qt.SmoothTransformation)
            item.setAcceptedMouseButtons(Qt.NoButton)
            setattr(self, attr, item)
        else:
            item.setPixmap(pix)
        item.setScale(scene_size[0] / w)

    def set_brush(self, brush: dict | None) -> None:
        """Vista previa del tamaño al usar una herramienta: dict con rx, ry (px de la foto), angle,
        soft, box y color. Se dibuja junto al puntero, como el cursor de un pincel."""
        self._brush = brush
        self.viewport().update()

    def has_selected_overlay(self) -> bool:
        return any(ov.get("selected") for ov in self._overlays)

    # --- Internos ----------------------------------------------------------------
    def _reset_scene(self) -> None:
        self._scene.clear()  # borra también los objetos hijos
        self._item = self._clip = self._item_b = self._mask_item = self._focus_item = None

    def _set_base(self, image: QImage, keep_view: bool, scene_size: QSize | None = None) -> None:
        """Pone la imagen A; conserva el encuadre si se pide y el tamaño de escena no cambia."""
        target = scene_size or image.size()
        scale = target.width() / max(image.width(), 1)
        same_size = self._item is not None and self._scene.sceneRect().size().toSize() == target
        if keep_view and same_size:
            self._item.setPixmap(QPixmap.fromImage(image))
            self._item.setScale(scale)
            return
        self._reset_scene()
        self._item = self._scene.addPixmap(QPixmap.fromImage(image))
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._item.setScale(scale)
        self._scene.setSceneRect(QRectF(0, 0, target.width(), target.height()))
        self.fit_to_window()

    def _remove_b(self) -> None:
        if self._clip is not None:
            self._scene.removeItem(self._clip)  # se lleva consigo a B (su hijo)
        self._clip = self._item_b = None

    def _image_rect(self) -> QRectF:
        return self._scene.sceneRect() if self._item is not None else QRectF()

    def _update_clip(self) -> None:
        """Ajusta la zona visible de B al lado del divisor que le corresponde."""
        if self._clip is not None:
            r = self._image_rect()
            if self._orientation == Qt.Horizontal:
                x = r.left() + r.width() * self._split
                self._clip.setRect(QRectF(x, r.top(), r.right() - x, r.height()))
            else:
                y = r.top() + r.height() * self._split
                self._clip.setRect(QRectF(r.left(), y, r.width(), r.bottom() - y))
        self.viewport().update()

    def _divider_viewport(self) -> tuple[QPointF, QPointF]:
        """Extremos del divisor en coordenadas de pantalla."""
        r = self._image_rect()
        if self._orientation == Qt.Horizontal:
            x = r.left() + r.width() * self._split
            a, b = QPointF(x, r.top()), QPointF(x, r.bottom())
        else:
            y = r.top() + r.height() * self._split
            a, b = QPointF(r.left(), y), QPointF(r.right(), y)
        return QPointF(self.mapFromScene(a)), QPointF(self.mapFromScene(b))

    def _near_divider(self, pos) -> bool:
        if not self.is_comparing():
            return False
        a, b = self._divider_viewport()
        if self._orientation == Qt.Horizontal:
            return abs(pos.x() - a.x()) <= self.GRAB_DISTANCE and a.y() - 4 <= pos.y() <= b.y() + 4
        return abs(pos.y() - a.y()) <= self.GRAB_DISTANCE and a.x() - 4 <= pos.x() <= b.x() + 4

    def _split_from_pos(self, pos) -> None:
        p = self.mapToScene(pos)
        r = self._image_rect()
        if self._orientation == Qt.Horizontal:
            self.set_split((p.x() - r.left()) / max(r.width(), 1.0))
        else:
            self.set_split((p.y() - r.top()) / max(r.height(), 1.0))

    # --- Dibujo del divisor, tirador y etiquetas ------------------------------------
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        self._draw_overlays(painter)
        if self._badge:
            painter.save()
            painter.resetTransform()
            font = QFont(painter.font())
            font.setPointSizeF(9.5)
            font.setBold(True)
            painter.setFont(font)
            self._draw_label(painter, self._badge, QPointF(self.viewport().rect().left() + 12, 12), Qt.AlignLeft,
                             theme.color("accent"))
            painter.restore()
        if not self.is_comparing():
            return
        a, b = self._divider_viewport()
        painter.save()
        painter.resetTransform()  # a partir de aquí, coordenadas de pantalla
        painter.setRenderHint(QPainter.Antialiasing)

        # Línea: oscura debajo y clara encima, para verse sobre cualquier imagen.
        painter.setPen(QPen(QColor(0, 0, 0, 140), 3))
        painter.drawLine(a, b)
        painter.setPen(QPen(QColor("#e5e5e5"), 1.5))
        painter.drawLine(a, b)

        # Tirador circular en el centro de la parte visible del divisor.
        visible = QRectF(self.viewport().rect())
        center = (a + b) / 2
        if self._orientation == Qt.Horizontal:
            center.setY(min(max(visible.center().y(), a.y() + 16), b.y() - 16))
        else:
            center.setX(min(max(visible.center().x(), a.x() + 16), b.x() - 16))
        painter.setPen(QPen(QColor(0, 0, 0, 120), 1))
        painter.setBrush(QBrush(theme.color("accent")))
        painter.drawEllipse(center, 12, 12)
        painter.setPen(QPen(theme.color("on_accent"), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self._orientation == Qt.Horizontal:   # flechas ‹ ›
            for d in (-1, 1):
                painter.drawLine(center + QPointF(d * 3, -4), center + QPointF(d * 7, 0))
                painter.drawLine(center + QPointF(d * 7, 0), center + QPointF(d * 3, 4))
        else:                                    # flechas ˄ ˅
            for d in (-1, 1):
                painter.drawLine(center + QPointF(-4, d * 3), center + QPointF(0, d * 7))
                painter.drawLine(center + QPointF(0, d * 7), center + QPointF(4, d * 3))

        # Etiquetas de cada lado, en las esquinas de la parte visible de la imagen.
        img = QRectF(self.mapFromScene(self._image_rect()).boundingRect()).intersected(visible)
        font = QFont(painter.font())
        font.setPointSizeF(8.5)
        painter.setFont(font)
        label_a, label_b = self._labels
        if self._orientation == Qt.Horizontal:
            self._draw_label(painter, label_a, QPointF(img.left() + 8, img.top() + 8), Qt.AlignLeft)
            self._draw_label(painter, label_b, QPointF(img.right() - 8, img.top() + 8), Qt.AlignRight)
        else:
            self._draw_label(painter, label_a, QPointF(img.left() + 8, img.top() + 8), Qt.AlignLeft)
            self._draw_label(painter, label_b, QPointF(img.left() + 8, img.bottom() - 30), Qt.AlignLeft)
        painter.restore()

    def _draw_overlays(self, painter: QPainter) -> None:
        """Contorno de cada área (elipse o caja), anillo de suavidad, nombre y, en la seleccionada,
        tiradores. Se dibuja en coordenadas de pantalla: el grosor y los tiradores no dependen del zoom."""
        if self._item is None:
            return
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.Antialiasing)
        to_view = self.viewportTransform()
        ordered = [ov for ov in self._overlays if not ov.get("selected")] + \
                  [ov for ov in self._overlays if ov.get("selected")]
        if self._tool not in ("paint", "erase"):      # pintando la máscara, los contornos y tiradores de las áreas estorban
            for ov in ordered:
                self._draw_area(painter, ov, to_view)
        self._draw_brush(painter, to_view)
        painter.restore()

    @staticmethod
    def _area_path(ov: dict, scale: float, to_view: QTransform) -> QPainterPath:
        """Forma del área (escalada `scale` respecto al borde) en coordenadas de pantalla."""
        if ov.get("poly"):                      # área inclinada: su contorno real, escalado hacia el centro
            cx, cy = ov["cx"], ov["cy"]
            path = QPainterPath()
            pts = [QPointF(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in ov["poly"]]
            path.addPolygon(QPolygonF(pts + [pts[0]]))
            return to_view.map(path)
        local = QTransform()
        local.translate(ov["cx"], ov["cy"])
        local.rotate(-ov.get("angle", 0.0))
        rect = QRectF(-ov["rx"] * scale, -ov["ry"] * scale, 2 * ov["rx"] * scale, 2 * ov["ry"] * scale)
        path = QPainterPath()
        path.addRect(rect) if ov.get("box") else path.addEllipse(rect)
        return (local * to_view).map(path)

    def _draw_area(self, painter: QPainter, ov: dict, to_view: QTransform) -> None:
        color = QColor.fromRgbF(*ov.get("color", (1.0, 0.81, 0.35)))
        selected = bool(ov.get("selected"))
        outer = self._area_path(ov, 1.0, to_view)
        under = QPen(QColor(0, 0, 0, 150), 3.6 if selected else 3.0)
        line = QPen(color, 2.2 if selected else 1.5, Qt.SolidLine if selected else Qt.DashLine)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 22 if selected else 0))
        painter.setPen(under)
        painter.drawPath(outer)
        painter.setPen(line)
        painter.drawPath(outer)
        # Anillo interior: hasta ahí la influencia es plena; entre él y el borde se desvanece.
        soft = float(ov.get("soft", 0.0))
        if soft > 0.03:
            inner = self._area_path(ov, max(1.0 - soft, 0.02), to_view)
            painter.setBrush(Qt.NoBrush)
            ring = QPen(color, 1.0, Qt.DotLine)
            painter.setPen(QPen(QColor(0, 0, 0, 110), 2.2))
            painter.drawPath(inner)
            painter.setPen(ring)
            painter.drawPath(inner)

        centre = self.mapFromScene(QPointF(ov["cx"], ov["cy"]))
        centre = QPointF(centre)
        painter.setBrush(Qt.NoBrush)
        for width, c in ((3.4, QColor(0, 0, 0, 150)), (1.6, color)):
            painter.setPen(QPen(c, width, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(centre + QPointF(-6, 0), centre + QPointF(6, 0))
            painter.drawLine(centre + QPointF(0, -6), centre + QPointF(0, 6))

        handles = self._handles(ov)
        top = handles["scale_y"]
        label = ov.get("name", "")
        if label:       # el nombre va bajo el área: arriba están los tiradores de escala y de giro
            bottom = centre * 2 - top
            self._draw_label(painter, f"{label}  {ov.get('strength', 0):+.2f}", QPointF(bottom.x(), bottom.y() + 16),
                             Qt.AlignHCenter, color)
        if not selected:
            return
        # Tiradores: cuadrados en los extremos de los semiejes y círculo de giro sobre el superior.
        arm = handles["rotate"]
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3.0))
        painter.drawLine(top, arm)
        painter.setPen(QPen(color, 1.3))
        painter.drawLine(top, arm)
        half = self.HANDLE_SIZE / 2
        for name in ("scale_x", "scale_y"):
            pt = handles[name]
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1.4))
            painter.setBrush(color)
            painter.drawRect(QRectF(pt.x() - half, pt.y() - half, self.HANDLE_SIZE, self.HANDLE_SIZE))
        painter.setPen(QPen(QColor(0, 0, 0, 200), 1.4))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(arm, half + 1, half + 1)
        # Rombos de paneo (izquierda, rotación Y) e inclinación (abajo, rotación X).
        painter.setBrush(color.lighter(130))
        for name in ("pan", "tilt"):
            pt, r = handles[name], half + 1.5
            painter.drawPolygon(QPolygonF([QPointF(pt.x(), pt.y() - r), QPointF(pt.x() + r, pt.y()),
                                           QPointF(pt.x(), pt.y() + r), QPointF(pt.x() - r, pt.y())]))

    def _draw_brush(self, painter: QPainter, to_view: QTransform) -> None:
        """Vista previa del tamaño y la suavidad junto al puntero al usar «dibujar» o «colocar»."""
        if not self._tool or self._brush is None or self._cursor_pos is None or self._active is not None:
            return
        scene = self.mapToScene(self._cursor_pos)
        if not self._image_rect().contains(scene):
            return
        ov = {**self._brush, "cx": scene.x(), "cy": scene.y(), "selected": False, "name": "",
              "color": self._brush.get("color", (1.0, 1.0, 1.0))}
        outer = self._area_path(ov, 1.0, to_view)
        painter.setBrush(QColor(255, 255, 255, 18))
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3.0))
        painter.drawPath(outer)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.4))
        painter.drawPath(outer)
        soft = float(ov.get("soft", 0.0))
        if soft > 0.03:
            inner = self._area_path(ov, max(1.0 - soft, 0.02), to_view)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 110), 2.2))
            painter.drawPath(inner)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0, Qt.DotLine))
            painter.drawPath(inner)

    # --- Geometría de la edición de áreas ---------------------------------------------------
    def _handles(self, ov: dict) -> dict[str, QPointF]:
        """Posición en pantalla de los tiradores de un área: scale_x, scale_y y rotate."""
        ux, uy = axes(ov.get("angle", 0.0))
        cx, cy, rx, ry = ov["cx"], ov["cy"], ov["rx"], ov["ry"]
        right = QPointF(self.mapFromScene(QPointF(cx + ux[0] * rx, cy + ux[1] * rx)))
        top = QPointF(self.mapFromScene(QPointF(cx - uy[0] * ry, cy - uy[1] * ry)))
        centre = QPointF(self.mapFromScene(QPointF(cx, cy)))
        direction = top - centre
        length = math.hypot(direction.x(), direction.y())
        unit = direction / length if length > 1e-6 else QPointF(0, -1)
        # «pan» y «tilt» están en los lados opuestos a los de escala: giran el área sobre su eje vertical / horizontal
        return {"scale_x": right, "scale_y": top, "rotate": top + unit * self.ROTATE_ARM,
                "pan": centre * 2 - right, "tilt": centre * 2 - top}

    def _inside(self, ov: dict, view_pos: QPointF) -> bool:
        """¿Está el punto de pantalla dentro del área (con un pequeño margen para agarrarla)?"""
        p = self.mapToScene(view_pos.toPoint())
        zoom = max(self.viewportTransform().m11(), 1e-6)
        margin = 4.0 / zoom
        if ov.get("poly"):                      # área inclinada: dentro de su contorno real
            return QPolygonF([QPointF(x, y) for x, y in ov["poly"]]).containsPoint(p, Qt.OddEvenFill)
        ux, uy = axes(ov.get("angle", 0.0))
        dx, dy = p.x() - ov["cx"], p.y() - ov["cy"]
        lx, ly = dx * ux[0] + dy * ux[1], dx * uy[0] + dy * uy[1]
        rx, ry = ov["rx"] + margin, ov["ry"] + margin
        if ov.get("box"):
            return abs(lx) <= rx and abs(ly) <= ry
        return (lx / rx) ** 2 + (ly / ry) ** 2 <= 1.0

    def _hit_test(self, view_pos: QPointF) -> tuple[int, str] | None:
        """(índice del área, gesto) bajo el puntero: primero los tiradores del área seleccionada,
        luego el interior de las áreas (la seleccionada, y las demás de arriba abajo)."""
        selected = next((ov for ov in self._overlays if ov.get("selected")), None)
        if selected is not None:
            handles = self._handles(selected)
            for name in ("rotate", "scale_x", "scale_y", "pan", "tilt"):
                d = handles[name] - view_pos
                if math.hypot(d.x(), d.y()) <= self.HANDLE_TOLERANCE:
                    return selected["index"], name
        order = ([selected] if selected is not None else []) + \
                [ov for ov in reversed(self._overlays) if not ov.get("selected")]
        for ov in order:
            if self._inside(ov, view_pos):
                return ov["index"], "move"
        return None

    def _selected_index(self) -> int:
        return next((ov["index"] for ov in self._overlays if ov.get("selected")), -1)

    # --- Gestos --------------------------------------------------------------------------------
    def _emit_gesture(self, phase: str, view_pos, modifiers=None) -> None:
        if self._active is None:
            return
        scene = self.mapToScene(view_pos.toPoint() if hasattr(view_pos, "toPoint") else view_pos)
        mods = int((modifiers if modifiers is not None else QApplication.keyboardModifiers()).value)
        self.gesture.emit({"kind": self._active["kind"], "phase": phase, "index": self._active["index"],
                           "x": scene.x(), "y": scene.y(), "mods": mods, "axis": self._axis})

    def _begin_gesture(self, kind: str, index: int, view_pos, modal: bool = False) -> None:
        self._active = {"kind": kind, "index": index, "modal": modal}
        self._axis = ""
        self._emit_gesture("press", view_pos)
        self._update_cursor()

    def _end_gesture(self, cancel: bool = False, view_pos=None) -> None:
        if self._active is None:
            return
        pos = view_pos if view_pos is not None else (self._cursor_pos or self.viewport().rect().center())
        self._emit_gesture("cancel" if cancel else "release", pos)
        self._active = None
        self._axis = ""
        self.hint.emit("")
        self._update_cursor()
        self.viewport().update()

    _MODAL_KEYS = {Qt.Key_G: ("move", "G · Mover"), Qt.Key_S: ("scale_all", "S · Escalar"),
                   Qt.Key_R: ("rotate_free", "R · Girar")}

    def _modal_hint(self, name: str) -> str:
        axis = f" · eje {self._axis.upper()}" if self._axis else ""
        extra = self.tr(" · Mayús: pegar a la superficie") if name.startswith("G") else \
            (self.tr(" · Ctrl: de 15° en 15°") if name.startswith("R") else "")
        return (self.tr("{0}{1} — X / Y / Z: fijar el eje{2} · clic o Intro: aceptar · Esc: cancelar").format(name, axis, extra))

    def _update_cursor(self, view_pos=None) -> None:
        """Forma del puntero según lo que haya debajo: tirador, área, herramienta o mano."""
        vp = self.viewport()
        if self._active is not None:
            vp.setCursor({"scale_x": Qt.SizeHorCursor, "scale_y": Qt.SizeVerCursor, "pan": Qt.SizeHorCursor,
                          "tilt": Qt.SizeVerCursor, "rotate": Qt.CrossCursor}.get(self._active["kind"], Qt.SizeAllCursor))
            return
        if self._tool:
            vp.setCursor(Qt.CrossCursor)
            return
        pos = view_pos if view_pos is not None else self._cursor_pos
        if pos is not None and self._overlays and self._item is not None:
            hit = self._hit_test(QPointF(pos))
            if hit is not None:
                vp.setCursor({"scale_x": Qt.SizeHorCursor, "scale_y": Qt.SizeVerCursor, "pan": Qt.SizeHorCursor,
                              "tilt": Qt.SizeVerCursor, "rotate": Qt.PointingHandCursor}.get(hit[1], Qt.SizeAllCursor))
                return
        vp.setCursor(Qt.OpenHandCursor)

    @staticmethod
    def _draw_label(painter: QPainter, text: str, anchor: QPointF, align, color: QColor | None = None) -> None:
        if not text:
            return
        metrics = painter.fontMetrics()
        w, h = metrics.horizontalAdvance(text) + 14, metrics.height() + 6
        x = anchor.x() - w if align == Qt.AlignRight else anchor.x() - w / 2 if align == Qt.AlignHCenter else anchor.x()
        box = QRectF(x, anchor.y(), w, h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(color if color is not None else QColor("#e5e5e5"))
        painter.drawText(box, Qt.AlignCenter, text)

    # --- Zoom ------------------------------------------------------------------
    def has_image(self) -> bool:
        return self._item is not None

    def fit_to_window(self) -> None:
        if self._item is not None:
            self._fit_mode = True
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_actual_size(self) -> None:
        self._fit_mode = False
        self.resetTransform()

    def zoom(self, factor: float) -> None:
        if self._item is not None:
            self._fit_mode = False
            self.scale(factor, factor)

    # --- Eventos ------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._item is None:
            return
        delta = event.angleDelta().y() or event.angleDelta().x()   # con Alt, algunos sistemas giran el eje
        mods = event.modifiers()
        edit = {Qt.ControlModifier: "scale", Qt.ShiftModifier: "depth", Qt.AltModifier: "softness"}.get(
            Qt.KeyboardModifier(mods & (Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier)))
        if edit and self._overlays and self.has_selected_overlay():
            self.areaWheel.emit(edit, delta / 120.0)
            event.accept()
            return
        self.zoom(self.ZOOM_STEP if delta > 0 else 1 / self.ZOOM_STEP)

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        if event.button() == Qt.MiddleButton:          # botón central: desplazar la vista
            self._panning, self._pan_last = True, event.position().toPoint()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self._active is not None and self._active["modal"]:     # gesto con teclado en curso
            self._end_gesture(cancel=event.button() == Qt.RightButton, view_pos=pos)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._item is not None:
            # Cerca del divisor: se arrastra el divisor en vez de desplazar la imagen.
            if self._near_divider(pos):
                self._dragging_split = True
                self.setDragMode(QGraphicsView.NoDrag)
                self._split_from_pos(pos.toPoint())
                event.accept()
                return
            if self._tool and self._image_rect().contains(self.mapToScene(pos.toPoint())):
                self._begin_gesture(self._tool, self._selected_index(), pos)
                event.accept()
                return
            if self._overlays and not self._tool:
                hit = self._hit_test(QPointF(pos))
                if hit is not None:
                    self._begin_gesture(hit[1], hit[0], pos)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._cursor_pos = pos.toPoint()
        if self._panning:
            d = self._cursor_pos - self._pan_last
            self._pan_last = self._cursor_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            event.accept()
            return
        if self._dragging_split:
            self._split_from_pos(pos.toPoint())
            event.accept()
            return
        if self._active is not None:
            self._emit_gesture("move", pos, event.modifiers())
            event.accept()
            return
        if self._near_divider(pos):
            self.viewport().setCursor(Qt.SplitHCursor if self._orientation == Qt.Horizontal
                                      else Qt.SplitVCursor)
        elif not (event.buttons() & Qt.LeftButton):
            self._update_cursor(pos.toPoint())
        if self._tool:
            self.viewport().update()      # la vista previa del pincel sigue al puntero
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._update_cursor()
            event.accept()
            return
        if self._dragging_split:
            self._dragging_split = False
            self.setDragMode(QGraphicsView.NoDrag if self._tool else QGraphicsView.ScrollHandDrag)
            event.accept()
            return
        if self._active is not None and not self._active["modal"] and event.button() == Qt.LeftButton:
            self._end_gesture(view_pos=event.position())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def event(self, event) -> bool:
        # Durante G / S / R las teclas X / Y / Z son del eje: que el atajo global de «Z» (zona en foco) no se las lleve.
        if event.type() == QEvent.ShortcutOverride and self._active is not None and self._active["modal"]:
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if self._active is not None and self._active["modal"]:
            axis = {Qt.Key_X: "x", Qt.Key_Y: "y", Qt.Key_Z: "z"}.get(key)
            if axis:
                self._axis = "" if self._axis == axis else axis   # pulsar de nuevo la quita
                self._emit_gesture("axis", self._cursor_pos or self.viewport().rect().center())
                name = next(n for k, (kind, n) in self._MODAL_KEYS.items() if kind == self._active["kind"])
                self.hint.emit(self._modal_hint(name))
            elif key in (Qt.Key_Return, Qt.Key_Enter):
                self._end_gesture()
            elif key == Qt.Key_Escape:
                self._end_gesture(cancel=True)
            event.accept()
            return
        if (key in self._MODAL_KEYS and not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)
                and self._overlays and self._item is not None and self._active is None and not self._tool
                and self._cursor_pos is not None and self.has_selected_overlay()):
            kind, name = self._MODAL_KEYS[key]
            self._begin_gesture(kind, self._selected_index(), self._cursor_pos, modal=True)
            self.hint.emit(self._modal_hint(name))
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        if self._active is not None and self._active["modal"]:
            self._end_gesture(cancel=True)      # perder el foco cancela el gesto con teclado
        super().focusOutEvent(event)

    def enterEvent(self, event) -> None:
        """Al entrar el ratón, el lienzo toma el teclado (para G / S / R), salvo que se esté
        escribiendo en un campo de texto."""
        focus = QApplication.focusWidget()
        if not isinstance(focus, (QLineEdit, QPlainTextEdit, QAbstractSpinBox)):
            self.setFocus(Qt.MouseFocusReason)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._empty.resize(self.viewport().size())
        if self._fit_mode:
            self.fit_to_window()
