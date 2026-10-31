"""Editor de la rampa artística (como el "Color Ramp" de Blender).

Barra con el degradado y una parada (triángulo) por cada color:
  - clic: selecciona la parada más cercana;  arrastrar: la mueve;
  - doble clic en una parada (el triángulo): elige su color, con vista previa en vivo;
  - doble clic en la barra: añade una parada con el color de ese punto;
  - Supr: borra la parada seleccionada (siempre queda al menos una).
Debajo: posición y color de la parada, interpolación, presets y "Ajustar al relieve".

Señales: changed() en cada cambio (vista previa) y committed() al terminar (deshacer).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deepshot.ui import theme
from deepshot import presets
from deepshot.tonemap import ColorRamp, RampStop
from deepshot.ui.color_pick import pick_color
from deepshot.ui.preset_bar import PresetBar
from deepshot.ui.value_slider import ValueSlider

BAR_HEIGHT = 22
HANDLE = 7           # medio ancho del triángulo
MARGIN = HANDLE + 1


class RampBar(QWidget):
    """La barra con el degradado y los tiradores."""

    changed = Signal()
    committed = Signal()
    selectionChanged = Signal(int)
    colorRequested = Signal(int)        # doble clic sobre una parada

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ramp = ColorRamp()
        self.selected = 0
        self._drag = False
        self._moved = False
        self.setMinimumHeight(BAR_HEIGHT + HANDLE * 2 + 4)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setToolTip(self.tr("Clic: seleccionar parada · Arrastrar: mover · Doble clic en una parada: "
                                "elegir su color · Doble clic en la barra: añadir parada · Supr: borrar"))

    def set_ramp(self, ramp: ColorRamp) -> None:
        self.ramp = ramp
        self.selected = min(self.selected, len(ramp.stops) - 1)
        self.update()

    # --- Geometría ---
    def _bar_rect(self) -> QRectF:
        return QRectF(MARGIN, 2, self.width() - 2 * MARGIN, BAR_HEIGHT)

    def _x_of(self, position: float) -> float:
        r = self._bar_rect()
        return r.left() + position * r.width()

    def _pos_of(self, x: float) -> float:
        r = self._bar_rect()
        return min(max((x - r.left()) / max(r.width(), 1), 0.0), 1.0)

    # --- Dibujo ---
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._bar_rect()
        width = max(int(r.width()), 2)
        # Degradado real: se calcula con la misma rampa que el procesado.
        preview = ColorRamp(self.ramp.stops, self.ramp.interpolation)
        colors = preview.apply(np.linspace(0, 1, width, dtype=np.float32)[None, :])
        data = np.ascontiguousarray((np.clip(colors, 0, 1) * 255).astype(np.uint8))
        img = QImage(data.data, width, 1, 3 * width, QImage.Format_RGB888).copy()
        p.drawImage(r, img)
        p.setPen(QPen(theme.color("line"), 1))
        p.drawRect(r)
        # Tiradores bajo la barra.
        for i, stop in enumerate(self.ramp.stops):
            x = self._x_of(stop.position)
            top = r.bottom() + 1
            tri = QPolygonF([QPointF(x, top), QPointF(x - HANDLE, top + HANDLE * 1.6),
                             QPointF(x + HANDLE, top + HANDLE * 1.6)])
            p.setBrush(QColor.fromRgbF(*stop.color))
            p.setPen(QPen(theme.color("text_strong") if i == self.selected else theme.color("line"),
                          2 if i == self.selected else 1))
            p.drawPolygon(tri)

    # --- Ratón ---
    def _nearest(self, x: float) -> int:
        return min(range(len(self.ramp.stops)), key=lambda i: abs(self._x_of(self.ramp.stops[i].position) - x))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self.ramp.stops:
            return
        self.selected = self._nearest(event.position().x())
        self._drag, self._moved = True, False
        self.selectionChanged.emit(self.selected)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag:
            self.ramp.stops[self.selected].position = round(self._pos_of(event.position().x()), 3)
            self._moved = True
            self.update()
            self.changed.emit()
            self.selectionChanged.emit(self.selected)

    def mouseReleaseEvent(self, _event) -> None:
        if self._drag and self._moved:
            self.committed.emit()
        self._drag = False

    def _handle_at(self, x: float, y: float) -> int | None:
        """Índice de la parada cuyo triángulo está bajo el ratón (o None)."""
        if y < self._bar_rect().bottom() - 2:
            return None                      # por encima de la zona de los triángulos
        i = self._nearest(x)
        if abs(self._x_of(self.ramp.stops[i].position) - x) <= HANDLE + 3:
            return i
        return None

    def mouseDoubleClickEvent(self, event) -> None:
        if not self.ramp.stops:
            return
        hit = self._handle_at(event.position().x(), event.position().y())
        if hit is not None:                  # doble clic en un triángulo: elegir color
            self.selected = hit
            self.update()
            self.selectionChanged.emit(hit)
            self.colorRequested.emit(hit)
            return
        pos = round(self._pos_of(event.position().x()), 3)
        color = tuple(float(c) for c in ColorRamp(self.ramp.stops, self.ramp.interpolation)
                      .apply(np.array([[pos]], dtype=np.float32))[0, 0])
        self.ramp.stops.append(RampStop(pos, color))
        self.selected = len(self.ramp.stops) - 1
        self.update()
        self.selectionChanged.emit(self.selected)
        self.changed.emit()
        self.committed.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_selected()
        else:
            super().keyPressEvent(event)

    def remove_selected(self) -> None:
        if len(self.ramp.stops) <= 1:
            return
        del self.ramp.stops[self.selected]
        self.selected = max(0, self.selected - 1)
        self.update()
        self.selectionChanged.emit(self.selected)
        self.changed.emit()
        self.committed.emit()


class RampEditor(QWidget):
    """Barra + controles de la parada seleccionada + presets."""

    changed = Signal()
    committed = Signal()
    fitRequested = Signal()
    presetApplying = Signal()           # un preset de rampa va a aplicarse (antes de tocar las paradas)

    INTERPOLATIONS = [("linear", "Lineal"), ("ease", "Suave"), ("constant", "Constante")]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ramp = ColorRamp()
        self.bar = RampBar()
        self.bar.changed.connect(self._on_bar_changed)
        self.bar.committed.connect(self.committed)
        self.bar.selectionChanged.connect(self._sync_stop_controls)

        self.bar.colorRequested.connect(self._pick_color)
        self.preset = PresetBar("ramp", self.tr("Cargar preset de rampa…"),
                                lambda: {"ramp": presets.ramp_to_dict(self.ramp)})
        self.preset.chosen.connect(self._apply_preset)

        add = QPushButton("+")
        add.setToolTip(self.tr("Añadir parada (también: doble clic en la barra)"))
        add.clicked.connect(self._add_stop)
        remove = QPushButton("−")
        remove.setToolTip(self.tr("Borrar la parada seleccionada (Supr)"))
        remove.clicked.connect(self.bar.remove_selected)
        for b in (add, remove):
            b.setFixedWidth(28)
            b.setStyleSheet("padding: 0;")
        self.interp = QComboBox()
        for key, label in self.INTERPOLATIONS:
            self.interp.addItem(self.tr(label), key)
        self.interp.setToolTip(self.tr("Cómo se pasa de una parada a la siguiente"))
        self.interp.activated.connect(self._on_interp)

        self.position = ValueSlider(self.tr("Pos"), 0.0, 0.0, 1.0, decimals=3, step=0.001)
        self.position.setToolTip(self.tr("Posición de la parada seleccionada: 0 = cerca, 1 = lejos"))
        self.position.valueChanged.connect(self._on_position)
        self.position.editingFinished.connect(self.committed)
        self.color_button = QPushButton()
        self.color_button.setFixedWidth(70)
        self.color_button.setToolTip(self.tr("Color de la parada seleccionada (también: doble clic en su triángulo)"))
        self.color_button.clicked.connect(lambda: self._pick_color(self.bar.selected))

        fit = QPushButton(self.tr("Adaptar rampa al rango de profundidad"))
        fit.setToolTip(self.tr("Ajusta la rampa al rango real de profundidad de la foto"))
        fit.clicked.connect(self.fitRequested)
        note = QLabel(self.tr("De cerca (izquierda) a lejos (derecha)"))
        note.setProperty("dim", True)

        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(add)
        row1.addWidget(remove)
        row1.addWidget(self.interp, 1)
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(self.position, 1)
        row2.addWidget(self.color_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.preset)
        layout.addLayout(row1)
        layout.addWidget(self.bar)
        layout.addLayout(row2)
        layout.addWidget(note)
        layout.addWidget(fit)

    # --- Enlace con el proyecto ---
    def set_ramp(self, ramp: ColorRamp) -> None:
        """Muestra (y edita directamente) esta rampa del proyecto."""
        self.ramp = ramp
        self.bar.set_ramp(ramp)
        self.interp.setCurrentIndex(max(0, self.interp.findData(ramp.interpolation)))
        self._sync_stop_controls(self.bar.selected)

    def _sync_stop_controls(self, index: int) -> None:
        if not self.ramp.stops:
            return
        stop = self.ramp.stops[min(index, len(self.ramp.stops) - 1)]
        self.position.setValue(stop.position)
        c = QColor.fromRgbF(*stop.color)
        self.color_button.setStyleSheet(f"background: {c.name()}; border: 1px solid {theme.color('line').name()};")

    # --- Acciones ---
    def _on_bar_changed(self) -> None:
        self.changed.emit()

    def _on_position(self, value: float) -> None:
        self.ramp.stops[self.bar.selected].position = value
        self.bar.update()
        self.changed.emit()

    def _on_interp(self, _i: int) -> None:
        self.ramp.interpolation = self.interp.currentData()
        self.bar.update()
        self.changed.emit()
        self.committed.emit()

    def _pick_color(self, index: int) -> None:
        """Selector de color con vista previa en vivo: la imagen cambia mientras se elige."""
        if not 0 <= index < len(self.ramp.stops):
            return
        stop = self.ramp.stops[index]
        original = stop.color

        def live(color) -> None:
            stop.color = color
            self._sync_stop_controls(index)
            self.bar.update()
            self.changed.emit()

        if pick_color(self, original, self.tr("Color de la parada"), live) is not None:
            self.committed.emit()          # aceptado: un paso de deshacer
        # Si se cancela, `live` ya devolvió el color original y la imagen no cambió.

    def _add_stop(self) -> None:
        stops = sorted(self.ramp.stops, key=lambda s: s.position)
        # Nueva parada a mitad del mayor hueco entre paradas.
        gaps = [(b.position - a.position, (a.position + b.position) / 2) for a, b in zip(stops, stops[1:])]
        pos = max(gaps)[1] if gaps else 0.5
        # Color de la rampa en ese punto (sin el rango de relieve: posición pura 0..1).
        sample = ColorRamp(self.ramp.stops, self.ramp.interpolation).apply(np.array([[pos]], dtype=np.float32))
        color = tuple(float(c) for c in sample[0, 0])
        self.ramp.stops.append(RampStop(round(pos, 3), color))
        self.bar.selected = len(self.ramp.stops) - 1
        self.bar.update()
        self._sync_stop_controls(self.bar.selected)
        self.changed.emit()
        self.committed.emit()

    def _apply_preset(self, preset: dict) -> None:
        """Aplica un preset de rampa (de fábrica o propio): paradas e interpolación."""
        self.presetApplying.emit()
        data = preset["ramp"]
        # Se conserva el rango ajustado al relieve; cambian paradas e interpolación.
        self.ramp.stops = [RampStop(float(s["position"]), tuple(float(c) for c in s["color"]))
                           for s in data["stops"]]
        self.ramp.interpolation = data.get("interpolation", "linear")
        self.bar.selected = 0
        self.set_ramp(self.ramp)
        self.changed.emit()
        self.committed.emit()
