"""Campo numérico arrastrable al estilo de Blender.

- Arrastrar en horizontal cambia el valor (Mayús = ajuste fino ×0,1, Ctrl = redondeo al paso).
- Un clic sin arrastrar abre la edición por teclado (Intro acepta, Esc cancela).
- La rueda del ratón sube o baja un paso.
- Con rango (`minimum`/`maximum`) se dibuja la barra de relleno azul.

Señales:
  valueChanged(float)  mientras cambia (vista previa en vivo).
  editingFinished()    al soltar o confirmar (momento de guardar en deshacer).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QLineEdit, QSizePolicy, QWidget

from deepshot.ui import theme

DRAG_THRESHOLD = 3   # píxeles antes de considerar que es un arrastre y no un clic


class _InlineEditor(QLineEdit):
    """Campo de texto que se pone encima del control para teclear un valor.

    Avisa con `closed(aceptar)` al pulsar Intro (True), Esc (False) o al perder el foco por
    cualquier motivo (True). QLineEdit no emite `editingFinished` al cambiar de ventana activa,
    y el campo se quedaba tapando el deslizador.
    """

    closed = Signal(bool)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.closed.emit(True)
        elif event.key() == Qt.Key_Escape:
            self.closed.emit(False)
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.closed.emit(True)


class ValueSlider(QWidget):
    valueChanged = Signal(float)
    editingFinished = Signal()

    def __init__(self, label: str, value: float = 0.0, minimum: float | None = 0.0,
                 maximum: float | None = 1.0, decimals: int = 2, step: float = 0.01,
                 drag_range: float | None = None, suffix: str = "", parent: QWidget | None = None) -> None:
        """drag_range: cuánto cambia el valor al arrastrar todo el ancho del control
        (por defecto, el rango completo; obligatorio si no hay mínimo y máximo)."""
        super().__init__(parent)
        self.label = label
        self.minimum, self.maximum = minimum, maximum
        self.decimals, self.step, self.suffix = decimals, step, suffix
        bounded = minimum is not None and maximum is not None
        self.drag_range = drag_range or ((maximum - minimum) if bounded else 1.0)
        self._value = self._clamp(value)
        self._press_x: float | None = None
        self._press_value = self._value
        self._dragging = False
        self._editor: QLineEdit | None = None
        self.setMinimumHeight(22)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.SizeHorCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    # --- Valor -------------------------------------------------------------------
    def value(self) -> float:
        return self._value

    def setValue(self, value: float, emit: bool = False) -> None:
        value = self._clamp(value)
        if value != self._value:
            self._value = value
            self.update()
            if emit:
                self.valueChanged.emit(value)

    def _clamp(self, v: float) -> float:
        if self.minimum is not None:
            v = max(v, self.minimum)
        if self.maximum is not None:
            v = min(v, self.maximum)
        return round(float(v), self.decimals + 2)

    def text(self) -> str:
        return f"{self._value:.{self.decimals}f}{self.suffix}"

    # --- Dibujo --------------------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 3, 3)
        enabled = self.isEnabled()
        p.fillPath(path, theme.color("button" if enabled else "button_disabled"))
        if self.minimum is not None and self.maximum is not None and self.maximum > self.minimum:
            frac = (self._value - self.minimum) / (self.maximum - self.minimum)
            if frac > 0:
                p.save()
                p.setClipPath(path)
                p.fillRect(QRectF(r.left(), r.top(), r.width() * frac, r.height()),
                           theme.color("accent" if enabled else "accent_disabled"))
                p.restore()
        if self.underMouse() and enabled:
            p.fillPath(path, QColor(255, 255, 255, 18))
        p.setPen(theme.color("text" if enabled else "text_disabled"))
        font = QFont(self.font())
        p.setFont(font)
        text_rect = r.adjusted(8, 0, -8, 0)
        p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.label)
        p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignRight, self.text())

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(22)
        size.setWidth(self.fontMetrics().horizontalAdvance(self.label + "  00.000") + 24)
        return size

    def minimumSizeHint(self):
        return self.sizeHint()

    # --- Ratón ------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.setFocus(Qt.MouseFocusReason)
            self._press_x = event.position().x()
            self._press_value = self._value
            self._dragging = False

    def mouseMoveEvent(self, event) -> None:
        if self._press_x is None:
            return
        dx = event.position().x() - self._press_x
        if not self._dragging and abs(dx) < DRAG_THRESHOLD:
            return
        self._dragging = True
        factor = 0.1 if event.modifiers() & Qt.ShiftModifier else 1.0
        new = self._press_value + dx / max(self.width(), 1) * self.drag_range * factor
        if event.modifiers() & Qt.ControlModifier and self.step:
            new = round(new / self.step) * self.step
        self.setValue(new, emit=True)

    def mouseReleaseEvent(self, event) -> None:
        if self._press_x is None:
            return
        was_drag = self._dragging
        self._press_x = None
        self._dragging = False
        if was_drag:
            self.editingFinished.emit()
        else:
            self._start_edit()

    def wheelEvent(self, event) -> None:
        if not self.isEnabled() or not self.hasFocus():
            event.ignore()  # no robar el desplazamiento del panel si no tiene el foco
            return
        direction = 1 if event.angleDelta().y() > 0 else -1
        self.setValue(self._value + direction * self.step, emit=True)
        self.editingFinished.emit()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)

    # --- Edición por teclado ------------------------------------------------------------
    def is_editing(self) -> bool:
        return self._editor is not None

    def _start_edit(self) -> None:
        if self._editor is not None:
            return
        self._editor = _InlineEditor(f"{self._value:.{self.decimals}f}", self)
        self._editor.setGeometry(self.rect())
        self._editor.setAlignment(Qt.AlignRight)
        self._editor.selectAll()
        self._editor.closed.connect(self._close_edit)
        self._editor.show()
        self._editor.setFocus()

    def _close_edit(self, accept: bool) -> None:
        """Cierra el campo de texto; si `accept`, aplica el número tecleado."""
        if self._editor is None:
            return
        editor, self._editor = self._editor, None
        text = editor.text().replace(",", ".").strip()
        editor.blockSignals(True)
        editor.hide()
        editor.deleteLater()
        if not accept:
            return
        try:
            number = float(text)
        except ValueError:
            return  # texto no numérico: se mantiene el valor anterior
        if number != self._value:
            self.setValue(number, emit=True)
            self.editingFinished.emit()

    # El campo de texto no puede quedarse si el control se oculta, se desactiva o cambia de tamaño.
    def hideEvent(self, event) -> None:
        self._close_edit(False)
        super().hideEvent(event)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.EnabledChange and not self.isEnabled():
            self._close_edit(False)
        super().changeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._editor is not None:
            self._editor.setGeometry(self.rect())
