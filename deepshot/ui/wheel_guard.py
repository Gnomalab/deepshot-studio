"""Protección contra cambios accidentales con la rueda del ratón.

Por defecto Qt cambia el valor de un desplegable, una casilla numérica o un deslizador al girar la rueda encima aunque
no se haya tocado: al desplazar un panel largo se cambiaba un preset o un ajuste sin querer. Como en Blender, aquí la
rueda solo cambia esos controles con Ctrl pulsado; sin Ctrl el giro se pasa al panel que se está desplazando.

Se instala una vez en la aplicación (`install(app)`). No toca el lienzo, el visor 3D ni las listas de los desplegables
abiertos."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractItemView, QAbstractScrollArea, QAbstractSpinBox, QApplication, QComboBox, QSlider, QWidget

GUARDED = (QComboBox, QAbstractSpinBox, QSlider)


def _guarded_owner(widget: QWidget) -> QWidget | None:
    """El control protegido al que pertenece `widget` (él mismo, o el desplegable / casilla que lo contiene)."""
    w = widget
    for _ in range(3):                                   # la línea de texto de una casilla numérica es hija suya
        if w is None:
            return None
        if isinstance(w, QAbstractItemView):             # la lista de un desplegable abierto: ahí la rueda es suya
            return None
        if isinstance(w, GUARDED):
            return w
        w = w.parentWidget()
    return None


def _scroll_area_of(widget: QWidget) -> QAbstractScrollArea | None:
    w = widget.parentWidget()
    while w is not None:
        if isinstance(w, QAbstractScrollArea):
            return w
        w = w.parentWidget()
    return None


class WheelGuard(QObject):
    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Wheel or not isinstance(obj, QWidget):
            return False
        owner = _guarded_owner(obj)
        if owner is None or event.modifiers() & Qt.ControlModifier:
            return False
        area = _scroll_area_of(owner)
        if area is not None:                             # el giro desplaza el panel, no cambia el control
            QApplication.sendEvent(area.verticalScrollBar(), event)
        return True                                      # y el control no lo recibe


def install(app: QApplication) -> WheelGuard:
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
