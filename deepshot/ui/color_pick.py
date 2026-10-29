"""Selector de color con vista previa en vivo.

El diálogo propio de Qt (no el nativo de Windows) avisa en cada cambio del color, así la
imagen se actualiza mientras se elige y se puede cancelar sin haber aplicado nada.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QWidget

Color = tuple[float, float, float]


def pick_color(parent: QWidget | None, initial: Color, title: str,
               on_live: Callable[[Color], None]) -> Color | None:
    """Abre el selector. `on_live(color)` se llama en cada cambio (también al cancelar, con el
    color inicial, para dejarlo como estaba). Devuelve el color aceptado o None si se cancela."""
    dialog = QColorDialog(QColor.fromRgbF(*initial), parent)
    dialog.setWindowTitle(title)
    dialog.setOption(QColorDialog.DontUseNativeDialog, True)   # el nativo no avisa de cada cambio

    def changed(color: QColor) -> None:
        if color.isValid():
            on_live((color.redF(), color.greenF(), color.blueF()))

    dialog.currentColorChanged.connect(changed)
    if dialog.exec() == QColorDialog.Accepted:
        color = dialog.currentColor()
        accepted = (color.redF(), color.greenF(), color.blueF())
        on_live(accepted)
        return accepted
    on_live(initial)
    return None
