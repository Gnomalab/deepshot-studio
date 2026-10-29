"""Iconos vectoriales de la interfaz, dibujados con QPainter (sin archivos ni recursos de terceros).

Cada icono se define como una pequeña función que dibuja sobre una cuadrícula de 24×24.
`icon(nombre)` devuelve un QIcon con dos colores: gris (reposo) y blanco (activo / marcado),
nítido a cualquier tamaño y en pantallas de alta densidad.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from deepshot.ui import icons_color, theme

GRID = 24.0
SIZES = (16, 20, 24, 32, 48)          # tamaños (px lógicos) con los que se genera cada icono


def _pen(color: QColor, width: float = 1.8) -> QPen:
    return QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)


def _poly(p: QPainter, *points: tuple[float, float], closed: bool = False) -> None:
    path = QPainterPath(QPointF(*points[0]))
    for pt in points[1:]:
        path.lineTo(*pt)
    if closed:
        path.closeSubpath()
    p.drawPath(path)


# --- Cada icono: función (painter, color) -> None, en cuadrícula 24×24 -----------------------
def _import(p: QPainter, c: QColor) -> None:           # flecha que entra en una bandeja
    p.setPen(_pen(c))
    _poly(p, (4, 15), (4, 19), (20, 19), (20, 15))
    _poly(p, (12, 4), (12, 14))
    _poly(p, (8, 10.5), (12, 14.5), (16, 10.5))


def _export(p: QPainter, c: QColor) -> None:           # flecha que sale de una bandeja
    p.setPen(_pen(c))
    _poly(p, (4, 15), (4, 19), (20, 19), (20, 15))
    _poly(p, (12, 15), (12, 5))
    _poly(p, (8, 8.5), (12, 4.5), (16, 8.5))


def _layers(p: QPainter, c: QColor) -> None:           # capas apiladas: profundidad
    p.setPen(_pen(c))
    _poly(p, (12, 3.5), (21, 8.5), (12, 13.5), (3, 8.5), closed=True)
    _poly(p, (3, 12.5), (12, 17.5), (21, 12.5))
    _poly(p, (3, 16.5), (12, 21.5), (21, 16.5))


def _sliders(p: QPainter, c: QColor) -> None:          # tres deslizadores: ajustes
    p.setPen(_pen(c))
    for y in (6, 12, 18):
        _poly(p, (4, y), (20, y))
    p.setBrush(theme.color("window"))
    for y, x in ((6, 8), (12, 16), (18, 10)):
        p.drawEllipse(QPointF(x, y), 2.6, 2.6)


def _photo(p: QPainter, c: QColor) -> None:            # marco con montaña y sol
    p.setPen(_pen(c))
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2, 2)
    _poly(p, (4.5, 17), (9.5, 11.5), (13, 15), (15.5, 12.5), (19.5, 17))
    p.drawEllipse(QPointF(16.5, 8.6), 1.5, 1.5)


def _depth(p: QPainter, c: QColor) -> None:            # recuadro con degradado
    grad = QLinearGradient(3.5, 0, 20.5, 0)
    grad.setColorAt(0.0, c)
    grad.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 25))
    p.setPen(_pen(c, 1.6))
    p.setBrush(grad)
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2, 2)


def _effect(p: QPainter, c: QColor) -> None:           # destello de cuatro puntas + uno pequeño
    p.setPen(_pen(c, 1.7))
    path = QPainterPath(QPointF(10, 3.5))
    path.cubicTo(10.6, 8.2, 11.8, 9.4, 16.5, 10)
    path.cubicTo(11.8, 10.6, 10.6, 11.8, 10, 16.5)
    path.cubicTo(9.4, 11.8, 8.2, 10.6, 3.5, 10)
    path.cubicTo(8.2, 9.4, 9.4, 8.2, 10, 3.5)
    p.drawPath(path)
    _poly(p, (18, 14.5), (18, 20.5))
    _poly(p, (15, 17.5), (21, 17.5))


def _compare(p: QPainter, c: QColor) -> None:          # recuadro partido por un divisor
    p.setPen(_pen(c, 1.6))
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2, 2)
    p.setBrush(c)
    p.drawRect(QRectF(5, 6, 6.5, 12))
    p.setPen(_pen(c, 1.8))
    _poly(p, (12, 3), (12, 21))


def _plus(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 2.0))
    _poly(p, (12, 5), (12, 19))
    _poly(p, (5, 12), (19, 12))


def _minus(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c, 2.0))
    _poly(p, (5, 12), (19, 12))


def _trash(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    _poly(p, (4, 7), (20, 7))
    _poly(p, (9, 7), (9, 4.5), (15, 4.5), (15, 7))
    _poly(p, (6, 7), (7, 20), (17, 20), (18, 7))
    _poly(p, (10, 11), (10, 16))
    _poly(p, (14, 11), (14, 16))


def _brush(p: QPainter, c: QColor) -> None:            # pincel
    p.setPen(_pen(c))
    _poly(p, (19.5, 4.5), (11, 13))
    path = QPainterPath(QPointF(10, 12.5))
    path.cubicTo(7, 12, 5.5, 14.5, 5, 17)
    path.cubicTo(4.7, 18.5, 4, 19, 3.5, 19.5)
    path.cubicTo(6.5, 20.5, 10, 20, 11.5, 17)
    path.cubicTo(12, 15.5, 11.5, 14, 10.5, 13)
    p.drawPath(path)


def _eraser(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    _poly(p, (4.5, 15), (13, 6.5), (19.5, 13), (14, 18.5), closed=True)
    _poly(p, (8.8, 10.7), (15.3, 17.2))
    _poly(p, (14, 18.5), (20, 18.5))


def _draw_area(p: QPainter, c: QColor) -> None:        # elipse punteada con cruz
    pen = _pen(c, 1.7)
    pen.setStyle(Qt.DashLine)
    p.setPen(pen)
    p.drawEllipse(QRectF(3.5, 5.5, 17, 13))
    p.setPen(_pen(c, 1.8))
    _poly(p, (12, 9.5), (12, 14.5))
    _poly(p, (9.5, 12), (14.5, 12))


def _target(p: QPainter, c: QColor) -> None:           # colocar: punto de mira
    p.setPen(_pen(c, 1.7))
    p.drawEllipse(QPointF(12, 12), 6.5, 6.5)
    _poly(p, (12, 2.5), (12, 7))
    _poly(p, (12, 17), (12, 21.5))
    _poly(p, (2.5, 12), (7, 12))
    _poly(p, (17, 12), (21.5, 12))


def _eye(p: QPainter, c: QColor) -> None:              # mostrar / ocultar: ojo
    p.setPen(_pen(c, 1.7))
    path = QPainterPath()
    path.moveTo(2.5, 12)
    path.quadTo(12, 3.0, 21.5, 12)
    path.quadTo(12, 21.0, 2.5, 12)
    p.drawPath(path)
    p.drawEllipse(QPointF(12, 12), 3.2, 3.2)


def _eye_off(p: QPainter, c: QColor) -> None:          # silenciar: ojo tachado
    _eye(p, c)
    p.setPen(_pen(c, 2.0))
    _poly(p, (4.5, 20), (19.5, 4))


def _cube(p: QPainter, c: QColor) -> None:             # vista 3D: cubo en perspectiva
    p.setPen(_pen(c, 1.7))
    _poly(p, (12, 3), (20, 7.5), (20, 16.5), (12, 21), (4, 16.5), (4, 7.5), closed=True)
    _poly(p, (4, 7.5), (12, 12), (20, 7.5))
    _poly(p, (12, 12), (12, 21))


def _plane(p: QPainter, c: QColor) -> None:            # vista 2D: una lámina plana con su cuadrícula
    p.setPen(_pen(c, 1.7))
    p.drawRoundedRect(QRectF(3.5, 5.5, 17, 13), 1.5, 1.5)
    _poly(p, (9.2, 5.5), (9.2, 18.5))
    _poly(p, (14.8, 5.5), (14.8, 18.5))
    _poly(p, (3.5, 12), (20.5, 12))


def _pipette(p: QPainter, c: QColor) -> None:          # cuentagotas
    p.setPen(_pen(c, 1.8))
    _poly(p, (4.5, 19.5), (5.2, 16.2), (13.2, 8.2), (15.8, 10.8), (7.8, 18.8), closed=True)
    _poly(p, (12.0, 9.4), (14.6, 12.0))
    p.drawEllipse(QPointF(17.4, 6.6), 2.6, 2.6)
    _poly(p, (15.0, 4.2), (19.8, 9.0))


def _clear(p: QPainter, c: QColor) -> None:            # limpiar: escoba / cruz en círculo
    p.setPen(_pen(c, 1.8))
    p.drawEllipse(QPointF(12, 12), 8, 8)
    _poly(p, (8.8, 8.8), (15.2, 15.2))
    _poly(p, (15.2, 8.8), (8.8, 15.2))


def _refresh(p: QPainter, c: QColor) -> None:          # flecha circular
    p.setPen(_pen(c))
    p.drawArc(QRectF(4.5, 4.5, 15, 15), 40 * 16, 270 * 16)
    _poly(p, (19.5, 4.5), (19.5, 10), (14, 10))


def _folder(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    _poly(p, (3.5, 18.5), (3.5, 6.5), (9.5, 6.5), (11.5, 8.5), (20.5, 8.5), (20.5, 18.5), closed=True)


def _copy(p: QPainter, c: QColor) -> None:
    p.setPen(_pen(c))
    p.drawRoundedRect(QRectF(8.5, 8.5, 11, 11), 1.8, 1.8)
    _poly(p, (15.5, 8.5), (15.5, 5.5), (5.5, 5.5), (5.5, 15.5), (8.5, 15.5))


def _save(p: QPainter, c: QColor) -> None:            # guardar preset: disquete
    p.setPen(_pen(c))
    _poly(p, (4.5, 4.5), (17, 4.5), (19.5, 7), (19.5, 19.5), (4.5, 19.5), closed=True)
    p.drawRect(QRectF(8, 4.5, 7, 4.5))
    p.drawRect(QRectF(7.5, 13, 9, 6.5))


def _arrow_up(p: QPainter, c: QColor) -> None:         # subir en la lista
    p.setPen(_pen(c))
    _poly(p, (12, 19), (12, 5))
    _poly(p, (6.5, 10.5), (12, 5), (17.5, 10.5))


def _arrow_down(p: QPainter, c: QColor) -> None:       # bajar en la lista
    p.setPen(_pen(c))
    _poly(p, (12, 5), (12, 19))
    _poly(p, (6.5, 13.5), (12, 19), (17.5, 13.5))


def _look(p: QPainter, c: QColor) -> None:             # círculo medio lleno: ajustes de tono y color (paso Look)
    p.setPen(_pen(c))
    p.drawEllipse(QPointF(12, 12), 8, 8)
    path = QPainterPath()
    path.moveTo(12, 4)
    path.arcTo(QRectF(4, 4, 16, 16), 90, -180)
    path.closeSubpath()
    p.setBrush(c)
    p.drawPath(path)


_DRAWERS: dict[str, Callable[[QPainter, QColor], None]] = {
    "import": _import, "export": _export, "layers": _layers, "sliders": _sliders,
    "photo": _photo, "depth": _depth, "effect": _effect, "compare": _compare,
    "plus": _plus, "minus": _minus, "trash": _trash, "brush": _brush, "eraser": _eraser,
    "draw_area": _draw_area, "target": _target, "clear": _clear, "refresh": _refresh,
    "folder": _folder, "copy": _copy, "save": _save, "eye": _eye, "cube": _cube, "pipette": _pipette,
    "plane": _plane, "eye_off": _eye_off, "arrow_up": _arrow_up, "arrow_down": _arrow_down, "look": _look,
}


def _pixmap(name: str, color: str, size: int, ratio: float) -> QPixmap:
    px = QPixmap(int(size * ratio), int(size * ratio))
    px.setDevicePixelRatio(ratio)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    p.scale(size / GRID, size / GRID)
    p.setBrush(Qt.NoBrush)
    _DRAWERS[name](p, QColor(color))
    p.end()
    return px


def _pixmap_color(name: str, dark_theme: bool, size: int, ratio: float) -> QPixmap:
    """El icono en color (SVG propio de `icons_color.py`) a ese tamaño."""
    from PySide6.QtCore import QByteArray
    from PySide6.QtSvg import QSvgRenderer

    px = QPixmap(int(size * ratio), int(size * ratio))
    px.setDevicePixelRatio(ratio)
    px.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(icons_color.svg(name, dark_theme)))
    p = QPainter(px)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    return px


@lru_cache(maxsize=None)
def _icon(name: str, idle: str, active: str, disabled: str, colored: bool, dark_theme: bool) -> QIcon:
    result = QIcon()
    for size in SIZES:
        for ratio in (1.0, 2.0):
            if colored:      # en color: reposo y activo iguales (el color ya dice qué es); desactivado, en gris
                rest = active_px = _pixmap_color(name, dark_theme, size, ratio)
            else:
                rest, active_px = _pixmap(name, idle, size, ratio), _pixmap(name, active, size, ratio)
            result.addPixmap(rest, QIcon.Normal, QIcon.Off)
            result.addPixmap(active_px, QIcon.Normal, QIcon.On)
            result.addPixmap(active_px, QIcon.Active, QIcon.Off)
            result.addPixmap(active_px, QIcon.Active, QIcon.On)
            gray = _pixmap(name, disabled, size, ratio)
            result.addPixmap(gray, QIcon.Disabled, QIcon.Off)
            result.addPixmap(gray, QIcon.Disabled, QIcon.On)
    return result


def icon(name: str) -> QIcon:
    """Icono por nombre (ver `_DRAWERS`) con los colores del tema actual: reposo, activo / marcado y desactivado.

    Según el estilo de iconos de Preferencias (`Appearance.icons`): «color» (un color por categoría) o «mono»
    (una sola tinta, la del tema)."""
    if name not in _DRAWERS:
        raise KeyError(f"Icono desconocido: {name}")
    colored = theme.current_appearance().icons == "color"
    dark_theme = theme.color("window").lightness() < 128
    return _icon(name, theme.color("icon_idle").name(), theme.color("icon_active").name(),
                 theme.color("icon_disabled").name(), colored, dark_theme)


ICON_PROPERTY = "icon_name"


def set_icon(widget, name: str) -> None:
    """Pone un icono a un botón y apunta su nombre, para poder repintarlo al cambiar de tema (`refresh_icons`)."""
    widget.setProperty(ICON_PROPERTY, name)
    widget.setIcon(icon(name))


TAB_ICONS_PROPERTY = "tab_icon_names"


def set_tab_icons(tabs, names: list[str]) -> None:
    """Pone un icono a cada pestaña de un QTabWidget (en el orden dado) y apunta los nombres para repintarlos al cambiar
    de tema (`refresh_icons`)."""
    tabs.setProperty(TAB_ICONS_PROPERTY, list(names))
    for i, name in enumerate(names):
        tabs.setTabIcon(i, icon(name))


def refresh_icons(root: QWidget) -> int:
    """Vuelve a poner los iconos de todos los botones y pestañas de `root` con los colores del tema actual. Devuelve cuántos."""
    count = 0
    for widget in [root, *root.findChildren(QWidget)]:
        name = widget.property(ICON_PROPERTY)
        if name and hasattr(widget, "setIcon"):
            widget.setIcon(icon(str(name)))
            count += 1
        tab_names = widget.property(TAB_ICONS_PROPERTY)
        if tab_names and hasattr(widget, "setTabIcon"):
            for i, tab_name in enumerate(tab_names):
                widget.setTabIcon(i, icon(str(tab_name)))
                count += 1
    return count


def available() -> list[str]:
    return sorted(_DRAWERS)
