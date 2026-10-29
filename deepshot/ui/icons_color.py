"""Iconos de color: la versión «en color» de los iconos de la interfaz (sin archivos ni recursos de terceros).

Cada icono es un pequeño SVG dibujado aquí, en la cuadrícula de 24×24, con un relleno suave y el trazo del color de su
categoría (un color por tipo de elemento o de acción). La paleta tiene dos versiones: tonos claros sobre los temas
oscuros y tonos más profundos sobre los temas claros. `icons.py` los pinta con QtSvg y, al elegir el estilo
«monocromo» en Preferencias, usa en su lugar los trazos de una sola tinta.
"""

from __future__ import annotations

# Un color por categoría. (oscuro = para temas oscuros, claro = para temas claros)
PALETTE_DARK = {
    "blue": "#6aa3e8", "green": "#7cc576", "red": "#e8685f", "orange": "#f0a050", "yellow": "#e9c458",
    "purple": "#a98ee8", "teal": "#4fc3b0", "pink": "#e88fb5", "gray": "#b8b8b8", "cyan": "#5fc6e8",
}
PALETTE_LIGHT = {
    "blue": "#2f6fc0", "green": "#3f9a3a", "red": "#c8423a", "orange": "#d9772a", "yellow": "#c99a1c",
    "purple": "#7a5ccc", "teal": "#1f9c88", "pink": "#c9588a", "gray": "#6f6f6f", "cyan": "#1b9fc5",
}

# Categoría de color de cada icono: pasos y vistas, herramientas, acciones.
CATEGORY = {
    "import": "blue", "layers": "purple", "sliders": "orange", "export": "green",
    "photo": "teal", "depth": "purple", "effect": "orange", "compare": "blue", "cube": "orange", "eye": "cyan",
    "brush": "orange", "eraser": "pink", "draw_area": "green", "target": "red", "pipette": "cyan",
    "plus": "green", "minus": "red", "trash": "red", "copy": "blue", "clear": "gray", "refresh": "blue",
    "folder": "yellow", "save": "blue", "plane": "cyan", "eye_off": "gray",
    "arrow_up": "gray", "arrow_down": "gray", "look": "pink",
}


def _fill(c: str, opacity: float = 0.32) -> str:
    return f'fill="{c}" fill-opacity="{opacity}"'


def _stroke(c: str, width: float = 1.8) -> str:
    return f'stroke="{c}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" fill="none"'


def _dot(c: str, cx: float, cy: float, r: float) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{c}" stroke="{c}" stroke-width="1.5"/>'


# Cada función recibe el color de la categoría y devuelve el contenido del SVG.
def _import(c: str) -> str:
    return (f'<path d="M4 15h16v4H4z" {_fill(c)}/>'
            f'<path d="M4 15v4h16v-4M12 4v10M8 10.5l4 4 4-4" {_stroke(c)}/>')


def _export(c: str) -> str:
    return (f'<path d="M4 15h16v4H4z" {_fill(c)}/>'
            f'<path d="M4 15v4h16v-4M12 15V5M8 8.5l4-4 4 4" {_stroke(c)}/>')


def _layers(c: str) -> str:
    return (f'<path d="M12 3.5l9 5-9 5-9-5z" {_fill(c)}/>'
            f'<path d="M12 3.5l9 5-9 5-9-5zM3 12.5l9 5 9-5M3 16.5l9 5 9-5" {_stroke(c)}/>')


def _sliders(c: str) -> str:
    return (f'<path d="M4 6h16M4 12h16M4 18h16" {_stroke(c)}/>'
            + _dot(c, 8, 6, 2.6) + _dot(c, 16, 12, 2.6) + _dot(c, 10, 18, 2.6))


def _photo(c: str) -> str:
    return (f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_fill(c)}/>'
            f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_stroke(c)}/>'
            f'<path d="M4.5 17l5-5.5 3.5 3.5 2.5-2.5 4 4.5z" fill="{c}" fill-opacity="0.6"/>'
            '<circle cx="16.5" cy="8.6" r="1.7" fill="#e9c458"/>')


def _depth(c: str) -> str:
    return (f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_fill(c)}/>'
            f'<path d="M3.5 6.5a2 2 0 0 1 2-2H9v15H5.5a2 2 0 0 1-2-2z" fill="{c}" fill-opacity="0.85"/>'
            f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_stroke(c)}/>'
            f'<path d="M9 4.5v15M14.5 4.5v15" stroke="{c}" stroke-width="1" stroke-opacity="0.6"/>')


_SPARK = "M10 3.5c.6 4.7 1.8 5.9 6.5 6.5-4.7.6-5.9 1.8-6.5 6.5-.6-4.7-1.8-5.9-6.5-6.5 4.7-.6 5.9-1.8 6.5-6.5z"


def _effect(c: str) -> str:
    return (f'<path d="{_SPARK}" {_fill(c)}/>'
            f'<path d="{_SPARK}M18 14.5v6M15 17.5h6" {_stroke(c)}/>')


def _compare(c: str) -> str:
    return (f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_fill(c)}/>'
            f'<rect x="5" y="6" width="6.5" height="12" fill="{c}" fill-opacity="0.8"/>'
            f'<rect x="3.5" y="4.5" width="17" height="15" rx="2" {_stroke(c)}/>'
            '<path d="M12 3v18" stroke="#ffffff" stroke-width="1.8" stroke-linecap="round"/>')


def _plus(c: str) -> str:
    return f'<path d="M12 5v14M5 12h14" {_stroke(c, 2.2)}/>'


def _minus(c: str) -> str:
    return f'<path d="M5 12h14" {_stroke(c, 2.2)}/>'


def _trash(c: str) -> str:
    return (f'<path d="M6 7l1 13h10l1-13z" {_fill(c)}/>'
            f'<path d="M4 7h16M9 7V4.5h6V7M6 7l1 13h10l1-13M10 11v6M14 11v6" {_stroke(c)}/>')


_BRUSH_TIP = "M10 12.5c-3-.5-4.5 2-5 4.5-.3 1.5-1 2-1.5 2.5 3 1 6.5.5 8-2.5.5-1.5 0-3-1.5-4.5z"


def _brush(c: str) -> str:
    return (f'<path d="{_BRUSH_TIP}" fill="{c}" fill-opacity="0.6"/>'
            f'<path d="M19.5 4.5L11 13M{_BRUSH_TIP[1:]}" {_stroke(c)}/>')


def _eraser(c: str) -> str:
    return (f'<path d="M4.5 15L13 6.5 19.5 13 14 18.5z" {_fill(c)}/>'
            f'<path d="M4.5 15L13 6.5 19.5 13 14 18.5zM8.8 10.7l6.5 6.5M14 18.5h6" {_stroke(c)}/>')


def _draw_area(c: str) -> str:
    return (f'<ellipse cx="12" cy="12" rx="8.5" ry="6.5" {_fill(c)}/>'
            f'<ellipse cx="12" cy="12" rx="8.5" ry="6.5" {_stroke(c)} stroke-dasharray="3 2.5"/>'
            f'<path d="M12 9.5v5M9.5 12h5" {_stroke(c)}/>')


def _target(c: str) -> str:
    return (f'<circle cx="12" cy="12" r="6.5" {_fill(c)}/>'
            f'<circle cx="12" cy="12" r="6.5" {_stroke(c)}/>'
            f'<path d="M12 2.5V7M12 17v4.5M2.5 12H7M17 12h4.5" {_stroke(c)}/>'
            f'<circle cx="12" cy="12" r="1.5" fill="{c}"/>')


def _clear(c: str) -> str:
    return (f'<circle cx="12" cy="12" r="8" {_fill(c)}/>'
            f'<circle cx="12" cy="12" r="8" {_stroke(c)}/>'
            f'<path d="M8.8 8.8l6.4 6.4M15.2 8.8l-6.4 6.4" {_stroke(c)}/>')


def _refresh(c: str) -> str:
    return (f'<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3" {_stroke(c)}/>'
            f'<path d="M19.5 4.5V10H14" {_stroke(c)}/>')


def _folder(c: str) -> str:
    return (f'<path d="M3.5 18.5v-12h6l2 2h9v10z" {_fill(c, 0.5)}/>'
            f'<path d="M3.5 18.5v-12h6l2 2h9v10z" {_stroke(c)}/>')


def _copy(c: str) -> str:
    return (f'<rect x="8.5" y="8.5" width="11" height="11" rx="1.8" {_fill(c)}/>'
            f'<rect x="8.5" y="8.5" width="11" height="11" rx="1.8" {_stroke(c)}/>'
            f'<path d="M15.5 8.5v-3h-10v10h3" {_stroke(c)}/>')


def _save(c: str) -> str:
    return (f'<path d="M4.5 4.5H17l2.5 2.5v12.5h-15z" {_fill(c)}/>'
            f'<path d="M4.5 4.5H17l2.5 2.5v12.5h-15zM8 4.5V9h7V4.5M7.5 19.5V13h9v6.5" {_stroke(c)}/>')


def _eye(c: str) -> str:
    return (f'<path d="M2.5 12Q12 3 21.5 12 12 21 2.5 12z" {_fill(c)}/>'
            f'<path d="M2.5 12Q12 3 21.5 12 12 21 2.5 12z" {_stroke(c)}/>'
            '<circle cx="12" cy="12" r="3.2" fill="#4fc3b0"/>')


def _eye_off(c: str) -> str:
    return (f'<path d="M2.5 12Q12 3 21.5 12 12 21 2.5 12z" {_fill(c, 0.18)}/>'
            f'<path d="M2.5 12Q12 3 21.5 12 12 21 2.5 12z" {_stroke(c)}/>'
            f'<circle cx="12" cy="12" r="3.2" fill="{c}" fill-opacity="0.6"/>'
            f'<path d="M4.5 20L19.5 4" {_stroke("#d9534f", 2.2)}/>')


def _cube(c: str) -> str:
    return (f'<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" {_fill(c)}/>'
            f'<path d="M4 7.5l8 4.5 8-4.5" fill="{c}" fill-opacity="0.45"/>'
            f'<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9zM4 7.5l8 4.5 8-4.5M12 12v9" {_stroke(c)}/>')


def _plane(c: str) -> str:
    return (f'<rect x="3.5" y="5.5" width="17" height="13" rx="1.5" {_fill(c)}/>'
            f'<path d="M9.2 5.5v13M14.8 5.5v13M3.5 12h17" {_stroke(c, 1.4)}/>'
            f'<rect x="3.5" y="5.5" width="17" height="13" rx="1.5" {_stroke(c)}/>')


def _pipette(c: str) -> str:
    return (f'<path d="M4.5 19.5l.7-3.3 8-8 2.6 2.6-8 8z" {_fill(c)}/>'
            f'<path d="M4.5 19.5l.7-3.3 8-8 2.6 2.6-8 8zM12 9.4l2.6 2.6" {_stroke(c)}/>'
            f'<circle cx="17.4" cy="6.6" r="2.6" fill="{c}" fill-opacity="0.85"/>')


def _arrow_up(c: str) -> str:
    return f'<path d="M12 19V5M6.5 10.5L12 5l5.5 5.5" {_stroke(c)}/>'


def _arrow_down(c: str) -> str:
    return f'<path d="M12 5v14M6.5 13.5L12 19l5.5-5.5" {_stroke(c)}/>'


def _look(c: str) -> str:
    return (f'<path d="M12 4a8 8 0 0 1 0 16z" {_fill(c, 0.7)}/>'
            f'<circle cx="12" cy="12" r="8" {_stroke(c)}/>')


BUILDERS = {
    "import": _import, "export": _export, "layers": _layers, "sliders": _sliders, "photo": _photo, "depth": _depth,
    "effect": _effect, "compare": _compare, "plus": _plus, "minus": _minus, "trash": _trash, "brush": _brush,
    "eraser": _eraser, "draw_area": _draw_area, "target": _target, "clear": _clear, "refresh": _refresh,
    "folder": _folder, "copy": _copy, "save": _save, "eye": _eye, "cube": _cube, "pipette": _pipette,
    "plane": _plane, "eye_off": _eye_off, "arrow_up": _arrow_up, "arrow_down": _arrow_down, "look": _look,
}


def svg(name: str, dark_theme: bool = True) -> bytes:
    """El icono `name` como documento SVG (cuadrícula de 24×24) con los colores de su categoría."""
    palette = PALETTE_DARK if dark_theme else PALETTE_LIGHT
    body = BUILDERS[name](palette[CATEGORY[name]])
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'.encode("utf-8")
