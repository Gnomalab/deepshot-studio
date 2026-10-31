"""Reencuadre previo: una región de la foto con la que se trabaja, sin tocar el archivo original (sin interfaz).

El rectángulo va en coordenadas normalizadas (0..1) de la foto ya enderezada (con la rotación EXIF aplicada), así que
vale igual con la foto completa que con una versión pequeña. Todo lo que viene después —profundidad, vista previa,
efecto y exportación— trabaja solo con esa región: la profundidad se calcula sobre el recorte (los píxeles del modelo se
gastan en la zona elegida y salen más detalle) y las fases pesadas (refinado, exportación) manejan menos píxeles.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from PySide6.QtCore import QT_TRANSLATE_NOOP

MIN_SIDE = 0.02                      # el rectángulo nunca baja del 2 % de la foto por un lado

# Proporciones predefinidas: (id, texto traducible, ancho/alto). None = libre; "original" = la de la foto entera.
ASPECTS: list[tuple[str, str, float | None]] = [
    ("free", QT_TRANSLATE_NOOP("Framing", "Libre"), None),
    ("original", QT_TRANSLATE_NOOP("Framing", "Original"), None),
    ("1:1", "1:1", 1.0),
    ("4:5", "4:5", 4 / 5),
    ("3:2", "3:2", 3 / 2),
    ("16:9", "16:9", 16 / 9),
    ("9:16", "9:16", 9 / 16),
]

Crop = tuple[float, float, float, float]          # (x0, y0, x1, y1), 0..1


def normalize(crop: "Crop | list | None") -> Crop | None:
    """El encuadre limpio: en rango, con el mínimo y ordenado. None si no recorta nada (o no es válido)."""
    if crop is None:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in crop)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        return None
    if x0 <= 0.0005 and y0 <= 0.0005 and x1 >= 0.9995 and y1 >= 0.9995:
        return None                               # toda la foto: es como no tener encuadre
    return (round(x0, 5), round(y0, 5), round(x1, 5), round(y1, 5))


def to_box(crop: Crop | None, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Rectángulo en píxeles (izquierda, arriba, derecha, abajo) para una foto de `size` = (ancho, alto)."""
    w, h = size
    if crop is None:
        return (0, 0, w, h)
    x0, y0, x1, y1 = crop
    left, top = int(round(x0 * w)), int(round(y0 * h))
    right, bottom = max(int(round(x1 * w)), left + 1), max(int(round(y1 * h)), top + 1)
    return (left, top, min(right, w), min(bottom, h))


def cropped_size(crop: Crop | None, size: tuple[int, int]) -> tuple[int, int]:
    left, top, right, bottom = to_box(crop, size)
    return (right - left, bottom - top)


def apply(image: Image.Image, crop: Crop | None) -> Image.Image:
    """La foto recortada (la misma si no hay encuadre)."""
    return image if crop is None else image.crop(to_box(crop, image.size))


def with_aspect(crop: Crop | None, aspect: float, size: tuple[int, int]) -> Crop:
    """El mayor rectángulo con esa proporción (ancho/alto en píxeles) centrado en el actual (o en la foto)."""
    w, h = size
    x0, y0, x1, y1 = crop or (0.0, 0.0, 1.0, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = (x1 - x0) * w, (y1 - y0) * h
    if cw / ch > aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    nw, nh = cw / w, ch / h
    left = min(max(cx - nw / 2, 0.0), 1.0 - nw)
    top = min(max(cy - nh / 2, 0.0), 1.0 - nh)
    return (left, top, left + nw, top + nh)


def depth_to_crop(depth_shape: tuple[int, int], crop: Crop | None) -> tuple[int, int, int, int]:
    """Para un mapa del tamaño de la foto ENTERA: la caja de píxeles que corresponde al encuadre."""
    return to_box(crop, (depth_shape[1], depth_shape[0]))
