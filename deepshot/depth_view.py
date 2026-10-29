"""Maneras de ver el mapa de profundidad (sin interfaz): gris, en color o con sus zonas marcadas.

Ninguna cambia el mapa real: solo producen una imagen para mostrar o exportar. Convención de la app:
0 = cerca, 1 = lejos; en gris, cerca = claro.
"""

from __future__ import annotations

import numpy as np

from deepshot.depth_control import DepthControl
from deepshot.tonemap import ColorRamp, RampStop, ramp_preset

# Paletas del mapa en color (cerca → lejos). Propias de la app; la clave se guarda en los ajustes.
PALETTES: dict[str, ColorRamp] = {
    "red_blue": ramp_preset("color"),
    "rainbow": ColorRamp([RampStop(0.0, (0.85, 0.10, 0.10)), RampStop(0.25, (0.98, 0.62, 0.10)),
                          RampStop(0.5, (0.90, 0.92, 0.20)), RampStop(0.75, (0.15, 0.75, 0.45)),
                          RampStop(1.0, (0.15, 0.30, 0.90))], "linear"),
    "thermal": ColorRamp([RampStop(0.0, (1.0, 1.0, 0.85)), RampStop(0.25, (0.98, 0.75, 0.15)),
                          RampStop(0.55, (0.85, 0.15, 0.20)), RampStop(0.8, (0.30, 0.05, 0.45)),
                          RampStop(1.0, (0.02, 0.0, 0.10))], "linear"),
    "ocean": ColorRamp([RampStop(0.0, (0.85, 0.97, 1.0)), RampStop(0.4, (0.20, 0.65, 0.80)),
                        RampStop(1.0, (0.02, 0.10, 0.30))], "ease"),
}
# Un color por zona para la vista «zonas» (cerca, medio, lejos y las añadidas).
ZONE_COLORS = [(0.25, 0.55, 1.0), (0.30, 0.85, 0.40), (1.0, 0.60, 0.20),
               (0.90, 0.35, 0.80), (0.95, 0.90, 0.30), (0.30, 0.90, 0.90)]
ZONE_TINT = 0.6                  # cuánto pesa el color de la zona sobre el gris del mapa


def colorize_float(depth: np.ndarray, palette: str = "red_blue") -> np.ndarray:
    """Mapa de profundidad (0 = cerca) coloreado con una paleta: float32 (alto, ancho, 3) en 0..1."""
    ramp = PALETTES.get(palette, PALETTES["red_blue"])
    return np.clip(ramp.apply(np.clip(depth, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)


def colorize(depth: np.ndarray, palette: str = "red_blue") -> np.ndarray:
    """Igual que `colorize_float`, en uint8 (alto, ancho, 3), para mostrar en pantalla."""
    return np.round(colorize_float(depth, palette) * 255.0).astype(np.uint8)


def normals_image(depth: np.ndarray, convention: str = "opengl", strength: float = 1.0) -> np.ndarray:
    """Mapa de normales calculado a partir del mapa de profundidad, como el que crea el modo rápido: uint8 (alto, ancho, 3).

    Usa los mismos cálculos que `simple_depth.normal_map` (mapa normalizado a todo el rango), así que lo que se ve aquí es
    lo que se exporta. `convention`: «opengl» (verde hacia arriba) o «directx» (verde hacia abajo)."""
    from deepshot import simple_depth

    base = simple_depth.normalize_range(depth)
    return np.round(simple_depth.normal_map(base, strength, convention) * 255.0).astype(np.uint8)


def zones_image(depth: np.ndarray, control: DepthControl) -> np.ndarray:
    """El mapa en gris con cada zona del control de profundidad teñida de su color: uint8 (alto, ancho, 3).

    Sirve para ver qué parte del mapa es de cada zona antes de moverla. Sin zonas activas es el mapa en gris."""
    d = np.clip(depth, 0.0, 1.0).astype(np.float32)
    gray = np.repeat((1.0 - d)[..., None], 3, axis=-1)
    weights = control.zone_weights(d)
    if not weights:
        return np.round(gray * 255.0).astype(np.uint8)
    names = control.zone_names()
    tint = np.zeros_like(gray)
    total = np.zeros(d.shape, dtype=np.float32)
    for name, w in weights:
        color = np.array(ZONE_COLORS[names.index(name) % len(ZONE_COLORS)], dtype=np.float32)
        tint += w[..., None] * color
        total += w
    total = np.maximum(total, 1.0)                    # zonas manuales que se solapan: se reparte el color
    alpha = (np.minimum(total, 1.0) * ZONE_TINT)[..., None]
    out = gray * (1.0 - alpha) + (tint / total[..., None]) * alpha
    return np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
