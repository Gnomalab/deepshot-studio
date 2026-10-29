"""Mezcla del resultado con la foto original y máscaras.

Imágenes float32 0..1, forma (alto, ancho, 3). Máscaras (alto, ancho), 0..1.
Fórmulas de los modos de fusión según la especificación W3C "Compositing and Blending".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from deepshot.tonemap import smoothstep

BLEND_MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "add",
               "darken", "lighten", "difference")


def _soft_light(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Luz suave W3C. b = fondo, s = capa superior."""
    d = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(b))
    return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (d - b))


def blend_pixels(base: np.ndarray, top: np.ndarray, mode: str) -> np.ndarray:
    """Resultado del modo de fusión con opacidad 100 % (base = fondo, top = capa)."""
    if mode == "normal":
        r = top
    elif mode == "multiply":
        r = base * top
    elif mode == "screen":
        r = base + top - base * top
    elif mode == "overlay":   # superponer: luz fuerte con las capas intercambiadas
        r = np.where(base <= 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
    elif mode == "soft_light":
        r = _soft_light(base, top)
    elif mode == "add":
        r = np.minimum(base + top, 1.0)
    elif mode == "darken":
        r = np.minimum(base, top)
    elif mode == "lighten":
        r = np.maximum(base, top)
    elif mode == "difference":
        r = np.abs(base - top)
    else:
        raise ValueError(f"Modo de fusión desconocido: {mode}")
    return r.astype(np.float32)


def blend(base: np.ndarray, top: np.ndarray, mode: str = "normal", opacity: float = 1.0,
          mask: np.ndarray | None = None) -> np.ndarray:
    """Aplica `top` sobre `base` con un modo, una opacidad y una máscara opcional."""
    mixed = blend_pixels(base, top, mode)
    alpha = np.float32(min(max(opacity, 0.0), 1.0))
    if mask is not None:
        alpha = alpha * np.clip(mask, 0.0, 1.0)[..., None]
    return np.clip(base + (mixed - base) * alpha, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Máscaras
# ---------------------------------------------------------------------------
MAX_FEATHER = 0.03          # a suavizado 1, el difuminado de la máscara es el 3 % del lado largo de la imagen
SNAP_RADIUS = (0.005, 0.035)   # radio de la ventana del ajuste a los bordes (mínimo, máximo) como fracción del lado largo
SNAP_EPS = 1e-3             # cuanto más bajo, más se pega la máscara a los bordes de la foto


def refine_mask(mask: np.ndarray, guide: np.ndarray | None, feather: float = 0.0, snap: float = 0.0) -> np.ndarray:
    """Retoca una máscara (alto, ancho) 0..1 sea cual sea su origen (áreas, rango, pincel).

    feather (0..1): difumina la máscara (suaviza el borde duro). snap (0..1): ajusta su borde a los bordes de la foto
    `guide` (alto, ancho, 3) con el guided filter: el pincel no tiene que ser preciso, la máscara se pega al objeto."""
    out = np.asarray(mask, dtype=np.float32)
    long_side = float(max(out.shape))
    if snap > 0.0 and guide is not None and guide.shape[:2] == out.shape:
        from deepshot.refine import guided_filter_color

        lo, hi = SNAP_RADIUS                                 # más ajuste = ventana más grande = alcanza bordes más lejanos
        radius = max(2, int(round((lo + (hi - lo) * snap) * long_side)))
        refined = np.clip(guided_filter_color(np.asarray(guide, dtype=np.float32), out, radius, SNAP_EPS), 0.0, 1.0)
        out = out * (1.0 - snap) + refined.astype(np.float32) * snap
    if feather > 0.0:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(out, sigma=max(feather * MAX_FEATHER * long_side, 0.3))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def depth_range_mask(depth: np.ndarray, near: float, far: float, softness: float = 0.05) -> np.ndarray:
    """1 entre las profundidades `near` y `far`, con transición suave a cada lado."""
    lo, hi = min(near, far), max(near, far)
    if softness <= 0:
        return ((depth >= lo) & (depth <= hi)).astype(np.float32)
    return smoothstep(lo - softness, lo, depth) * smoothstep(hi + softness, hi, depth)


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Lee una máscara pintada (PNG gris, 8 o 16 bits). size = (ancho, alto) para reescalar."""
    with Image.open(path) as img:
        if img.mode.startswith("I"):
            data = np.asarray(img, dtype=np.float32) / 65535.0
        else:
            data = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        if size and (data.shape[1], data.shape[0]) != size:
            data = np.asarray(Image.fromarray(data, mode="F").resize(size, Image.BILINEAR))
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Guarda una máscara como PNG de 16 bits en gris."""
    Image.fromarray(np.round(np.clip(mask, 0, 1) * 65535).astype(np.uint16)).save(path)
