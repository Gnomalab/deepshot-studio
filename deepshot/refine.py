"""Refinado del mapa de profundidad guiado por la foto (guided filter).

Marigold calcula la profundidad a ~768-1536 px y la amplía al tamaño de la foto, así
que los bordes salen algo blandos. El "guided filter" (He, Sun y Tang, 2010) ajusta el
mapa a los bordes reales de la foto: en cada ventanita, la profundidad se aproxima como
una combinación lineal de los colores de la foto. Donde la foto tiene un borde, el mapa
también; donde la foto es lisa, el mapa se suaviza. No inventa detalle: solo alinea.

Se usa la versión "rápida" (He y Sun, 2015): los coeficientes se calculan a resolución
reducida y se aplican a resolución completa. Todo en numpy, sin dependencias nuevas.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def box_filter(x: np.ndarray, r: int) -> np.ndarray:
    """Media en una ventana (2r+1)×(2r+1) con imagen integral; bordes normalizados.

    Admite arrays 2D (alto, ancho) o con canales extra al final (alto, ancho, ...).
    Coste O(N), independiente del radio.
    """
    h, w = x.shape[:2]

    def _cumbox(a: np.ndarray, axis: int, n: int) -> np.ndarray:
        c = np.cumsum(a, axis=axis, dtype=np.float64)
        pad = [(0, 0)] * a.ndim
        pad[axis] = (1, 0)
        c = np.pad(c, pad)
        idx = np.arange(n)
        hi = np.minimum(idx + r + 1, n)
        lo = np.maximum(idx - r, 0)
        return np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)

    s = _cumbox(_cumbox(x, 0, h), 1, w)
    # Número de píxeles de cada ventana (menor en los bordes de la imagen).
    ones = np.ones((h, w), dtype=np.float64)
    count = _cumbox(_cumbox(ones, 0, h), 1, w)
    if x.ndim > 2:
        count = count.reshape(count.shape + (1,) * (x.ndim - 2))
    return (s / count).astype(np.float32)


def _resize(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Redimensiona un array float32 2D (o 3D por canales) con interpolación bilineal."""
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(size, Image.BILINEAR))
    return np.stack([_resize(a[..., c], size) for c in range(a.shape[2])], axis=-1)


def guided_filter_color(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter con guía en color (alto, ancho, 3) y entrada (alto, ancho). Valores 0..1."""
    mean_i = box_filter(guide, radius)                       # (h, w, 3)
    mean_p = box_filter(src, radius)                         # (h, w)
    mean_ip = box_filter(guide * src[..., None], radius)     # (h, w, 3)
    cov_ip = mean_ip - mean_i * mean_p[..., None]

    # Covarianza de los colores en cada ventana: matriz 3×3 por píxel.
    outer = guide[..., :, None] * guide[..., None, :]        # (h, w, 3, 3)
    sigma = box_filter(outer.reshape(*outer.shape[:2], 9), radius).reshape(outer.shape)
    sigma -= mean_i[..., :, None] * mean_i[..., None, :]
    sigma += eps * np.eye(3, dtype=np.float32)               # regularización: evita copiar textura

    a = np.linalg.solve(sigma, cov_ip[..., None])[..., 0]    # (h, w, 3)
    b = mean_p - np.sum(a * mean_i, axis=-1)
    return np.sum(box_filter(a, radius) * guide, axis=-1) + box_filter(b, radius)


def refine_depth(depth: np.ndarray, photo_rgb: np.ndarray, strength: float = 1.0,
                 work_size: int = 1600, radius: int = 6, eps: float = 2e-3) -> np.ndarray:
    """Alinea los bordes del mapa con los de la foto, a la resolución completa de la foto.

    depth:      float32 (alto, ancho), 0..1, mismo tamaño que la foto.
    photo_rgb:  uint8 o float (alto, ancho, 3).
    strength:   0 = sin cambios, 1 = refinado completo (mezcla lineal).
    work_size:  lado mayor de la resolución de trabajo de los coeficientes.
    radius:     radio de la ventana en la resolución de trabajo.
    eps:        regularización; más alto = más suave y menos textura de la foto en el mapa.
    """
    if strength <= 0:
        return depth
    h, w = depth.shape
    guide = photo_rgb.astype(np.float32)
    if guide.max() > 1.0:
        guide /= 255.0

    # Coeficientes a resolución reducida (rápido), aplicados a resolución completa.
    scale = min(1.0, work_size / max(h, w))
    small = (max(1, round(w * scale)), max(1, round(h * scale)))
    g_small = _resize(guide, small)
    d_small = _resize(depth, small)

    mean_i = box_filter(g_small, radius)
    mean_p = box_filter(d_small, radius)
    cov_ip = box_filter(g_small * d_small[..., None], radius) - mean_i * mean_p[..., None]
    outer = g_small[..., :, None] * g_small[..., None, :]
    sigma = box_filter(outer.reshape(*outer.shape[:2], 9), radius).reshape(outer.shape)
    sigma -= mean_i[..., :, None] * mean_i[..., None, :]
    sigma += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(sigma, cov_ip[..., None])[..., 0]
    b = mean_p - np.sum(a * mean_i, axis=-1)
    a = box_filter(a, radius)
    b = box_filter(b, radius)

    if small != (w, h):
        a = _resize(a, (w, h))
        b = _resize(b, (w, h))
    refined = np.sum(a * guide, axis=-1) + b
    out = depth + strength * (refined - depth) if strength < 1 else refined
    return np.clip(out, 0.0, 1.0).astype(np.float32)
