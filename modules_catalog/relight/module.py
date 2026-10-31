"""Relighting por profundidad: una luz direccional sobre el relieve del mapa de profundidad (solo numpy y scipy)."""

import numpy as np
from scipy.ndimage import gaussian_filter


def _normals(depth, relief):
    """Normales (alto, ancho, 3) del relieve. Escena como la de la app: x ∈ ±aspecto, y ∈ ±1 (arriba +), z = profundidad
    × relieve, así que un píxel mide 2/alto en la escena y las pendientes salen en unidades comparables."""
    h, w = depth.shape
    # El mapa se suaviza un poco (los mapas de difusión traen ruido fino que con luz rasante parecería arena).
    smooth = gaussian_filter(depth.astype(np.float32), max(h, w) * 0.0015)
    step = 2.0 / h
    dz_dy_img, dz_dx = np.gradient(smooth)                       # filas hacia abajo, columnas hacia la derecha
    slope_x = dz_dx * relief / step
    slope_y = -dz_dy_img * relief / step                          # la y de la escena crece hacia arriba
    normal = np.stack([slope_x, slope_y, -np.ones_like(slope_x)], axis=-1)    # z crece hacia el fondo: la cara mira a -z
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def process(image, depth, params):
    strength = float(params.get("strength", 0.6))
    if strength <= 0:
        return image
    azimuth = np.radians(float(params.get("azimuth", 315.0)))
    elevation = np.radians(float(params.get("elevation", 35.0)))
    # Dirección HACIA la luz: x derecha, y arriba, -z hacia la cámara (la cara del relieve mira a -z).
    light = np.array([np.sin(azimuth) * np.cos(elevation), np.cos(azimuth) * np.cos(elevation), -np.sin(elevation)])
    normal = _normals(depth, float(params.get("relief", 1.0)))
    diffuse = np.clip(normal @ light, 0.0, 1.0)
    flat = max(float(-light[2]), 1e-3)                            # lo que recibe una superficie plana de frente
    lit = np.clip(diffuse / flat, 0.0, 2.0)[..., None]            # 1 = como una pared plana, <1 sombra, >1 luz
    factor = 1.0 + strength * (lit - 1.0)
    warmth = float(params.get("warmth", 0.0))
    tint = 1.0 + 0.25 * warmth * np.maximum(lit - 1.0, 0.0) * np.array([1.0, 0.0, -1.0], dtype=np.float32)
    return np.clip(image * factor * tint, 0.0, 1.0).astype(np.float32)
