"""Grano de película: ruido gaussiano filtrado, más fuerte en los medios tonos.

El ruido sale siempre de la misma semilla, así que el mismo proyecto da siempre el mismo resultado. El tamaño del grano
se mide respecto al lado largo de la imagen (1 = un píxel en una imagen de 2000 px de lado largo): la vista previa, que
va a menos resolución, y la exportación a resolución completa se ven igual.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

SEED = 20241021


def process(image, depth, params):
    amount = float(params["amount"])
    h, w = image.shape[:2]
    sigma = max(float(params["size"]) * max(h, w) / 2000.0 * 0.6, 0.0)
    noise = np.random.default_rng(SEED).standard_normal((h, w)).astype(np.float32)
    if sigma > 0.3:
        noise = gaussian_filter(noise, sigma)
    noise /= max(float(noise.std()), 1e-6)                     # mismo contraste de grano sea cual sea el tamaño
    luma = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    weight = 0.35 + 0.65 * 4.0 * luma * (1.0 - luma)           # los medios tonos llevan más grano que negros y blancos
    return np.clip(image + (noise * weight * amount * 0.12)[..., None], 0.0, 1.0)
