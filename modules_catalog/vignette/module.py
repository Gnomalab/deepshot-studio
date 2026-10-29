"""Viñeta: oscurece o aclara los bordes con una caída suave desde el centro."""

import numpy as np


def process(image, depth, params):
    amount = float(params["amount"])
    h, w = image.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r = np.sqrt(x * x + y * y) / np.sqrt(2.0)                   # 0 en el centro, 1 en las esquinas
    start = float(params["size"]) * 0.8
    width = max(float(params["softness"]), 0.05)
    t = np.clip((r - start) / width, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)                                 # smoothstep
    if amount >= 0:                                             # oscurecer: multiplicar por menos de 1
        out = image * (1.0 - amount * t)[..., None]
    else:                                                       # aclarar: acercar a blanco
        out = image + (1.0 - image) * (-amount * t)[..., None]
    return np.clip(out, 0.0, 1.0)
