"""Nitidez: máscara de enfoque (imagen + cantidad × (imagen − imagen desenfocada)).

El radio va en % del lado largo de la imagen para que la vista previa (a menos resolución) y la exportación a resolución
completa den el mismo aspecto.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def process(image, depth, params):
    amount = float(params["amount"])
    sigma = max(float(params["radius"]) / 100.0 * max(image.shape[:2]), 0.3)
    blurred = np.stack([gaussian_filter(image[..., c], sigma) for c in range(image.shape[2])], axis=-1)
    return np.clip(image + (image - blurred) * amount, 0.0, 1.0)
