"""Quitar fondo: máscara del sujeto con SAM 2.1 (transformers, Apache 2.0) y fondo negro, blanco o difuminado.

Llamada al modelo según la documentación oficial de transformers (Sam2Model + Sam2Processor): un punto positivo
(label 1) con `input_points = [[[[x, y]]]]`, `multimask_output=True` (3 máscaras con su `iou_scores`) y
`post_process_masks`. Todo en local; los pesos los instala el gestor de modelos (solo safetensors, sin código remoto).
"""

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter

# La carga de SAM 2.1, el punto automático y la segmentación viven en deepshot/maps.py (los comparte con los mapas
# auxiliares del paso Look, p. ej. «Sujeto»): un solo modelo cargado y una sola copia de la lógica.
from deepshot.maps import SAM_MODEL_ID as MODEL_ID, WORK_SIDE  # noqa: F401,E402
from deepshot.maps import auto_point as _auto_point, load_sam as _load, segment_at as _segment  # noqa: F401,E402

_last = {}                             # última máscara calculada: {"key": ..., "mask": ...}


def _mask(image, depth, params):
    """La máscara elegida con «Alcance»: 0 = la más pequeña (una parte), 1 = la mayor (el conjunto)."""
    px, py = float(params.get("point_x", -1.0)), float(params.get("point_y", -1.0))
    point = (min(px, 1.0), min(py, 1.0)) if px >= 0 and py >= 0 else _auto_point(depth)
    key = hashlib.blake2b(image.tobytes() + repr(point).encode(), digest_size=16).hexdigest()
    if _last.get("key") != key:                 # la vista previa repite la misma imagen al mover otros ajustes
        _last.update(key=key, masks=_segment(image, point))
    masks = _last["masks"]
    return masks[int(round(min(max(float(params.get("extent", 1.0)), 0.0), 1.0) * (len(masks) - 1)))]


def process(image, depth, params):
    h, w = image.shape[:2]
    mask = _mask(image, depth, params)
    feather = float(params.get("feather", 0.3)) / 100.0 * max(h, w)
    if feather > 0.3:
        mask = gaussian_filter(mask, feather)
    mode = int(round(float(params.get("background", 0))))
    if mode == 1:
        background = np.ones_like(image)
    elif mode == 2:
        sigma = max(float(params.get("blur", 3.0)) / 100.0 * max(h, w), 0.3)
        background = np.stack([gaussian_filter(image[..., c], sigma) for c in range(3)], axis=-1)
    else:
        background = np.zeros_like(image)
    m = mask[..., None]
    return np.clip(image * m + background * (1.0 - m), 0.0, 1.0).astype(np.float32)
