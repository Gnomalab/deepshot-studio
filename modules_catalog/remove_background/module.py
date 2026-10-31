"""Quitar fondo: máscara del sujeto con SAM 2.1 (transformers, Apache 2.0) y fondo negro, blanco o difuminado.

Llamada al modelo según la documentación oficial de transformers (Sam2Model + Sam2Processor): un punto positivo
(label 1) con `input_points = [[[[x, y]]]]`, `multimask_output=True` (3 máscaras con su `iou_scores`) y
`post_process_masks`. Todo en local; los pesos los instala el gestor de modelos (solo safetensors, sin código remoto).
"""

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter

MODEL_ID = "sam2-1-hiera-small"
WORK_SIDE = 1024                       # SAM 2 trabaja a 1024 px: no hace falta darle más
_model = {}                            # modelo y procesador cargados (se reutilizan entre vistas previas)
_last = {}                             # última máscara calculada: {"key": ..., "mask": ...}


def _load():
    """(modelo, procesador, dispositivo). Se carga una vez desde la carpeta de modelos del gestor."""
    if not _model:
        import torch
        from transformers import Sam2Model, Sam2Processor

        from deepshot.device import detect_device
        from deepshot.models import ModelManager

        folder = ModelManager().snapshot(MODEL_ID)
        if folder is None:
            raise RuntimeError("Falta el modelo SAM 2.1 Small: instálalo desde el gestor de modelos (Ctrl+M).")
        device = detect_device("auto")
        dtype = torch.float16 if device.kind == "cuda" else torch.float32     # en MPS y CPU, float32 (más fiable)
        model = Sam2Model.from_pretrained(str(folder), dtype=dtype).to(device.kind).eval()
        _model.update(model=model, processor=Sam2Processor.from_pretrained(str(folder)), device=device.kind)
    return _model["model"], _model["processor"], _model["device"]


def _auto_point(depth):
    """El punto más cercano del mapa (0 = cerca) tras suavizarlo: casi siempre cae dentro del sujeto principal."""
    h, w = depth.shape
    smooth = gaussian_filter(depth.astype(np.float32), max(h, w) * 0.02)
    y, x = np.unravel_index(int(np.argmin(smooth)), smooth.shape)
    return (x + 0.5) / w, (y + 0.5) / h


def _segment(image, point):
    """Las máscaras suaves (n, alto, ancho) 0..1 del objeto bajo `point` (x, y en 0..1), de la más pequeña (una parte) a la
    mayor (el conjunto): un solo punto es ambiguo y SAM da tres lecturas."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model, processor, device = _load()
    h, w = image.shape[:2]
    scale = min(1.0, WORK_SIDE / max(h, w))
    small = Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    sw, sh = small.size
    inputs = processor(images=small, input_points=[[[[point[0] * sw, point[1] * sh]]]], input_labels=[[[1]]],
                       return_tensors="pt")
    inputs = {k: (v.to(device, model.dtype) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    # Logits sin binarizar (el borde sale suave); original_sizes es el tamaño de la imagen que le dimos.
    logits = processor.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu(), binarize=False)[0]
    logits = logits.reshape(-1, sh, sw)
    soft = torch.sigmoid(logits * 4.0)[:, None]
    soft = F.interpolate(soft, size=(h, w), mode="bilinear", align_corners=False)[:, 0].numpy().astype(np.float32)
    return soft[np.argsort(soft.reshape(len(soft), -1).sum(axis=1))]


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
