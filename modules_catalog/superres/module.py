"""Superresolución: Swin2SR x2 (transformers, Apache 2.0) y vuelta al tamaño original.

Una capa de Look tiene que devolver una imagen del mismo tamaño, así que el resultado x2 del modelo se reduce otra vez
(promedio 2x2): se queda con el detalle recuperado y sin el ruido de compresión. Entrada del modelo según su ficha
oficial: `pixel_values` en 0..1 (su procesador solo reescala y rellena a múltiplos de 8) y salida `reconstruction`.
Se procesa por mosaicos con solape para que quepa en 8 GB de VRAM o en la memoria unificada de un Mac de 16 GB.
"""

import hashlib

import numpy as np

MODEL_ID = "swin2sr-classical-sr-x2"
OVERLAP = 16                           # píxeles extra de contexto alrededor de cada mosaico (se descartan al pegar)
_model = {}
_last = {}                             # último resultado del modelo: mover «Fuerza» no lo vuelve a calcular


def _load():
    """(modelo, dispositivo). Se carga una vez desde la carpeta de modelos del gestor."""
    if not _model:
        import torch
        from transformers import Swin2SRForImageSuperResolution

        from deepshot.device import detect_device
        from deepshot.models import ModelManager

        folder = ModelManager().snapshot(MODEL_ID)
        if folder is None:
            raise RuntimeError("Falta el modelo Swin2SR x2: instálalo desde el gestor de modelos (Ctrl+M).")
        device = detect_device("auto")
        dtype = torch.float16 if device.kind == "cuda" else torch.float32
        model = Swin2SRForImageSuperResolution.from_pretrained(str(folder), dtype=dtype).to(device.kind).eval()
        _model.update(model=model, device=device.kind)
    return _model["model"], _model["device"]


def _upscale_tile(model, device, tile):
    """Un trozo (alto, ancho, 3) 0..1 -> el mismo trozo ampliado x`upscale` con el modelo."""
    import torch
    import torch.nn.functional as F

    h, w = tile.shape[:2]
    window = int(getattr(model.config, "window_size", 8))
    x = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[None].to(device, model.dtype)
    pad_h, pad_w = (-h) % window, (-w) % window                 # el modelo pide múltiplos del tamaño de ventana
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        y = model(pixel_values=x).reconstruction
    scale = int(model.config.upscale)
    return y[:, :, :h * scale, :w * scale].float().clamp_(0, 1)[0].cpu().numpy().transpose(1, 2, 0)


def _restore(image, tile):
    import torch
    import torch.nn.functional as F

    model, device = _load()
    scale = int(model.config.upscale)
    h, w = image.shape[:2]
    out = np.empty_like(image)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(y0 - OVERLAP, 0), max(x0 - OVERLAP, 0)
            yb, xb = min(y1 + OVERLAP, h), min(x1 + OVERLAP, w)
            big = _upscale_tile(model, device, image[ya:yb, xa:xb])
            small = F.avg_pool2d(torch.from_numpy(big.transpose(2, 0, 1))[None], scale)[0].numpy().transpose(1, 2, 0)
            out[y0:y1, x0:x1] = small[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    return out


def process(image, depth, params):
    strength = float(params.get("strength", 1.0))
    if strength <= 0:
        return image
    tile = int(max(64, float(params.get("tile", 256))))
    key = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
    if _last.get("key") != key:                 # el resultado del modelo no depende de la fuerza ni del tamaño del mosaico
        _last.update(key=key, restored=_restore(image, tile))
    restored = _last["restored"]
    return np.clip(image + strength * (restored - image), 0.0, 1.0).astype(np.float32)
