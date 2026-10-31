"""Mapas auxiliares por foto (sin interfaz): imágenes en gris 0..1 que dicen DÓNDE actúa un efecto del paso Look.

Una capa de Pre FX o de Post FX puede limitarse a un mapa («Limitar a mapa», `LookLayer.limit = "mask"`) en vez de afectar a
toda la imagen o a un rango de profundidad. Cada mapa tiene un id; el primero es «subject» (el sujeto, aislado con SAM 2.1
Small). `look.apply_look(..., masks={id: array})` y `project.render(..., masks=...)` ya reciben el diccionario.

Un mapa tarda segundos en calcularse, así que NO se calcula en cada cambio de ajustes: se calcula al pedirlo y se guarda en
disco, como el mapa de profundidad (`<datos>/map_cache`, clave = contenido de la foto + reencuadre + mapa de profundidad +
ajustes del productor). Todo en local; el modelo (solo safetensors) lo instala el gestor de modelos.

Aquí vive también la lógica de SAM 2.1 (carga, punto automático, segmentación) que reutiliza el módulo «Quitar fondo».
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication
from scipy.ndimage import gaussian_filter

from deepshot.depth import file_hash
from deepshot.paths import app_data_dir

log = logging.getLogger(__name__)

SUBJECT = "subject"
SAM_MODEL_ID = "sam2-1-hiera-small"
WORK_SIDE = 1024                       # SAM 2 trabaja a 1024 px: no hace falta darle más
MAX_MAP_SIDE = 2048                    # lado largo máximo al que se guarda un mapa (la foto entera no hace falta)
Progress = Callable[[str, float], None]


class MapError(RuntimeError):
    """No se pudo calcular el mapa (p. ej. falta el modelo). El mensaje se puede enseñar al usuario."""


# ---------------------------------------------------------------------------
# SAM 2.1 (llamada según la documentación oficial de transformers: Sam2Model + Sam2Processor)
# ---------------------------------------------------------------------------
_model: dict = {}                      # modelo y procesador cargados (se reutilizan)


def sam_installed() -> bool:
    """True si el modelo SAM 2.1 Small está instalado en la carpeta de modelos."""
    from deepshot.models import ModelManager

    try:
        return ModelManager().snapshot(SAM_MODEL_ID) is not None
    except Exception:                                  # catálogo roto o carpeta ilegible: lo mismo que «no instalado»
        return False


def load_sam():
    """(modelo, procesador, dispositivo). Se carga una vez desde la carpeta de modelos del gestor."""
    if not _model:
        import torch
        from transformers import Sam2Model, Sam2Processor

        from deepshot.device import detect_device
        from deepshot.models import ModelManager

        folder = ModelManager().snapshot(SAM_MODEL_ID)
        if folder is None:
            raise MapError(QCoreApplication.translate(
                "Maps", "Falta el modelo SAM 2.1 Small: instálalo desde el gestor de modelos (Ctrl+M)."))
        device = detect_device("auto")
        dtype = torch.float16 if device.kind == "cuda" else torch.float32     # en MPS y CPU, float32 (más fiable)
        model = Sam2Model.from_pretrained(str(folder), dtype=dtype).to(device.kind).eval()
        _model.update(model=model, processor=Sam2Processor.from_pretrained(str(folder)), device=device.kind)
    return _model["model"], _model["processor"], _model["device"]


def auto_point(depth: np.ndarray) -> tuple[float, float]:
    """El punto más cercano del mapa (0 = cerca) tras suavizarlo: casi siempre cae dentro del sujeto principal.
    Devuelve (x, y) en 0..1."""
    h, w = depth.shape
    smooth = gaussian_filter(depth.astype(np.float32), max(h, w) * 0.02)
    y, x = np.unravel_index(int(np.argmin(smooth)), smooth.shape)
    return (x + 0.5) / w, (y + 0.5) / h


def segment_at(image: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    """Las máscaras suaves (n, alto, ancho) 0..1 del objeto bajo `point` (x, y en 0..1), de la más pequeña (una parte) a la
    mayor (el conjunto): un solo punto es ambiguo y SAM da tres lecturas."""
    import torch
    import torch.nn.functional as F

    model, processor, device = load_sam()
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


# ---------------------------------------------------------------------------
# Productores
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MapKind:
    """Un tipo de mapa auxiliar: su nombre (traducible, contexto «Maps»), el modelo que necesita y cómo se calcula."""

    key: str
    title: str
    model_id: str
    compute: Callable[[np.ndarray, np.ndarray], np.ndarray]       # compute(foto float32 0..1, profundidad) -> mapa 0..1
    version: str = "1"                                           # sube si cambia el resultado: invalida la caché


def _subject(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """El sujeto: el objeto bajo el punto más cercano del mapa de profundidad, en su lectura mayor (el conjunto)."""
    masks = segment_at(image, auto_point(depth))
    return masks[-1].astype(np.float32)


KINDS: dict[str, MapKind] = {k.key: k for k in (
    MapKind(SUBJECT, QT_TRANSLATE_NOOP("Maps", "Sujeto"), SAM_MODEL_ID, _subject),
)}


def title(map_id: str) -> str:
    """Nombre visible (traducido) del mapa; el propio id si no se conoce."""
    kind = KINDS.get(map_id)
    return QCoreApplication.translate("Maps", kind.title) if kind else map_id


def is_available(map_id: str) -> bool:
    """True si el modelo que necesita el mapa está instalado."""
    kind = KINDS.get(map_id)
    return kind is not None and (kind.model_id != SAM_MODEL_ID or sam_installed())


# ---------------------------------------------------------------------------
# Caché en disco
# ---------------------------------------------------------------------------
def cache_dir() -> Path:
    path = app_data_dir() / "map_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def depth_digest(depth: np.ndarray) -> str:
    """Huella corta del mapa de profundidad (el sujeto sale del punto más cercano, así que depende de él)."""
    small = np.asarray(Image.fromarray(np.clip(depth, 0, 1).astype(np.float32), mode="F").resize((64, 64), Image.BILINEAR))
    return hashlib.sha256(np.round(small * 255).astype(np.uint8).tobytes()).hexdigest()[:16]


def cache_path(map_id: str, photo: Path, depth: np.ndarray, crop: tuple | None = None,
               photo_hash: str | None = None) -> Path:
    """Archivo de caché del mapa de esta foto: contenido de la foto + reencuadre + profundidad + productor."""
    kind = KINDS[map_id]
    text = "|".join((photo_hash or file_hash(Path(photo)), repr(crop), depth_digest(depth), kind.key, kind.version))
    return cache_dir() / f"{map_id}-{hashlib.sha256(text.encode()).hexdigest()[:32]}.png"


def save_map(mask: np.ndarray, path: Path) -> None:
    """Guarda el mapa como PNG gris de 16 bits (escritura atómica: archivo temporal + renombrar)."""
    data = np.round(np.clip(mask, 0.0, 1.0) * 65535.0).astype(np.uint16)
    tmp = path.with_suffix(".tmp")
    Image.fromarray(data).save(tmp, format="PNG")
    tmp.replace(path)


def load_map(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    return (arr.astype(np.float32) / scale).clip(0.0, 1.0)


def cached(map_id: str, photo: Path, depth: np.ndarray, crop: tuple | None = None) -> np.ndarray | None:
    """El mapa ya calculado de esta foto, o None. Barato: no carga ningún modelo."""
    try:
        path = cache_path(map_id, photo, depth, crop)
        return load_map(path) if path.is_file() else None
    except (OSError, ValueError, KeyError):
        return None


def clear_cache() -> tuple[int, int]:
    """Borra la caché de mapas (se rehace sola al pedirlos). Devuelve (archivos, bytes)."""
    count = freed = 0
    for f in cache_dir().glob("*.png"):
        try:
            size = f.stat().st_size
            f.unlink()
        except OSError:
            continue
        count, freed = count + 1, freed + size
    return count, freed


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
def compute(map_id: str, photo_path: Path, depth: np.ndarray, crop: tuple | None = None,
            progress: Progress | None = None) -> np.ndarray:
    """El mapa de esta foto: de la caché si está; si no, se calcula (en el hilo que llama) y se guarda.

    Devuelve float32 0..1 de alto × ancho ≤ MAX_MAP_SIDE; quien lo use lo reescala al tamaño de su imagen.
    Lanza `MapError` si falta el modelo."""
    from deepshot.depth import load_photo

    if map_id not in KINDS:
        raise MapError(QCoreApplication.translate("Maps", "Mapa desconocido: {0}").format(map_id))
    report = progress or (lambda _m, _f: None)
    path = cache_path(map_id, photo_path, depth, crop)
    if path.is_file():
        try:
            return load_map(path)
        except OSError:
            log.warning("Mapa en caché ilegible, se calcula otra vez: %s", path)
    kind = KINDS[map_id]
    if not is_available(map_id):
        raise MapError(QCoreApplication.translate(
            "Maps", "Falta el modelo SAM 2.1 Small: instálalo desde el gestor de modelos (Ctrl+M)."))
    report(QCoreApplication.translate("Maps", "Cargando la fotografía…"), 0.1)
    photo = load_photo(Path(photo_path), crop)
    scale = min(1.0, MAX_MAP_SIDE / max(photo.size))
    if scale < 1.0:
        photo = photo.resize((max(1, round(photo.width * scale)), max(1, round(photo.height * scale))), Image.LANCZOS)
    image = np.asarray(photo.convert("RGB"), dtype=np.float32) / 255.0
    h, w = image.shape[:2]
    small_depth = np.asarray(Image.fromarray(np.clip(depth, 0, 1).astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))
    report(QCoreApplication.translate("Maps", "Calculando «{0}»…").format(title(map_id)), 0.3)
    mask = np.clip(kind.compute(image, small_depth), 0.0, 1.0).astype(np.float32)
    report(QCoreApplication.translate("Maps", "Guardando el mapa…"), 0.95)
    try:
        save_map(mask, path)
    except OSError:
        log.warning("No se pudo guardar el mapa en la caché: %s", path)
    report(QCoreApplication.translate("Maps", "Listo"), 1.0)
    return mask


def resize(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Reescala un mapa a `size` = (ancho, alto), bilineal."""
    h, w = mask.shape
    if (w, h) == tuple(size):
        return mask
    return np.clip(np.asarray(Image.fromarray(mask.astype(np.float32), mode="F").resize(tuple(size), Image.BILINEAR)),
                   0.0, 1.0).astype(np.float32)
