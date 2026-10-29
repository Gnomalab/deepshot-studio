"""Estimación de profundidad: fotografía -> mapa de profundidad relativo (0..1).

Modelo por defecto: Marigold V1.0 (prs-eth/marigold-depth-v1-0, Apache 2.0) con el
pipeline oficial de diffusers (`MarigoldDepthPipeline`). Los modelos disponibles, su
descarga e instalación los gestiona `deepshot/models.py`. Marigold V2 no cabe en
8 GB de VRAM (ver ROADMAP.md).

Convención de la app (igual que la salida de Marigold):
    0.0 = cerca, 1.0 = lejos.  Profundidad RELATIVA (afín), no métrica.
Para mostrar el mapa "cerca = claro" basta con dibujar 1 - depth.

Caché:
    - Pesos del modelo: <app_data>/models (ver models.py).
    - Mapas: <app_data>/depth_cache/<hash>.png en PNG de 16 bits (0..65535).
      La clave incluye el contenido de la foto, el modelo y los ajustes.
"""

from __future__ import annotations

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from deepshot.models import DEFAULT_MODEL, ModelError, ModelManager, ProgressCallback, _no_progress
from deepshot.paths import app_data_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ajustes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepthSettings:
    """Parámetros de la inferencia. None = valor por defecto del modelo."""

    model: str = DEFAULT_MODEL                # id del modelo en el gestor (models.py)
    steps: int | None = None                 # V1.0: 10 pasos DDIM por defecto
    ensemble_size: int = 1                    # >= 3 promedia varias predicciones (más lento, más fino)
    processing_resolution: int | None = None  # V1.0: 768 px por defecto en el lado mayor
    seed: int = 2024                          # semilla fija: mismo resultado en cada cálculo
    refine: float = 0.0                       # refinado guiado por la foto: 0 = no, 1 = completo

    def key(self) -> str:
        """Texto estable para la clave de caché (incluye el refinado)."""
        return json.dumps(asdict(self), sort_keys=True)

    def raw_key(self) -> str:
        """Clave de la salida del modelo sin refinar (el refinado se aplica después)."""
        return replace(self, refine=0.0).key()


DA_DEFAULT_RESOLUTION = 756       # Depth Anything se entrenó a 518 px; algo más da más detalle sin romper

# Presets de calidad, medidos en una RTX 4060 Laptop (8 GB) con una foto de 6000×4000
# (2026-10-18). El tiempo no incluye la carga del modelo (~11 s, una vez por sesión).
#   768 px ×1: 3,3 s, 3,6 GB  ·  1024 ×1: 4,1 s, 4,3 GB  ·  1280 ×1: 6,2 s, 5,3 GB
#   1536 ×1: 13 s, 6,4 GB     ·  1280 ×3: 12 s, 5,3 GB   ·  1536 ×3: 36 s, 6,4 GB
#   2048 ×1: 48 s, 9,4 GB (no cabe en 8 GB: usa memoria compartida y se vuelve lento)
@dataclass(frozen=True)
class QualityPreset:
    id: str
    label: str
    processing_resolution: int
    ensemble_size: int
    hint: str


QUALITY_PRESETS: dict[str, QualityPreset] = {
    "fast": QualityPreset("fast", QT_TRANSLATE_NOOP("Depth", "Rápida"), 768, 1,
                          QT_TRANSLATE_NOOP("Depth", "768 px · ~3 s · 3,6 GB de VRAM")),
    "high": QualityPreset("high", QT_TRANSLATE_NOOP("Depth", "Alta"), 1280, 1,
                          QT_TRANSLATE_NOOP("Depth", "1280 px · ~6 s · 5,3 GB de VRAM")),
    "max": QualityPreset("max", QT_TRANSLATE_NOOP("Depth", "Máxima"), 1280, 3,
                         QT_TRANSLATE_NOOP("Depth", "1280 px × 3 pasadas promediadas · ~12 s · 5,3 GB")),
}
DEFAULT_QUALITY = "fast"


# Resoluciones máximas elegibles a mano (lado mayor, en píxeles); None = la del preset de calidad.
RESOLUTION_CHOICES = (256, 384, 512, 768, 1024, 1280, 1600)


def settings_for(model: str = DEFAULT_MODEL, quality: str = DEFAULT_QUALITY, refine: float = 0.0,
                 resolution: int | None = None) -> DepthSettings:
    """Ajustes de inferencia a partir del modelo, el preset de calidad, el refinado y, si se pide, la resolución."""
    q = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])
    return DepthSettings(model=model, ensemble_size=q.ensemble_size,
                         processing_resolution=resolution or q.processing_resolution, refine=refine)


# ---------------------------------------------------------------------------
# Utilidades de imagen y caché
# ---------------------------------------------------------------------------
def normalize_depth(raw: np.ndarray, metric: bool) -> np.ndarray:
    """Lleva la salida de un modelo a la convención de la app: float32 0..1 con 0 = cerca.

    metric=False: profundidad inversa relativa (MiDaS, Depth Anything): más grande = más cerca.
    metric=True: profundidad en metros (ZoeDepth, Depth Pro): más grande = más lejos; se pasa a inversa (1 / metros)
    antes de normalizar, para que lo cercano tenga el contraste que tiene en el resto de modelos."""
    raw = np.squeeze(np.asarray(raw, dtype=np.float32))
    inverse = 1.0 / np.clip(raw, 1e-3, None) if metric else raw
    lo, hi = float(inverse.min()), float(inverse.max())
    if hi <= lo:
        return np.zeros_like(inverse, dtype=np.float32)
    return np.clip(1.0 - (inverse - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_photo(path: Path) -> Image.Image:
    """Abre una foto en RGB aplicando la orientación EXIF (igual que el lienzo)."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def file_hash(path: Path) -> str:
    """SHA-256 del contenido: la caché sigue valiendo si se renombra o mueve la foto."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir() -> Path:
    path = app_data_dir() / "depth_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_files() -> list[Path]:
    """Los archivos de la caché: PNG de mapas y JSON de tiempos, solo los que están directamente en su carpeta."""
    return [f for f in cache_dir().iterdir() if f.is_file() and f.suffix.lower() in (".png", ".json")]


def cache_size() -> tuple[int, int]:
    """(archivos, bytes) que ocupa la caché de mapas de profundidad."""
    files = _cache_files()
    return len(files), sum(f.stat().st_size for f in files)


def clear_cache() -> tuple[int, int]:
    """Borra la caché de mapas (se rehace sola al calcular otra vez). Devuelve (archivos borrados, bytes liberados).
    No toca nada fuera de la carpeta de la caché ni los proyectos guardados: llevan sus propios mapas."""
    count = freed = 0
    for f in _cache_files():
        try:
            size = f.stat().st_size
            f.unlink()
        except OSError:                           # un archivo en uso se queda; el resto se borra
            continue
        count, freed = count + 1, freed + size
    return count, freed


def cache_path(image_path: Path, settings: DepthSettings, raw: bool = False, photo_hash: str | None = None) -> Path:
    """Archivo de caché del mapa. raw=True: salida del modelo sin refinar.

    photo_hash: huella de la foto si ya se calculó (evita leerla entera otra vez al mirar muchas combinaciones)."""
    text = settings.raw_key() if raw else settings.key()
    key = hashlib.sha256(((photo_hash or file_hash(image_path)) + text).encode()).hexdigest()[:32]
    return cache_dir() / f"{key}.png"


# Tiempos de cada cálculo: junto a cada mapa en caché va un .json con lo que tardó, para poder enseñarlo
# (y compararlo) aunque el mapa se lea de la caché en otra sesión.
@dataclass
class DepthRun:
    """Datos de un cálculo de profundidad: con qué ajustes y cuánto tardó."""

    model: str
    processing_resolution: int | None
    ensemble_size: int
    refine: float
    infer_seconds: float | None = None      # solo el modelo (sin cargarlo); None si se desconoce
    refine_seconds: float = 0.0             # el refinado con la foto
    load_seconds: float = 0.0               # cargar el modelo en la GPU (0 si ya estaba cargado)
    from_cache: bool = False                # el mapa salió de la caché: los tiempos son los del cálculo original
    device: str = ""

    @property
    def seconds(self) -> float | None:
        """Tiempo de procesar la foto (modelo + refinado), sin la carga del modelo."""
        return None if self.infer_seconds is None else self.infer_seconds + self.refine_seconds


def timing_path(png: Path) -> Path:
    return png.with_suffix(".json")


def read_timing(png: Path) -> dict:
    """Tiempos guardados junto a un mapa de la caché ({} si no hay o no se puede leer)."""
    try:
        data = json.loads(timing_path(png).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_timing(png: Path, data: dict) -> None:
    try:
        timing_path(png).write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:                  # un tiempo que no se guarda no debe romper el cálculo
        log.warning("No se pudo guardar el tiempo de %s: %s", png.name, exc)


def save_depth_png16(depth: np.ndarray, path: Path) -> None:
    """Guarda un mapa 0..1 como PNG de 16 bits en escala de grises."""
    data = np.round(np.clip(depth, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(data).save(path)


def load_depth_png(path: Path) -> np.ndarray:
    """Lee un PNG de profundidad (16 u 8 bits) y lo devuelve en float32 0..1."""
    with Image.open(path) as img:
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            data = np.asarray(img, dtype=np.float32)
            max_value = 65535.0
        else:
            data = np.asarray(img.convert("L"), dtype=np.float32)
            max_value = 255.0
    return np.clip(data / max_value, 0.0, 1.0)


def load_external_depth(path: Path, near_is_bright: bool = True) -> np.ndarray:
    """Carga un mapa de profundidad hecho fuera de la app (p. ej. Marigold V2 en ComfyUI).

    Se reescala a 0..1 con su mínimo y máximo reales y se pasa a la convención de la
    app (0 = cerca). Muchos programas guardan "cerca = claro": en ese caso se invierte.
    Por ahora: PNG/TIFF de 8 o 16 bits. EXR llegará en la fase 5.
    """
    depth = load_depth_png(path)
    lo, hi = float(depth.min()), float(depth.max())
    if hi > lo:
        depth = (depth - lo) / (hi - lo)
    return 1.0 - depth if near_is_bright else depth


# ---------------------------------------------------------------------------
# Estimador
# ---------------------------------------------------------------------------
class DepthEstimator:
    """Carga y ejecuta el modelo elegido en el gestor. Mantiene en memoria el último usado."""

    def __init__(self, device: str = "auto", manager: ModelManager | None = None) -> None:
        self.preferred_device = device
        self.manager = manager or ModelManager()
        self._pipe = None
        self._processor = None                 # procesador de imágenes de los modelos de transformers
        self._load_lock = threading.RLock()    # la carga al arrancar y un cálculo no cargan a la vez
        self._family = ""                      # familia del modelo cargado (cómo se le llama)
        self._loaded_model: str | None = None
        self._device = None
        self.last_run: DepthRun | None = None            # datos del último `estimate`
        self._timing = (0.0, 0.0)                        # (carga del modelo, inferencia) del último `_infer`

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    def load(self, model: str = DEFAULT_MODEL, progress: ProgressCallback = _no_progress) -> None:
        """Carga el modelo (ver `_load`). Si otro hilo lo está cargando, espera a que termine."""
        with self._load_lock:
            self._load(model, progress)

    def _load(self, model: str, progress: ProgressCallback) -> None:
        """Carga el modelo en la GPU (o CPU). Si es del catálogo y falta, lo descarga."""
        if self._pipe is not None and self._loaded_model == model:
            return
        self.unload()  # un solo modelo en memoria a la vez (8 GB de VRAM)
        entry = self.manager.get(model)
        snapshot = self.manager.snapshot(model)
        if snapshot is None:
            if entry.source in ("local", "folder"):
                raise ModelError(QCoreApplication.translate("Depth", "No se encuentran los archivos de «{0}». Reinstálalo desde el gestor.").format(entry.display_label))
            if entry.warning or entry.dev_only:       # modelos enormes o de prueba: nunca se descargan solos al calcular
                raise ModelError(QCoreApplication.translate("Depth", "«{0}» no está descargado. Descárgalo desde el gestor de modelos (ocupa mucho).").format(entry.display_label))
            snapshot = self.manager.download(model, progress)

        import torch

        from deepshot.device import detect_device

        device = detect_device(self.preferred_device)
        # fp16 en GPU (la mitad de memoria); fp32 en CPU, donde fp16 es lento o no está soportado.
        dtype = torch.float32 if device.kind == "cpu" else torch.float16
        progress(QCoreApplication.translate("Depth", "Cargando {0} en {1}…").format(entry.display_label, device.name), None)

        if entry.family == "marigold-diffusers":
            from diffusers import MarigoldDepthPipeline

            pipe = MarigoldDepthPipeline.from_pretrained(
                snapshot, variant=self.manager.variant(snapshot), torch_dtype=dtype, use_safetensors=True
            )
            pipe.set_progress_bar_config(disable=True)
        elif entry.family == "depth-anything":
            from transformers import AutoModelForDepthEstimation

            pipe = AutoModelForDepthEstimation.from_pretrained(snapshot, torch_dtype=dtype, use_safetensors=True)
            pipe.eval()
        elif entry.family == "transformers-depth":
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            try:
                self._processor = AutoImageProcessor.from_pretrained(snapshot)
            except ImportError as exc:      # Depth Pro pide torchvision
                raise ModelError(QCoreApplication.translate("Depth", "Falta una biblioteca para cargar este modelo: {0}").format(exc)) from exc
            pipe = AutoModelForDepthEstimation.from_pretrained(snapshot, torch_dtype=dtype, use_safetensors=True)
            pipe.eval()
        elif entry.family == "marigold-v2":
            # TODO: la inferencia de Marigold V2 aún no está escrita (solo la descarga y la lista).
            raise ModelError(QCoreApplication.translate("Depth", "El cálculo con Marigold V2 aún no está implementado."))
        else:
            raise ModelError(QCoreApplication.translate("Depth", "La familia «{0}» no está soportada en esta versión.").format(entry.family))

        self._pipe = pipe.to(device.kind)
        self._family = entry.family
        self._loaded_model = model
        self._device = device
        log.info("%s cargado en %s (%s)", entry.display_label, device.name, dtype)

    def unload(self) -> None:
        """Libera la memoria de vídeo."""
        self._pipe = None
        self._processor = None
        self._loaded_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- Inferencia ------------------------------------------------------------
    def estimate(
        self,
        image_path: Path,
        settings: DepthSettings = DepthSettings(),
        progress: ProgressCallback = _no_progress,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Devuelve la profundidad de la foto: float32 [alto, ancho], 0 = cerca, 1 = lejos.

        Si el mapa ya está en caché, no se carga el modelo.
        """
        image_path = Path(image_path)
        digest = file_hash(image_path)
        cached = cache_path(image_path, settings, photo_hash=digest)

        def run(**kwargs) -> DepthRun:
            return DepthRun(settings.model, settings.processing_resolution, settings.ensemble_size, settings.refine,
                            device=self._device.name if self._device is not None else "", **kwargs)

        if use_cache and cached.is_file():
            log.info("Profundidad leída de caché: %s", image_path.name)
            t = read_timing(cached)
            self.last_run = run(infer_seconds=t.get("infer_seconds"), refine_seconds=float(t.get("refine_seconds", 0.0)),
                                from_cache=True)
            return load_depth_png(cached)

        # Salida del modelo: de caché si existe (cambiar solo el refinado no recalcula).
        raw_cached = cache_path(image_path, settings, raw=True, photo_hash=digest)
        self._timing = (0.0, 0.0)
        raw_from_cache = bool(use_cache and raw_cached.is_file())
        if raw_from_cache:
            depth = load_depth_png(raw_cached)
            infer = read_timing(raw_cached).get("infer_seconds")
            load_s = 0.0
        else:
            depth = self._infer(image_path, settings, progress)
            load_s, infer = self._timing
            save_depth_png16(depth, raw_cached)
            write_timing(raw_cached, {"infer_seconds": infer, "model": settings.model,
                                      "device": self._device.name if self._device is not None else ""})

        refine_s = 0.0
        if settings.refine > 0:
            from deepshot.refine import refine_depth

            progress(QCoreApplication.translate("Depth", "Refinando bordes con la foto…"), None)
            started = time.perf_counter()
            depth = refine_depth(depth, np.asarray(load_photo(image_path)), strength=settings.refine)
            refine_s = time.perf_counter() - started
            save_depth_png16(depth, cached)
            write_timing(cached, {"infer_seconds": infer, "refine_seconds": refine_s})
        self.last_run = run(infer_seconds=infer, refine_seconds=refine_s, load_seconds=load_s,
                            from_cache=raw_from_cache)
        return depth

    def _infer(self, image_path: Path, settings: DepthSettings, progress: ProgressCallback) -> np.ndarray:
        """Ejecuta el modelo sobre la foto (sin caché).

        Si la GPU se queda sin memoria (8 GB justos con fotos grandes u otros programas abiertos),
        libera la caché y reintenta con la mitad de resolución en vez de fallar.
        """
        loading = time.perf_counter()
        self.load(settings.model, progress)
        load_seconds = time.perf_counter() - loading if self._pipe is not None else 0.0
        import torch

        if settings.ensemble_size > 1:
            progress(QCoreApplication.translate("Depth", "Calculando profundidad ({0} pasadas)…").format(settings.ensemble_size), None)
        else:
            progress(QCoreApplication.translate("Depth", "Calculando profundidad…"), None)
        image = load_photo(image_path)
        resolution, ensemble = settings.processing_resolution, settings.ensemble_size
        inferring = time.perf_counter()
        try:
            depth = self._run_pipe(image, settings, resolution, ensemble)
        except torch.cuda.OutOfMemoryError:
            self._free_gpu_memory()
            resolution = max(512, (resolution or 768) // 2)
            log.warning("Sin memoria de vídeo: se reintenta a %d px y una sola pasada.", resolution)
            progress(QCoreApplication.translate("Depth", "Poca memoria de vídeo: reintentando a menor resolución…"), None)
            depth = self._run_pipe(image, settings, resolution, 1)
        finally:
            self._free_gpu_memory()   # devuelve la VRAM sobrante al sistema entre cálculos
            self._timing = (load_seconds if load_seconds > 0.05 else 0.0, time.perf_counter() - inferring)
        log.info("Profundidad calculada: %s (%d×%d, %s px, ×%d)", image_path.name, depth.shape[1],
                 depth.shape[0], resolution or "por defecto", ensemble)
        return depth

    def _run_pipe(self, image: Image.Image, settings: DepthSettings, resolution: int | None,
                  ensemble: int) -> np.ndarray:
        if self._family == "depth-anything":
            return self._run_depth_anything(image, resolution)
        if self._family == "transformers-depth":
            return self._run_transformers_depth(image)
        import torch

        # Generador en CPU: válido para CUDA, MPS y CPU, y reproducible entre equipos.
        generator = torch.Generator(device="cpu").manual_seed(settings.seed)
        with torch.inference_mode():
            out = self._pipe(
                image,
                num_inference_steps=settings.steps,
                ensemble_size=ensemble,
                processing_resolution=resolution,
                match_input_resolution=True,  # el mapa sale al tamaño de la foto
                generator=generator,
                output_type="np",
            )
        return np.clip(np.squeeze(np.asarray(out.prediction, dtype=np.float32)), 0.0, 1.0)

    def _run_depth_anything(self, image: Image.Image, resolution: int | None) -> np.ndarray:
        """Depth Anything V2: una sola pasada sobre la foto reducida al lado mayor `resolution`.

        La red da profundidad inversa relativa (más grande = más cerca): se normaliza a 0..1 y se invierte
        para la convención de la app (0 = cerca). La entrada va en múltiplos de 14 (tamaño de parche)
        y se normaliza con la media y desviación de ImageNet, como pide el modelo oficial.
        """
        import torch
        import torch.nn.functional as F

        w, h = image.size
        scale = (resolution or DA_DEFAULT_RESOLUTION) / max(w, h)
        iw, ih = (max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14))
        small = np.asarray(image.resize((iw, ih), Image.BICUBIC), dtype=np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
        tensor = torch.from_numpy(((small - mean) / std).transpose(2, 0, 1)[None])
        param = next(self._pipe.parameters())
        with torch.inference_mode():
            pred = self._pipe(pixel_values=tensor.to(param.device, param.dtype)).predicted_depth
            pred = F.interpolate(pred[:, None].float(), size=(h, w), mode="bicubic", align_corners=False)
        inv = pred[0, 0].cpu().numpy()
        lo, hi = float(inv.min()), float(inv.max())
        inv = (inv - lo) / (hi - lo) if hi > lo else np.zeros_like(inv)
        return np.clip(1.0 - inv, 0.0, 1.0).astype(np.float32)

    def _run_transformers_depth(self, image: Image.Image) -> np.ndarray:
        """MiDaS (DPT), ZoeDepth y Depth Pro: una pasada con el procesador de imágenes propio de cada modelo.

        El procesador decide el tamaño de entrada (512 para MiDaS BEiT, 384×512 para ZoeDepth, 1536 para Depth Pro), así
        que la resolución de calidad no se aplica. MiDaS da profundidad inversa relativa (más grande = más cerca);
        ZoeDepth y Depth Pro dan profundidad métrica en metros (más grande = más lejos). Las dos se llevan a «0 = cerca».
        """
        import inspect

        import torch

        w, h = image.size
        param = next(self._pipe.parameters())
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: (v.to(param.device, param.dtype) if v.is_floating_point() else v.to(param.device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._pipe(**inputs)
            wanted = inspect.signature(self._processor.post_process_depth_estimation).parameters
            kwargs = {"target_sizes": [(h, w)]}
            if "source_sizes" in wanted:            # ZoeDepth quita el relleno que le añadió el procesador
                kwargs["source_sizes"] = [(h, w)]
            depth = self._processor.post_process_depth_estimation(outputs, **kwargs)[0]["predicted_depth"]
        metric = getattr(self._pipe.config, "model_type", "") in ("zoedepth", "depth_pro")
        return normalize_depth(depth.float().cpu().numpy(), metric=metric)

    @staticmethod
    def _free_gpu_memory() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
