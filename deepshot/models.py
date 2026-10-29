"""Gestor de modelos de profundidad: catálogo, descarga, instalación desde carpeta y borrado.

Dos capas (ROADMAP.md, decisión 9):
  - FAMILIA: tipo de modelo con código de inferencia en la app. Hoy solo
    "marigold-diffusers" (cualquier checkpoint para `MarigoldDepthPipeline`).
    Una familia nueva (p. ej. Marigold V2) exige añadir código en `depth.py`.
  - MODELO: un checkpoint concreto de una familia. Puede venir de:
      "catalog"      catálogo integrado (MODELS_CATALOG), se descarga de Hugging Face;
      "huggingface"  repo añadido por el usuario por su nombre (org/modelo);
      "local"        carpeta del disco, copiada a la carpeta de datos de la app.

Seguridad: solo se copian/descargan archivos de datos (.json, .txt, .safetensors…).
Los pesos en formato pickle (.bin, .pt, .pth, .ckpt, .pkl) se ignoran: pueden
ejecutar código al cargarse.

Carpeta de modelos (`resolve_models_dir`): por defecto `models/` junto al programa, fácil de encontrar; se puede
cambiar en el gestor de modelos (ajuste «models/folder») o con la variable DEEPSHOT_MODELS_DIR. Si la carpeta del
programa no admite escritura (instalado en «Archivos de programa»), se usa la de datos de la aplicación.

Estructura en disco (<carpeta de modelos>):
    installed.json                 modelos añadidos por el usuario
    <id>.complete.json             marcador de descarga completa (catálogo y HF)
    models--org--nombre/...        caché de Hugging Face
    local/<id>/...                 modelos instalados desde carpeta
    <cualquier carpeta>/           modelo soltado a mano (config.json o model_index.json + .safetensors): aparece solo
                                   en la lista al actualizarla (origen «folder»)
"""

from __future__ import annotations

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication

import io
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from deepshot.paths import app_data_dir

# Sin telemetría y sin avisos de symlinks. Debe fijarse antes de importar huggingface_hub. La caché xet va dentro de la
# carpeta de modelos elegida (se fija al crear el ModelManager), salvo que el usuario ya haya definido HF_XET_CACHE.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_XET_SET_BY_USER = "HF_XET_CACHE" in os.environ

log = logging.getLogger(__name__)

# progress(mensaje, fracción 0..1 o None si no se puede medir)
ProgressCallback = Callable[[str, "float | None"], None]


def _no_progress(_msg: str, _frac: float | None) -> None:
    pass


SAFE_EXTENSIONS = {".json", ".txt", ".safetensors", ".model"}  # .model = tokenizer sentencepiece
UNSAFE_EXTENSIONS = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
# Licencias aceptadas por la política del proyecto (CLAUDE.md). Otras piden confirmación.
PERMISSIVE_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "bsd", "hpnd"}


MARIGOLD_V2_LORA = "depth/Log-stage2/trainables.safetensors"
MARIGOLD_V2_BASE = "Qwen/Qwen-Image-Edit-2509"
MARIGOLD_V2_EMBEDS = "qwen_text_embeddings/qwen_edit_2509_qwen_depth_realimg512_prompt_{0}.pt"


@dataclass(frozen=True)
class Family:
    id: str
    label: str
    pipeline_class: str  # valor de "_class_name" en model_index.json ("" si no es un modelo diffusers)
    index_file: str = "model_index.json"   # archivo que marca la carpeta de un modelo de esta familia


# Familias que se pueden añadir por nombre o desde carpeta (formato diffusers).
FAMILIES: dict[str, Family] = {
    "marigold-diffusers": Family("marigold-diffusers", "Marigold (diffusers)", "MarigoldDepthPipeline"),
}
# Todas las familias con código de inferencia. Depth Anything solo viene del catálogo: es un modelo
# de transformers (config.json + model.safetensors), no tiene model_index.json.
ALL_FAMILIES: dict[str, Family] = {
    **FAMILIES,
    "depth-anything": Family("depth-anything", "Depth Anything V2 (transformers)", "", "config.json"),
    # Marigold V2: LoRA sobre Qwen-Image-Edit-2509. Dos repos (los adaptadores y el modelo base); el archivo que
    # marca la instalación es el adaptador de profundidad.
    "marigold-v2": Family("marigold-v2", "Marigold V2 (Qwen-Image-Edit)", "", MARIGOLD_V2_LORA),
    # Modelos de profundidad de transformers que traen su propio procesador de imágenes: MiDaS (DPT), ZoeDepth y
    # Depth Pro. Se cargan con AutoModelForDepthEstimation; el tipo de salida se lee de su configuración.
    "transformers-depth": Family("transformers-depth", "MiDaS / ZoeDepth / Depth Pro (transformers)", "", "config.json"),
}


@dataclass
class ModelEntry:
    id: str
    label: str
    family: str
    source: str                     # "catalog" | "huggingface" | "local" | "folder" (soltado a mano en la carpeta)
    repo_id: str | None = None      # para "catalog" y "huggingface"
    license: str = "desconocida"
    patterns: list[str] = field(default_factory=list)  # archivos a descargar (vacío = automático)
    path: str | None = None         # para "local" y "folder": carpeta relativa a models_dir
    approx_gb: float | None = None
    # Modelos que necesitan más de un repositorio (p. ej. Marigold V2: adaptadores + modelo base):
    # {repo: patrones}. Se descargan junto con el principal y se borran con él.
    extra_repos: dict[str, list[str]] = field(default_factory=dict)
    # Archivos .pt (pickle) de la lista de patrones que se aceptan: solo tensores del autor original, que se
    # leen con torch.load(weights_only=True) (no puede ejecutar código). El resto de .pt sigue prohibido.
    tensor_files: list[str] = field(default_factory=list)
    warning: str = ""               # aviso de requisitos que se enseña antes de descargar y en la lista
    # Modelo de prueba: licencia no comercial o modelo más antiguo, para comparar calidad internamente. Se ve con el
    # interruptor «Mostrar modelos de prueba» de Preferencias; en la versión que se distribuya, apagado por defecto.
    dev_only: bool = False
    min_vram_gb: float | None = None   # VRAM que piden sus autores; se compara con la GPU detectada (solo aviso)
    # Tipo de entrada: "depth" (modelo de profundidad, lo de siempre) o un tipo de módulo (deepshot/modules.py: "look").
    # Los módulos no se descargan de Hugging Face: se copian del catálogo local a la carpeta de módulos.
    kind: str = "depth"
    description: str = ""           # módulos: qué hacen (viene de su module.json)
    version: str = ""
    requires: list[str] = field(default_factory=list)   # módulos: ids de modelos que necesitan, del propio gestor
    url: str = ""                   # módulos: su página web (enlace de la ficha en el gestor)

    @property
    def is_module(self) -> bool:
        return self.kind != "depth"

    @property
    def display_label(self) -> str:
        """Nombre para mostrar: el de los modelos del catálogo se traduce; el de los del usuario va tal cual."""
        return QCoreApplication.translate("Models", self.label) if self.source == "catalog" else self.label

    @property
    def display_warning(self) -> str:
        return QCoreApplication.translate("Models", self.warning) if self.warning else ""

    @property
    def removable(self) -> bool:
        """Las entradas del catálogo y los módulos no se quitan de la lista (solo se desinstalan)."""
        return self.source != "catalog" and not self.is_module


# Archivos de un checkpoint Marigold V1 en diffusers: configuraciones + pesos fp16 en safetensors.
_MARIGOLD_V1_FILES = [
    "model_index.json",
    "scheduler/*.json",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]

_NON_COMMERCIAL_WARNING = QT_TRANSLATE_NOOP(
    "Models", "Licencia CC-BY-NC 4.0: este modelo es para investigación y uso personal, NO para uso comercial. "
              "Úsalo para probar y comparar la calidad; revisa su licencia antes de usar sus resultados en un trabajo "
              "comercial. Solo se descarga si lo pides.")

MODELS_CATALOG: list[ModelEntry] = [
    ModelEntry(
        id="marigold-v1-0",
        label="Marigold V1.0",
        family="marigold-diffusers",
        source="catalog",
        repo_id="prs-eth/marigold-depth-v1-0",
        license="apache-2.0",
        patterns=list(_MARIGOLD_V1_FILES),
        approx_gb=2.6,
        min_vram_gb=4.0,       # medido en la RTX 4060: 3,6 GB de VRAM
    ),
    # Versión rápida (Latent Consistency Model): 1 paso por defecto en vez de 10.
    ModelEntry(
        id="marigold-lcm-v1-0",
        label=QT_TRANSLATE_NOOP("Models", "Marigold LCM V1.0 (rápido)"),
        family="marigold-diffusers",
        source="catalog",
        repo_id="prs-eth/marigold-depth-lcm-v1-0",
        license="apache-2.0",
        patterns=list(_MARIGOLD_V1_FILES),
        approx_gb=2.6,
        min_vram_gb=4.0,       # mismo tamaño que Marigold V1.0
    ),
    # Depth Anything V2 Small: una sola pasada (décimas de segundo), 99 MB, Apache 2.0. Menos detalle fino que
    # Marigold. Base y Large son CC-BY-NC (no comercial): no están en el catálogo.
    ModelEntry(
        id="depth-anything-v2-small",
        label=QT_TRANSLATE_NOOP("Models", "Depth Anything V2 Small (muy rápido)"),
        family="depth-anything",
        source="catalog",
        repo_id="depth-anything/Depth-Anything-V2-Small-hf",
        license="apache-2.0",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        approx_gb=0.1,
    ),
    # Depth Anything V2 Base y Large: más detalle que Small, pero con licencia CC-BY-NC 4.0 (NO comercial). Decisión de
    # Juanjo (2026-10-20): se ofrecen para probar y comparar calidad, con aviso claro y confirmación antes de descargar,
    # y nunca se descargan solos. Antes de vender o distribuir la aplicación hay que revisar esta decisión.
    ModelEntry(
        id="depth-anything-v2-base",
        label=QT_TRANSLATE_NOOP("Models", "Depth Anything V2 Base (solo uso no comercial)"),
        family="depth-anything",
        source="catalog",
        repo_id="depth-anything/Depth-Anything-V2-Base-hf",
        license="cc-by-nc-4.0",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        approx_gb=0.4,
        warning=_NON_COMMERCIAL_WARNING,
        dev_only=True,
    ),
    ModelEntry(
        id="depth-anything-v2-large",
        label=QT_TRANSLATE_NOOP("Models", "Depth Anything V2 Large (solo uso no comercial)"),
        family="depth-anything",
        source="catalog",
        repo_id="depth-anything/Depth-Anything-V2-Large-hf",
        license="cc-by-nc-4.0",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        approx_gb=1.4,
        warning=_NON_COMMERCIAL_WARNING,
        dev_only=True,
    ),
    # Modelos de PRUEBA (solo se ven con «Mostrar modelos de prueba»): MiDaS y ZoeDepth de Intel (MIT / Apache 2.0, más
    # antiguos que Depth Anything V2 y Marigold) y Depth Pro de Apple (licencia de solo investigación). Sirven para comparar
    # calidad; no forman parte de la oferta final. Todos son de transformers, con safetensors.
    ModelEntry(
        id="midas-3-1-beit-large",
        label=QT_TRANSLATE_NOOP("Models", "MiDaS 3.1 BEiT Large 512 (prueba)"),
        family="transformers-depth", source="catalog", repo_id="Intel/dpt-beit-large-512", license="mit",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"], approx_gb=1.4, dev_only=True,
    ),
    ModelEntry(
        id="midas-3-0-dpt-large",
        label=QT_TRANSLATE_NOOP("Models", "MiDaS 3.0 DPT Large (prueba)"),
        family="transformers-depth", source="catalog", repo_id="Intel/dpt-large", license="apache-2.0",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"], approx_gb=1.4, dev_only=True,
    ),
    ModelEntry(
        id="zoedepth-nyu-kitti",
        label=QT_TRANSLATE_NOOP("Models", "ZoeDepth interior y exterior (prueba)"),
        family="transformers-depth", source="catalog", repo_id="Intel/zoedepth-nyu-kitti", license="mit",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"], approx_gb=1.4, dev_only=True,
    ),
    ModelEntry(
        id="zoedepth-nyu",
        label=QT_TRANSLATE_NOOP("Models", "ZoeDepth interior NYU (prueba)"),
        family="transformers-depth", source="catalog", repo_id="Intel/zoedepth-nyu", license="mit",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"], approx_gb=1.4, dev_only=True,
    ),
    ModelEntry(
        id="zoedepth-kitti",
        label=QT_TRANSLATE_NOOP("Models", "ZoeDepth exterior KITTI (prueba)"),
        family="transformers-depth", source="catalog", repo_id="Intel/zoedepth-kitti", license="mit",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"], approx_gb=1.4, dev_only=True,
    ),
    ModelEntry(
        id="depth-pro",
        label=QT_TRANSLATE_NOOP("Models", "Depth Pro de Apple (prueba, solo investigación)"),
        family="transformers-depth", source="catalog", repo_id="apple/DepthPro-hf", license="apple-amlr",
        patterns=["config.json", "preprocessor_config.json", "model.safetensors"], approx_gb=1.9, dev_only=True,
        warning=QT_TRANSLATE_NOOP(
            "Models", "Licencia de investigación de Apple (apple-amlr): este modelo es solo para investigación y "
                      "pruebas, NO para uso comercial. Úsalo para comparar la calidad y revisa su licencia antes de "
                      "usar sus resultados en un trabajo comercial. Solo se descarga si lo pides."),
    ),
    # Marigold V2 (Huawei Bayer Lab, SIGGRAPH Asia 2026): LoRA sobre Qwen-Image-Edit-2509 (20 000 M de parámetros).
    # El más detallado, pero pesado: unos 43 GB de descarga y, según sus autores, ~17 GB de VRAM a 1024².
    # Apache 2.0 los adaptadores y el modelo base. Es la misma variante que usa el add-on de referencia.
    ModelEntry(
        id="marigold-v2-log",
        label=QT_TRANSLATE_NOOP("Models", "Marigold V2 (máxima calidad, exige GPU potente)"),
        family="marigold-v2",
        source="catalog",
        repo_id="huawei-bayerlab/marigold-v2-0",
        license="apache-2.0",
        patterns=[MARIGOLD_V2_LORA, MARIGOLD_V2_EMBEDS.format("embeds"), MARIGOLD_V2_EMBEDS.format("mask")],
        approx_gb=43.0,
        extra_repos={MARIGOLD_V2_BASE: ["transformer/*.json", "transformer/*.safetensors",
                                        "vae/*.json", "vae/*.safetensors"]},
        tensor_files=[MARIGOLD_V2_EMBEDS.format("embeds"), MARIGOLD_V2_EMBEDS.format("mask")],
        min_vram_gb=17.0,
        warning=QT_TRANSLATE_NOOP(
            "Models", "Necesita un equipo potente: unos 43 GB de descarga (y ~55 GB libres durante la "
                      "instalación), una GPU NVIDIA con CUDA y mucha RAM. Sus autores indican ~17 GB de VRAM a "
                      "1024²; con 8 GB solo puede funcionar de forma experimental y lenta. Marigold V1 y "
                      "Depth Anything no tienen estos requisitos."),
    ),
]
DEFAULT_MODEL = "marigold-v1-0"
MODULE_ID_PREFIX = "module:"          # los módulos comparten la lista con los modelos: su id es «module:<id>»
MODULE_FAMILY = "look-module"


def fits_vram(entry: ModelEntry, vram_gb: float | None) -> bool | None:
    """¿Cabe el modelo en la memoria de vídeo del equipo? None si no se sabe (el modelo no declara su mínimo o
    no hay GPU con VRAM medible, p. ej. CPU o Mac, donde la memoria es compartida)."""
    if not entry.min_vram_gb or vram_gb is None:
        return None
    return vram_gb >= entry.min_vram_gb

# Interruptor de los modelos de prueba (Preferencias › General). Por defecto encendido mientras se desarrolla; la
# variable DEEPSHOT_DEV_MODELS (1 / 0) manda sobre el ajuste. Para una versión distribuida, poner DEV_MODELS_DEFAULT = False.
DEV_MODELS_DEFAULT = True


def program_dir() -> Path:
    """La carpeta del programa: junto al ejecutable (PyInstaller) o la raíz del proyecto en desarrollo."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _writable_folder(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".escritura"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def resolve_models_dir() -> Path:
    """Dónde se guardan y se buscan los modelos, por orden: DEEPSHOT_MODELS_DIR; la carpeta de datos forzada con
    DEEPSHOT_DATA_DIR (pruebas y uso portátil); la elegida en el gestor de modelos; `models/` junto al programa; y, si
    ahí no se puede escribir, la carpeta de datos de la aplicación."""
    forced = os.environ.get("DEEPSHOT_MODELS_DIR", "").strip()
    if forced:
        return Path(forced)
    if os.environ.get("DEEPSHOT_DATA_DIR"):
        return app_data_dir() / "models"
    from PySide6.QtCore import QSettings

    chosen = str(QSettings().value("models/folder", "") or "").strip()
    if chosen and _writable_folder(Path(chosen)):
        return Path(chosen)
    beside = program_dir() / "models"
    return beside if _writable_folder(beside) else app_data_dir() / "models"


def save_models_dir(path: Path | None) -> None:
    """Recuerda la carpeta elegida (None = volver a la de por defecto)."""
    from PySide6.QtCore import QSettings

    if path is None:
        QSettings().remove("models/folder")
    else:
        QSettings().setValue("models/folder", str(path))


def _folder_family(folder: Path) -> str | None:
    """Familia de un modelo soltado a mano en una carpeta, o None si no es uno que la app sepa ejecutar.

    Acepta formato diffusers (model_index.json) y de transformers (config.json con su `model_type`), siempre con pesos
    .safetensors (los .bin / .pt / .ckpt pueden ejecutar código al cargarse y no se aceptan)."""
    if not any(folder.rglob("*.safetensors")):
        return None
    try:
        if (folder / "model_index.json").is_file():
            return detect_family(json.loads((folder / "model_index.json").read_text(encoding="utf-8")))
        if (folder / "config.json").is_file():
            kind = json.loads((folder / "config.json").read_text(encoding="utf-8")).get("model_type")
            if kind == "depth_anything":
                return "depth-anything"
            if kind in ("dpt", "zoedepth", "depth_pro"):
                return "transformers-depth"
    except (OSError, ValueError, AttributeError):
        return None
    return None


def dev_models_enabled() -> bool:
    forced = os.environ.get("DEEPSHOT_DEV_MODELS", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    from PySide6.QtCore import QSettings

    value = QSettings().value("models/show_test_models", DEV_MODELS_DEFAULT)
    return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def select_safe_files(names: list[str]) -> list[str]:
    """Filtra una lista de archivos de un modelo diffusers.

    - Solo extensiones seguras (descarta pickle, imágenes, README…).
    - Si una carpeta tiene pesos fp16 (`*.fp16.safetensors`), se quedan solo esos:
      ocupan la mitad y son los que se cargan en GPU.
    """
    safe = [n for n in names if PurePosixPath(n).suffix.lower() in SAFE_EXTENSIONS]
    dirs_with_fp16 = {str(PurePosixPath(n).parent) for n in safe if n.endswith(".fp16.safetensors")}
    result = []
    for n in safe:
        p = PurePosixPath(n)
        is_full_precision_weight = p.suffix == ".safetensors" and not n.endswith(".fp16.safetensors")
        if is_full_precision_weight and str(p.parent) in dirs_with_fp16:
            continue
        result.append(n)
    return sorted(result)


def detect_family(model_index: dict) -> str | None:
    """Familia según el `model_index.json` de diffusers; None si no está soportada."""
    class_name = model_index.get("_class_name")
    for fam in FAMILIES.values():
        if class_name == fam.pipeline_class:
            return fam.id
    return None


# Familias que se pueden descargar pero cuyo cálculo todavía no está programado en la app (ver depth.py).
UNIMPLEMENTED_FAMILIES = {"marigold-v2"}


def can_compute(entry: ModelEntry) -> bool:
    """¿Sabe la app calcular la profundidad con este modelo? Si no, descargarlo no sirve de nada todavía."""
    return entry.family not in UNIMPLEMENTED_FAMILIES


def is_permissive(license_id: str | None) -> bool:
    return (license_id or "").lower() in PERMISSIVE_LICENSES


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "modelo"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


# Autores de los modelos originales. El resto suelen ser copias o variantes de terceros.
OFFICIAL_AUTHORS = {"prs-eth", "huawei-bayerlab"}


@dataclass(frozen=True)
class SearchResult:
    repo_id: str
    family: str
    license: str
    downloads: int
    likes: int
    official: bool


class ModelError(Exception):
    """Error explicable al usuario (el mensaje ya está en español)."""


# ---------------------------------------------------------------------------
# Gestor
# ---------------------------------------------------------------------------
class ModelManager:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.set_models_dir(models_dir or resolve_models_dir())

    def set_models_dir(self, path: Path) -> None:
        """Usa otra carpeta de modelos (no mueve lo que haya en la anterior)."""
        self.models_dir = Path(path)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        from deepshot import modules

        modules.set_models_base(self.models_dir)       # los módulos viven en «looks», dentro de la carpeta de modelos
        self._user_file = self.models_dir / "installed.json"
        if not _XET_SET_BY_USER:
            os.environ["HF_XET_CACHE"] = str(self.models_dir / "xet")

    # --- Lista de modelos ------------------------------------------------------
    def _user_entries(self) -> list[ModelEntry]:
        try:
            raw = json.loads(self._user_file.read_text(encoding="utf-8"))
            return [ModelEntry(**item) for item in raw]
        except (OSError, ValueError, TypeError):
            return []

    def _save_user_entries(self, entries: list[ModelEntry]) -> None:
        self._user_file.write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")

    def all(self, include_hidden: bool = False, with_modules: bool = False) -> list[ModelEntry]:
        """Catálogo + modelos del usuario, en ese orden. Los modelos de prueba solo salen si están activados. Los módulos
        (capas de Look…) solo salen con `with_modules=True`: los selectores de modelos de profundidad no los quieren."""
        show_dev = include_hidden or dev_models_enabled()
        entries = [*MODELS_CATALOG, *self._user_entries(), *self.scan_folders()]
        if with_modules:
            entries += self.module_entries()
        return [e for e in entries if show_dev or not e.dev_only]

    # --- Módulos (capas de Look…): el catálogo local y los instalados, como entradas más de la lista --------------------
    @staticmethod
    def module_entries() -> list[ModelEntry]:
        """Los módulos del catálogo local y los ya instalados (aunque ya no estén en el catálogo)."""
        from deepshot import modules

        infos = {i.id: i for i in modules.catalog()}
        infos.update({i.id: i for i in modules.installed()})       # lo instalado manda sobre el catálogo
        return [ModelEntry(id=MODULE_ID_PREFIX + i.id, label=i.name, family=MODULE_FAMILY, source="module",
                           license=i.license, kind=i.kind, description=i.description, version=i.version,
                           requires=list(i.requires_models), url=i.homepage, path=i.id)
                for i in sorted(infos.values(), key=lambda i: i.name.lower())]

    def missing_requirements(self, entry: ModelEntry) -> list[ModelEntry]:
        """Los modelos que un módulo necesita y todavía no están instalados (o no existen en la lista)."""
        missing = []
        for model_id in entry.requires:
            try:
                needed = self.get(model_id)
            except ModelError:
                needed = ModelEntry(id=model_id, label=model_id, family="", source="catalog")
            if not self.is_installed_safe(needed.id):
                missing.append(needed)
        return missing

    # --- Modelos de la carpeta de datos anterior -------------------------------------------------------------------
    def legacy_dir(self) -> Path | None:
        """La carpeta de modelos antigua (dentro de los datos de la aplicación), si aún guarda modelos y no es la actual."""
        old = app_data_dir() / "models"
        if old.resolve() == self.models_dir.resolve() or not old.is_dir():
            return None
        has_models = any(old.glob("*.complete.json")) or any(old.glob("models--*")) or (old / "local").is_dir()
        return old if has_models else None

    def adopt_from(self, source: Path, progress: ProgressCallback = _no_progress) -> int:
        """Mueve los modelos de `source` a la carpeta de modelos actual (sin pisar nada que ya esté). Devuelve cuántas
        carpetas y archivos movió. En el mismo disco es instantáneo; entre discos copia y borra, y puede tardar."""
        moved = 0
        items = [i for i in sorted(source.iterdir()) if i.name not in ("xet", ".locks", "CACHEDIR.TAG")]
        for n, item in enumerate(items, 1):
            target = self.models_dir / item.name
            if target.exists():
                continue                                  # no se pisa nada
            progress(QCoreApplication.translate("Models", "Moviendo {0}…").format(item.name), n / max(len(items), 1))
            try:
                shutil.move(str(item), str(target))
                moved += 1
            except OSError as exc:
                raise ModelError(QCoreApplication.translate("Models", "No se pudo mover {0}: {1}").format(item.name, exc.strerror or exc)) from exc
        return moved

    # --- Qué modelos salen en el selector ----------------------------------------------------------------------------
    def disabled_ids(self) -> set[str]:
        """Los modelos que el usuario ha desmarcado en el gestor: siguen instalados, pero no salen en el selector."""
        from PySide6.QtCore import QSettings

        raw = QSettings().value("models/disabled", [])
        if isinstance(raw, str):
            raw = [raw] if raw else []
        return {str(i) for i in (raw or [])}

    def is_enabled(self, model_id: str) -> bool:
        return model_id not in self.disabled_ids()

    def set_enabled(self, model_id: str, enabled: bool) -> None:
        from PySide6.QtCore import QSettings

        disabled = self.disabled_ids()
        disabled.discard(model_id) if enabled else disabled.add(model_id)
        QSettings().setValue("models/disabled", sorted(disabled))

    def scan_folders(self) -> list[ModelEntry]:
        """Modelos que el usuario ha soltado a mano en la carpeta de modelos (cada uno en su propia carpeta).

        Se ignoran las carpetas que gestiona la app (`models--…`, `local`, `xet`, las que empiezan por punto) y las que
        no son un modelo compatible con pesos .safetensors."""
        found: list[ModelEntry] = []
        try:
            folders = sorted(p for p in self.models_dir.iterdir() if p.is_dir())
        except OSError:
            return found
        for folder in folders:
            if folder.name.startswith(("models--", ".")) or folder.name in ("local", "xet", "looks"):    # «looks»: módulos
                continue
            family = _folder_family(folder)
            if family:
                found.append(ModelEntry(id="dir-" + _slug(folder.name), label=folder.name, family=family,
                                        source="folder", path=folder.name))
        return found

    def get(self, model_id: str) -> ModelEntry:
        for entry in self.all(include_hidden=True, with_modules=True):      # un modelo oculto sigue encontrándose por su id
            if entry.id == model_id:
                return entry
        raise ModelError(QCoreApplication.translate("Models", "El modelo «{0}» no está en la lista.").format(model_id))

    # --- Estado en disco -------------------------------------------------------
    def _marker(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.complete.json"

    def _hf_repo_dir(self, repo_id: str) -> Path:
        return self.models_dir / ("models--" + repo_id.replace("/", "--"))

    def snapshot(self, model_id: str) -> Path | None:
        """Carpeta lista para `from_pretrained`, o None si el modelo no está instalado."""
        entry = self.get(model_id)
        if entry.is_module:
            from deepshot import modules

            return modules.modules_dir() / (entry.path or "") if modules.installed_info(entry.path or "") else None
        if entry.source in ("local", "folder"):
            folder = self.models_dir / (entry.path or "")
        else:
            try:
                info = json.loads(self._marker(model_id).read_text(encoding="utf-8"))
                # Ruta relativa: la carpeta de datos se puede mover sin volver a descargar.
                folder = self.models_dir / info["snapshot"]
            except (OSError, ValueError, KeyError):
                return None
        family = ALL_FAMILIES.get(entry.family)
        if not (folder / (family.index_file if family else "model_index.json")).is_file():
            return None
        # Si el modelo tiene más repositorios, todos deben estar completos.
        if entry.extra_repos and any(self.part_dir(model_id, repo) is None for repo in entry.extra_repos):
            return None
        return folder

    def part_dir(self, model_id: str, repo_id: str) -> Path | None:
        """Carpeta de un repositorio adicional del modelo (p. ej. el modelo base de Marigold V2), o None."""
        try:
            info = json.loads(self._marker(model_id).read_text(encoding="utf-8"))
            folder = self.models_dir / info["extra"][repo_id]
        except (OSError, ValueError, KeyError):
            return None
        return folder if folder.is_dir() else None

    def is_installed(self, model_id: str) -> bool:
        return self.snapshot(model_id) is not None

    def disk_size(self, model_id: str) -> int:
        """Bytes ocupados por el modelo (0 si no está instalado)."""
        entry = self.get(model_id)
        if entry.is_module:
            from deepshot import modules

            return modules.folder_size(entry.path or "")
        if entry.source in ("local", "folder"):
            return _dir_size(self.models_dir / (entry.path or ""))
        snap = self.snapshot(model_id)
        if not snap:
            return 0
        extra = sum(_dir_size(d) for d in (self.part_dir(model_id, r) for r in entry.extra_repos) if d)
        return _dir_size(snap) + extra

    @staticmethod
    def variant(snapshot: Path) -> str | None:
        """'fp16' si la carpeta trae pesos fp16; si no, None (pesos completos)."""
        return "fp16" if any(snapshot.rglob("*.fp16.safetensors")) else None

    # --- Descarga ----------------------------------------------------------------
    def download(self, model_id: str, progress: ProgressCallback = _no_progress, force: bool = False) -> Path:
        """Descarga un modelo del catálogo o de Hugging Face, con progreso en bytes.

        force=True vuelve a descargarlo aunque ya esté (es lo que usa «Actualizar»): al terminar bien se borra la versión
        anterior; si algo falla, la anterior se queda como estaba."""
        entry = self.get(model_id)
        if entry.is_module:                       # un módulo se copia del catálogo local, no se descarga
            from deepshot import modules

            progress(QCoreApplication.translate("Models", "Instalando el módulo {0}…").format(entry.display_label), None)
            try:
                return modules.install(entry.path or "").path
            except modules.ModuleError as exc:
                raise ModelError(str(exc)) from exc
        from huggingface_hub import HfApi, hf_hub_download
        from tqdm.auto import tqdm

        existing = self.snapshot(model_id)
        if existing is not None and not force:
            return existing
        old_marker = self._read_marker(model_id) if existing is not None else {}
        entry = self.get(model_id)
        if entry.source in ("local", "folder") or not entry.repo_id:
            raise ModelError(QCoreApplication.translate("Models", "«{0}» se instaló desde una carpeta y no se puede descargar.").format(entry.display_label))

        progress(QCoreApplication.translate("Models", "Consultando {0}…").format(entry.display_label), None)
        import fnmatch

        # Un trabajo por repositorio: el principal y, si los hay, los adicionales (Marigold V2 lleva dos).
        jobs: list[tuple[str, list[str], str, dict[str, int], list[str]]] = []   # repo, patrones, sha, tamaños, archivos
        for repo, patterns in [(entry.repo_id, entry.patterns), *entry.extra_repos.items()]:
            info = HfApi().model_info(repo, files_metadata=True)
            sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
            if patterns:
                names = [n for n in sizes if any(fnmatch.fnmatch(n, p) for p in patterns)]
            else:
                names = select_safe_files(list(sizes))
            # Los .pt solo se aceptan si el modelo los declara (tensores del autor, leídos sin ejecutar código).
            unsafe = [n for n in names if PurePosixPath(n).suffix.lower() in UNSAFE_EXTENSIONS
                      and n not in entry.tensor_files]
            if unsafe or not names:
                raise ModelError(QCoreApplication.translate("Models", "«{0}» no tiene un formato compatible y seguro.").format(entry.display_label))
            jobs.append((repo, patterns, info.sha, sizes, names))
        index_file = ALL_FAMILIES[entry.family].index_file
        if index_file not in jobs[0][4]:
            raise ModelError(QCoreApplication.translate("Models", "«{0}» no tiene un formato compatible y seguro.").format(entry.display_label))

        total = sum(job[3][n] for job in jobs for n in job[4]) or 1
        # Espacio libre: los modelos grandes (Marigold V2, ~43 GB) fallarían a mitad de la descarga. Se descuenta lo
        # que ya esté bajado (la descarga se reanuda) y se deja un margen de 2 GB.
        have = sum(_dir_size(self._hf_repo_dir(job[0])) for job in jobs)
        free = shutil.disk_usage(self.models_dir).free
        if total - have + 2e9 > free:
            raise ModelError(QCoreApplication.translate("Models", "No hay espacio suficiente en el disco: «{0}» necesita {1:.0f} GB libres y hay {2:.0f} GB.").format(entry.display_label, (total - have + 2e9) / 1e9, free / 1e9))
        done = 0

        class _FileProgress(tqdm):
            """tqdm silencioso que reenvía los bytes descargados a `progress`."""

            def __init__(self, *args, **kwargs):
                kwargs["disable"] = False
                kwargs["file"] = io.StringIO()  # no escribir en la consola
                super().__init__(*args, **kwargs)

            def update_transfer(self, n=1):
                # Bytes de red (xet). Basta con los bytes escritos en disco (`update`);
                # tener este método hace que huggingface_hub use una sola barra.
                pass

            def set_transfer_postfix_str(self, *args, **kwargs):
                pass  # huggingface_hub lo llama junto con update_transfer

            def update(self, n=1):
                super().update(n)
                mb = (done + self.n) / 1e6
                progress(QCoreApplication.translate("Models", "Descargando {0}… {1:,.0f} / {2:,.0f} MB").format(entry.display_label, mb, total / 1e6),
                         min((done + self.n) / total, 1.0))

        snapshot_dir: Path | None = None
        roots: dict[str, str] = {}        # repo -> carpeta de su versión descargada (relativa a models_dir)
        for repo, _patterns, sha, sizes, names in jobs:
            for name in names:
                local = hf_hub_download(
                    repo,
                    filename=name,
                    revision=sha,  # todos los archivos de la misma versión
                    cache_dir=self.models_dir,
                    tqdm_class=_FileProgress,
                )
                done += sizes[name]
                # La carpeta de la versión es la que contiene el archivo, tantos niveles arriba como partes tiene su ruta.
                root = Path(local).parents[len(PurePosixPath(name).parts) - 1]
                roots[repo] = root.relative_to(self.models_dir).as_posix()
                if repo == entry.repo_id and name == index_file:
                    snapshot_dir = root

        assert snapshot_dir is not None
        info = jobs[0]
        self._marker(model_id).write_text(
            json.dumps({"model": entry.repo_id, "revision": info[2],
                        "snapshot": snapshot_dir.relative_to(self.models_dir).as_posix(),
                        "extra": {r: roots[r] for r in entry.extra_repos}}),
            encoding="utf-8",
        )
        if old_marker:
            self._remove_old_revision(entry, old_marker, keep=snapshot_dir)
        log.info("Modelo %s descargado en %s", entry.label, snapshot_dir)
        return snapshot_dir

    # --- Actualizar ----------------------------------------------------------------
    def _read_marker(self, model_id: str) -> dict:
        try:
            return json.loads(self._marker(model_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def installed_revision(self, model_id: str) -> str | None:
        """La versión (commit) del repositorio con la que se descargó el modelo, o None si no se sabe."""
        return self._read_marker(model_id).get("revision") or None

    def check_updates(self, progress: ProgressCallback = _no_progress) -> set[str]:
        """Ids de los modelos instalados de Hugging Face cuyo repositorio tiene una versión más nueva.

        Es una consulta manual a Hugging Face (solo lee la versión actual de cada repositorio, no descarga nada)."""
        from huggingface_hub import HfApi

        newer: set[str] = set()
        entries = [e for e in self.all(include_hidden=True)
                   if e.repo_id and e.source != "local" and self.is_installed(e.id) and self.installed_revision(e.id)]
        for n, entry in enumerate(entries, 1):
            progress(QCoreApplication.translate("Models", "Comprobando {0}…").format(entry.display_label),
                     n / max(len(entries), 1))
            try:
                latest = HfApi().model_info(entry.repo_id).sha
            except Exception as exc:
                log.warning("No se pudo consultar %s: %s", entry.repo_id, exc)
                raise ModelError(QCoreApplication.translate("Models", "No se pudo conectar con Hugging Face. Comprueba la conexión.")) from exc
            if latest and latest != self.installed_revision(entry.id):
                newer.add(entry.id)
        return newer

    def update(self, model_id: str, progress: ProgressCallback = _no_progress) -> Path:
        """Descarga la versión más nueva de un modelo y borra la anterior al terminar bien."""
        return self.download(model_id, progress, force=True)

    def _remove_old_revision(self, entry: ModelEntry, old_marker: dict, keep: Path) -> None:
        """Borra la carpeta de la versión anterior (y su caché de Hugging Face) tras una actualización correcta."""
        old_rel = old_marker.get("snapshot")
        if not old_rel:
            return
        old_dir = self.models_dir / old_rel
        if not old_dir.exists() or old_dir.resolve() == keep.resolve():
            return
        old_revision = old_marker.get("revision")
        try:
            from huggingface_hub import scan_cache_dir

            info = scan_cache_dir(self.models_dir)
            hashes = [r.commit_hash for repo in info.repos if repo.repo_id == entry.repo_id
                      for r in repo.revisions if r.commit_hash == old_revision]
            if hashes:
                info.delete_revisions(*hashes).execute()
        except Exception as exc:                          # la limpieza fina falla: se borra la carpeta a mano
            log.info("No se pudo limpiar la caché de Hugging Face (%s); se borra la carpeta.", exc)
        shutil.rmtree(old_dir, ignore_errors=True)

    # --- Cargar al arrancar ---------------------------------------------------------
    def preload_id(self) -> str | None:
        """El modelo que se carga en la GPU al abrir la aplicación (solo uno), si lo hay y está instalado."""
        from PySide6.QtCore import QSettings

        model_id = str(QSettings().value("models/preload", "") or "")
        return model_id if model_id and self.is_installed_safe(model_id) else None

    def is_installed_safe(self, model_id: str) -> bool:
        try:
            return self.is_installed(model_id)
        except ModelError:
            return False

    def set_preload(self, model_id: str | None) -> None:
        from PySide6.QtCore import QSettings

        if model_id:
            QSettings().setValue("models/preload", model_id)
        else:
            QSettings().remove("models/preload")

    # --- Buscar ------------------------------------------------------------------
    def search_compatible(self, limit: int = 50) -> list[SearchResult]:
        """Modelos públicos de Hugging Face compatibles con alguna familia soportada.

        Hugging Face etiqueta cada modelo diffusers con su pipeline
        (p. ej. "diffusers:MarigoldDepthPipeline"); se busca por esa etiqueta.
        Se excluyen los repos con acceso restringido (gated), que exigen cuenta.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        results: list[SearchResult] = []
        for fam in FAMILIES.values():
            try:
                found = api.list_models(filter=f"diffusers:{fam.pipeline_class}", sort="downloads",
                                        limit=limit, expand=["downloads", "likes", "tags", "gated"])
                found = list(found)
            except Exception as exc:
                log.warning("Búsqueda en Hugging Face fallida: %s", exc)
                raise ModelError(QCoreApplication.translate("Models", "No se pudo conectar con Hugging Face. Comprueba la conexión.")) from exc
            for info in found:
                if getattr(info, "gated", False) or getattr(info, "private", False):
                    continue
                license_id = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")),
                                  "desconocida")
                results.append(SearchResult(
                    repo_id=info.id, family=fam.id, license=license_id,
                    downloads=int(getattr(info, "downloads", 0) or 0), likes=int(getattr(info, "likes", 0) or 0),
                    official=info.id.split("/")[0] in OFFICIAL_AUTHORS))
        # Primero los oficiales; después, por descargas.
        results.sort(key=lambda r: (not r.official, -r.downloads))
        return results

    def is_listed(self, repo_id: str) -> bool:
        """¿Está ya este repo en la lista (catálogo o añadido)?"""
        return any(e.repo_id == repo_id for e in self.all())

    # --- Añadir modelos ----------------------------------------------------------
    def add_from_huggingface(self, repo_id: str, progress: ProgressCallback = _no_progress) -> ModelEntry:
        """Registra un repo de Hugging Face (org/modelo) tras comprobar que es compatible.

        No descarga los pesos: solo lee su `model_index.json` y la licencia.
        Después se descarga con `download()`.
        """
        from huggingface_hub import HfApi, hf_hub_download

        repo_id = repo_id.strip().strip("/")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
            raise ModelError(QCoreApplication.translate("Models", "Escribe el nombre del repositorio como «organización/modelo»."))
        model_id = "hf-" + _slug(repo_id)
        if any(e.id == model_id or e.repo_id == repo_id for e in self.all()):
            raise ModelError(QCoreApplication.translate("Models", "«{0}» ya está en la lista.").format(repo_id))

        progress(QCoreApplication.translate("Models", "Comprobando {0}…").format(repo_id), None)
        try:
            info = HfApi().model_info(repo_id)
            index_path = hf_hub_download(repo_id, "model_index.json", revision=info.sha,
                                         cache_dir=self.models_dir)
        except Exception as exc:  # repo inexistente, privado, sin model_index…
            log.warning("No se pudo leer %s: %s", repo_id, exc)
            raise ModelError(QCoreApplication.translate("Models", "No se encontró un modelo diffusers en «{0}».").format(repo_id)) from exc

        family = detect_family(json.loads(Path(index_path).read_text(encoding="utf-8")))
        if family is None:
            raise ModelError(QCoreApplication.translate("Models", "«{0}» no es de una familia soportada ({1}).").format(repo_id, ', '.join(f.label for f in FAMILIES.values())))
        card = getattr(info, "card_data", None) or {}
        license_id = (card.get("license") if hasattr(card, "get") else None) or "desconocida"

        entry = ModelEntry(id=model_id, label=repo_id.split("/")[-1], family=family,
                           source="huggingface", repo_id=repo_id, license=str(license_id))
        self._save_user_entries([*self._user_entries(), entry])
        log.info("Modelo añadido desde Hugging Face: %s (%s, licencia %s)", repo_id, family, license_id)
        return entry

    def install_from_folder(self, folder: Path, label: str | None = None,
                            progress: ProgressCallback = _no_progress) -> ModelEntry:
        """Copia un modelo diffusers desde una carpeta local (solo archivos seguros)."""
        folder = Path(folder)
        index = folder / "model_index.json"
        if not index.is_file():
            raise ModelError(QCoreApplication.translate("Models", "La carpeta no contiene un modelo diffusers (falta model_index.json)."))
        try:
            family = detect_family(json.loads(index.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise ModelError(QCoreApplication.translate("Models", "El archivo model_index.json no es válido.")) from exc
        if family is None:
            raise ModelError(QCoreApplication.translate("Models", "El modelo de la carpeta no es de una familia soportada ({0}).").format(', '.join(f.label for f in FAMILIES.values())))

        all_files = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()]
        names = select_safe_files(all_files)
        if not any(n.endswith(".safetensors") for n in names):
            raise ModelError(QCoreApplication.translate("Models", "La carpeta no tiene pesos en formato .safetensors. Por seguridad no se aceptan .bin, .pt ni .ckpt."))
        skipped = sorted(set(all_files) - set(names))
        if skipped:
            log.info("Archivos no copiados (formato no seguro o innecesario): %s", skipped)

        label = (label or folder.name).strip()
        model_id = "local-" + _slug(label)
        existing_ids = {e.id for e in self.all()}
        base, n = model_id, 2
        while model_id in existing_ids:
            model_id, n = f"{base}-{n}", n + 1
        dest = self.models_dir / "local" / model_id

        total = sum((folder / n).stat().st_size for n in names) or 1
        done = 0
        try:
            for name in names:
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder / name, target)
                done += (folder / name).stat().st_size
                progress(QCoreApplication.translate("Models", "Copiando {0}… {1:,.0f} / {2:,.0f} MB").format(label, done / 1e6, total / 1e6), done / total)
        except OSError as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise ModelError(QCoreApplication.translate("Models", "No se pudo copiar el modelo: {0}").format(exc.strerror or exc)) from exc

        entry = ModelEntry(id=model_id, label=label, family=family, source="local",
                           path=dest.relative_to(self.models_dir).as_posix())
        self._save_user_entries([*self._user_entries(), entry])
        log.info("Modelo instalado desde carpeta: %s -> %s", folder, dest)
        return entry

    # --- Borrar ------------------------------------------------------------------
    def delete(self, model_id: str) -> None:
        """Borra los archivos del modelo. Los del usuario además salen de la lista."""
        entry = self.get(model_id)
        if entry.is_module:
            from deepshot import modules

            modules.uninstall(entry.path or "__nada__")
            log.info("Módulo borrado: %s", entry.label)
            return
        if entry.source in ("local", "folder"):
            shutil.rmtree(self.models_dir / (entry.path or "__nada__"), ignore_errors=True)
        elif entry.repo_id:
            shutil.rmtree(self._hf_repo_dir(entry.repo_id), ignore_errors=True)
            for repo in entry.extra_repos:      # p. ej. el modelo base de Marigold V2
                shutil.rmtree(self._hf_repo_dir(repo), ignore_errors=True)
        self._marker(model_id).unlink(missing_ok=True)
        if entry.removable:
            self._save_user_entries([e for e in self._user_entries() if e.id != model_id])
        log.info("Modelo borrado: %s", entry.label)
