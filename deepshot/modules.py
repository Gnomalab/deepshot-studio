"""Módulos descargables: capas de Look (y más adelante otras familias) que se instalan desde el gestor (sin interfaz).

Un módulo es una carpeta con dos archivos:
    module.json   nombre, versión, licencia, tipo, ajustes (con rango y valor inicial), modelos que necesita
    module.py     una función `process(image, depth, params) -> image` (float32 alto×ancho×3 en 0..1)
La interfaz dibuja los controles a partir de `params` del JSON, así que un módulo no necesita saber de Qt.

Los módulos son para cosas complejas que necesitan modelos o librerías (relighting, embellecer…). Los efectos básicos
(niveles, contraste, saturación, grano, nitidez, viñeta) vienen de serie en deepshot/look.py.

Dónde viven:
    <carpeta de modelos>/looks/<id>/   módulos instalados, junto a los modelos pero en su propia carpeta
                                       (DEEPSHOT_MODULES_DIR cambia la carpeta)
    modules_catalog/<id>/              catálogo local junto al programa: lo que se ofrece en el gestor
                                       (DEEPSHOT_MODULES_CATALOG)
«Instalar» copia la carpeta del catálogo a la de módulos; «Desinstalar» la borra. Nada se descarga de internet.

Seguridad: `module.py` es código que se ejecuta con los permisos del usuario. Solo se carga un módulo que el usuario
haya aprobado (al instalarlo desde el gestor o en el aviso que sale al arrancar). La aprobación guarda una huella de
los archivos: si alguien edita el módulo después, hay que volver a aprobarlo. Leer `module.json` no ejecuta nada.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QSettings

from deepshot import look
from deepshot.look import LayerKind, ParamSpec
from deepshot.models import program_dir, resolve_models_dir

log = logging.getLogger(__name__)

LOOKS_DIRNAME = "looks"                            # subcarpeta de la carpeta de modelos donde viven los módulos de Look
MODULE_FILE = "module.json"
CODE_FILE = "module.py"
MODULE_KINDS = ("look",)                           # familias de módulo que la app sabe usar
KIND_PREFIX = "module."                            # el tipo de capa de un módulo es «module.<id>»
_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_PARAM = re.compile(r"^[a-z][a-z0-9_]{0,30}$")
_COPY_SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")


class ModuleError(Exception):
    """Error explicable al usuario."""


@dataclass
class ModuleInfo:
    id: str
    name: str
    version: str = "1.0"
    license: str = "desconocida"
    kind: str = "look"
    description: str = ""
    author: str = ""
    requires_models: list[str] = field(default_factory=list)      # ids de modelos del gestor que necesita
    homepage: str = ""                                            # página del módulo (http/https), para el enlace del gestor
    params: list[ParamSpec] = field(default_factory=list)
    path: Path = Path(".")

    @property
    def layer_kind(self) -> str:
        return KIND_PREFIX + self.id


# ---------------------------------------------------------------------------
# Carpetas
# ---------------------------------------------------------------------------
_models_base: Path | None = None


def set_models_base(path: Path) -> None:
    """La carpeta de modelos en uso (la fija el gestor de modelos al arrancar y al cambiarla): los módulos viven en su
    subcarpeta `looks`, separados de los modelos de profundidad."""
    global _models_base
    _models_base = Path(path)


def modules_dir() -> Path:
    """Carpeta de los módulos instalados: `looks` dentro de la carpeta de modelos."""
    forced = os.environ.get("DEEPSHOT_MODULES_DIR", "").strip()
    if forced:
        return Path(forced)
    return (_models_base if _models_base is not None else resolve_models_dir()) / LOOKS_DIRNAME


def catalog_dir() -> Path:
    """Carpeta del catálogo local de módulos (junto al programa)."""
    forced = os.environ.get("DEEPSHOT_MODULES_CATALOG", "").strip()
    return Path(forced) if forced else program_dir() / "modules_catalog"


# ---------------------------------------------------------------------------
# Lectura de module.json (no ejecuta código)
# ---------------------------------------------------------------------------
def _number(spec: dict, key: str, default: float) -> float:
    value = spec.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModuleError(QCoreApplication.translate("Modules", "El ajuste «{0}» tiene un valor no numérico en «{1}».").format(spec.get("name", "?"), key))
    return float(value)


def read_info(folder: Path) -> ModuleInfo:
    """Lee y comprueba `module.json` de una carpeta. No importa `module.py`."""
    import json

    manifest = Path(folder) / MODULE_FILE
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModuleError(QCoreApplication.translate("Modules", "Falta {0} en la carpeta del módulo.").format(MODULE_FILE)) from exc
    except ValueError as exc:
        raise ModuleError(QCoreApplication.translate("Modules", "{0} no es un JSON válido: {1}").format(MODULE_FILE, exc)) from exc
    if not isinstance(data, dict):
        raise ModuleError(QCoreApplication.translate("Modules", "{0} tiene que ser un objeto JSON.").format(MODULE_FILE))
    module_id = str(data.get("id", ""))
    if not _ID.match(module_id):
        raise ModuleError(QCoreApplication.translate("Modules", "El id del módulo «{0}» no vale: usa minúsculas, números y guion bajo, empezando por letra.").format(module_id))
    if module_id != Path(folder).name:
        raise ModuleError(QCoreApplication.translate("Modules", "La carpeta del módulo debe llamarse como su id («{0}»).").format(module_id))
    kind = str(data.get("kind", "look"))
    if kind not in MODULE_KINDS:
        raise ModuleError(QCoreApplication.translate("Modules", "Esta versión de la aplicación no sabe usar módulos de tipo «{0}».").format(kind))
    if not (Path(folder) / CODE_FILE).is_file():
        raise ModuleError(QCoreApplication.translate("Modules", "Falta {0} en la carpeta del módulo.").format(CODE_FILE))
    params, seen = [], set()
    for spec in data.get("params", []):
        name = str(spec.get("name", "")) if isinstance(spec, dict) else ""
        if not _PARAM.match(name) or name in seen:
            raise ModuleError(QCoreApplication.translate("Modules", "El nombre de ajuste «{0}» no vale o está repetido.").format(name))
        seen.add(name)
        lo, hi = _number(spec, "min", 0.0), _number(spec, "max", 1.0)
        default = _number(spec, "default", lo)
        if not lo < hi or not lo <= default <= hi:
            raise ModuleError(QCoreApplication.translate("Modules", "El ajuste «{0}» tiene un rango o un valor inicial incoherente.").format(name))
        params.append(ParamSpec(name=name, label=str(spec.get("label", name)), default=default, minimum=lo, maximum=hi,
                                decimals=int(_number(spec, "decimals", 2)), step=_number(spec, "step", (hi - lo) / 100),
                                tooltip=str(spec.get("tooltip", ""))))
    requires = data.get("requires_models", [])
    if not isinstance(requires, list) or not all(isinstance(r, str) for r in requires):
        raise ModuleError(QCoreApplication.translate("Modules", "«requires_models» tiene que ser una lista de ids de modelos."))
    homepage = str(data.get("homepage", "")).strip()
    if not homepage.startswith(("https://", "http://")):
        homepage = ""                                                    # solo enlaces web: nada de file:// ni rutas
    return ModuleInfo(id=module_id, name=str(data.get("name", module_id)), version=str(data.get("version", "1.0")),
                      license=str(data.get("license", "desconocida")), kind=kind,
                      description=str(data.get("description", "")), author=str(data.get("author", "")),
                      requires_models=list(requires), homepage=homepage, params=params, path=Path(folder))


def _scan(root: Path) -> list[ModuleInfo]:
    found: list[ModuleInfo] = []
    try:
        folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    except OSError:
        return found
    for folder in folders:
        try:
            found.append(read_info(folder))
        except ModuleError as exc:                       # un módulo roto no impide ver los demás
            log.warning("Módulo ignorado (%s): %s", folder, exc)
    return found


def installed() -> list[ModuleInfo]:
    return _scan(modules_dir())


def catalog() -> list[ModuleInfo]:
    return _scan(catalog_dir())


def installed_info(module_id: str) -> ModuleInfo | None:
    folder = modules_dir() / module_id
    if not (folder / MODULE_FILE).is_file():
        return None
    try:
        return read_info(folder)
    except ModuleError:
        return None


# ---------------------------------------------------------------------------
# Instalar y quitar
# ---------------------------------------------------------------------------
def install(module_id: str) -> ModuleInfo:
    """Copia un módulo del catálogo a la carpeta de módulos (sustituye la versión que hubiera)."""
    source = catalog_dir() / module_id
    info = read_info(source) if source.is_dir() else None
    if info is None:
        raise ModuleError(QCoreApplication.translate("Modules", "El módulo «{0}» no está en el catálogo.").format(module_id))
    return install_from_folder(source)


def install_from_folder(folder: Path) -> ModuleInfo:
    """Instala un módulo desde cualquier carpeta que lo contenga (se valida antes de copiar)."""
    info = read_info(folder)
    dest = modules_dir() / info.id
    try:
        modules_dir().mkdir(parents=True, exist_ok=True)
        staging = modules_dir() / (".nuevo_" + info.id)
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(folder, staging, ignore=_COPY_SKIP)
        shutil.rmtree(dest, ignore_errors=True)
        staging.rename(dest)
    except OSError as exc:
        raise ModuleError(QCoreApplication.translate("Modules", "No se pudo copiar el módulo: {0}").format(exc.strerror or exc)) from exc
    log.info("Módulo instalado: %s -> %s", info.id, dest)
    return read_info(dest)


def uninstall(module_id: str) -> None:
    look.unregister_module_kinds(module_id)
    shutil.rmtree(modules_dir() / module_id, ignore_errors=True)
    forget(module_id)
    log.info("Módulo desinstalado: %s", module_id)


def folder_size(module_id: str) -> int:
    folder = modules_dir() / module_id
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else 0


# ---------------------------------------------------------------------------
# Aprobación (el código del módulo se ejecuta con los permisos del usuario)
# ---------------------------------------------------------------------------
def digest(info: ModuleInfo) -> str:
    """Huella de los archivos del módulo: cambia si se edita cualquiera."""
    h = hashlib.sha256()
    for name in (MODULE_FILE, CODE_FILE):
        h.update((Path(info.path) / name).read_bytes())
    return h.hexdigest()[:32]


def _approvals() -> dict[str, str]:
    raw = QSettings().value("modules/approved", [])
    if isinstance(raw, str):
        raw = [raw] if raw else []
    out = {}
    for item in raw or []:
        module_id, _, h = str(item).partition(":")
        if module_id and h:
            out[module_id] = h
    return out


def is_approved(info: ModuleInfo) -> bool:
    return _approvals().get(info.id) == digest(info)


def approve(info: ModuleInfo) -> None:
    approvals = _approvals()
    approvals[info.id] = digest(info)
    QSettings().setValue("modules/approved", [f"{k}:{v}" for k, v in sorted(approvals.items())])


def forget(module_id: str) -> None:
    approvals = _approvals()
    if approvals.pop(module_id, None) is not None:
        QSettings().setValue("modules/approved", [f"{k}:{v}" for k, v in sorted(approvals.items())])


# ---------------------------------------------------------------------------
# Carga (aquí sí se ejecuta module.py)
# ---------------------------------------------------------------------------
def load_kind(info: ModuleInfo) -> LayerKind:
    """Importa `module.py` y devuelve el tipo de capa del módulo. Solo para módulos aprobados."""
    spec = importlib.util.spec_from_file_location("deepshot_module_" + info.id, Path(info.path) / CODE_FILE)
    if spec is None or spec.loader is None:
        raise ModuleError(QCoreApplication.translate("Modules", "No se pudo leer {0}.").format(CODE_FILE))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        log.exception("El módulo %s falló al cargarse", info.id)
        raise ModuleError(QCoreApplication.translate("Modules", "El módulo «{0}» falló al cargarse: {1}").format(info.name, exc)) from exc
    process = getattr(module, "process", None)
    if not callable(process):
        raise ModuleError(QCoreApplication.translate("Modules", "El módulo «{0}» no define la función process(image, depth, params).").format(info.name))
    return LayerKind(key=info.layer_kind, title=info.name, params=tuple(info.params),
                     apply=lambda image, depth, params, _p=process: _p(image, depth, params), module_id=info.id)


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)                 # ids cargados
    pending: list[ModuleInfo] = field(default_factory=list)         # sin aprobar: no se cargan
    failed: dict[str, str] = field(default_factory=dict)            # id -> motivo


def register_installed() -> LoadReport:
    """Carga los módulos instalados y aprobados y los da de alta como tipos de capa de Look."""
    look.unregister_module_kinds()
    report = LoadReport()
    for info in installed():
        if info.kind != "look":
            continue
        if not is_approved(info):
            report.pending.append(info)
            continue
        try:
            look.register_kind(load_kind(info))
            report.loaded.append(info.id)
        except (ModuleError, ValueError) as exc:
            report.failed[info.id] = str(exc)
    return report
