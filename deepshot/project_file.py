"""Archivo de proyecto (.deepshot): guarda el trabajo para retomarlo en otro momento. Sin interfaz.

Es un ZIP con:
  manifest.json   lista de fotos, foto actual, versión
  depth/<n>.png   mapa de profundidad de cada foto (PNG de 16 bits)
  mask/<n>.png    máscara pintada con pincel (si la hay)

  photo/<n>_<nombre>  la foto misma (solo si se guardó con «Incluir las fotos»)

Por defecto las fotos NO se copian dentro (pueden pesar mucho): se guarda su ruta, su nombre, su tamaño en bytes y su
huella SHA-256. Al abrir, se buscan en la ruta original y junto al proyecto; si el contenido cambió se avisa. Con
«Incluir las fotos» el proyecto lleva también una copia de cada foto: si al abrir no está la original (o ha cambiado),
se usa la copia, que se saca a `<datos>/embedded/<huella>/` (un solo miembro, por su nombre, nunca el ZIP entero).
El ZIP nunca se extrae a disco: solo se leen sus miembros por nombre (no hay riesgo de rutas maliciosas).
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication

import hashlib
import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

EXTENSION = ".deepshot"
PROJECT_VERSION = 1
MANIFEST = "manifest.json"


class ProjectFileError(ValueError):
    """El archivo no es un proyecto válido (o está dañado)."""


@dataclass
class PhotoEntry:
    """Una foto de la biblioteca con todo lo suyo."""

    path: str                              # ruta de la foto en el equipo que guardó el proyecto
    name: str = ""                         # nombre del archivo (para buscarla junto al proyecto)
    size: int = 0                          # bytes
    sha256: str = ""                       # huella del contenido ("" si no se pudo calcular)
    source: str = ""                       # modelo que calculó el mapa, o "external"
    project: dict | None = None            # ajustes del efecto (Project.to_dict)
    depth: np.ndarray | None = None        # float32 0..1, 0 = cerca
    mask: bytes | None = None              # PNG de la máscara pintada
    crop: list | None = None               # reencuadre previo (x0, y0, x1, y1 normalizados) o None = foto entera
    embedded: str = ""                     # nombre del miembro del ZIP con la copia de la foto ("" = no la lleva)
    exports: list = field(default_factory=list)    # rutas de lo que se exportó de esta foto (para «Recopilar archivos»)


@dataclass
class LoadedProject:
    photos: list[PhotoEntry] = field(default_factory=list)
    current: int = 0
    app_version: str = ""
    saved: str = ""


# ---------------------------------------------------------------------------
# PNG de 16 bits en memoria (solo Pillow y numpy)
# ---------------------------------------------------------------------------
def _depth_to_png(depth: np.ndarray) -> bytes:
    data = np.round(np.clip(depth, 0.0, 1.0) * 65535.0).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(data).save(buf, format="PNG")
    return buf.getvalue()


def _png_to_depth(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as img:
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            data, top = np.asarray(img, dtype=np.float32), 65535.0
        else:
            data, top = np.asarray(img.convert("L"), dtype=np.float32), 255.0
    return np.clip(data / top, 0.0, 1.0)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def photo_size(path: Path) -> tuple[int, int] | None:
    """(ancho, alto) tal como la app ve la foto (con la rotación EXIF aplicada), o None si no se puede abrir.

    Las fotos de móvil en vertical se guardan giradas con una etiqueta EXIF: la app las endereza al cargarlas
    (depth.py), así que el mapa tiene el tamaño ya enderezado. Sin tenerlo en cuenta, el ancho y el alto salían
    cambiados y el proyecto descartaba el mapa creyendo que la foto era otra."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):      # orientaciones que giran 90°
                w, h = h, w
            return w, h
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Guardar
# ---------------------------------------------------------------------------
def save_project(path: Path, photos: list[PhotoEntry], current: int = 0, app_version: str = "",
                 embed_photos: bool = False) -> None:
    """Escribe el proyecto. Se escribe a un archivo temporal y se renombra: un fallo a medias no destruye
    el proyecto anterior. Con `embed_photos` copia dentro también cada foto (las que existan en disco)."""
    path = Path(path)
    manifest = {"deepshot_project": PROJECT_VERSION, "app_version": app_version,
                "saved": datetime.now().isoformat(timespec="seconds"),
                "current": max(0, min(current, max(len(photos) - 1, 0))), "photos": []}
    tmp = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for i, e in enumerate(photos):
                item = {"path": e.path, "name": e.name or Path(e.path).name, "size": e.size, "sha256": e.sha256,
                        "source": e.source, "project": e.project, "depth": None, "mask": None,
                        "crop": list(e.crop) if e.crop else None, "photo": None, "exports": list(e.exports)}
                if embed_photos and Path(e.path).is_file():
                    item["photo"] = f"photo/{i}_{Path(e.path).name}"
                    z.write(e.path, item["photo"], compress_type=zipfile.ZIP_STORED)      # un JPG ya va comprimido
                if e.depth is not None:
                    item["depth"] = f"depth/{i}.png"
                    z.writestr(item["depth"], _depth_to_png(e.depth), compress_type=zipfile.ZIP_STORED)
                if e.mask:
                    item["mask"] = f"mask/{i}.png"
                    z.writestr(item["mask"], e.mask, compress_type=zipfile.ZIP_STORED)
                manifest["photos"].append(item)
            z.writestr(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    log.info("Proyecto guardado: %s (%d foto/s)", path, len(photos))


def entry_for_photo(path: Path, project: dict | None, depth: np.ndarray | None, source: str,
                    mask_file: Path | None, crop: tuple | None = None, exports: list | None = None) -> PhotoEntry:
    """Reúne los datos de una foto de la biblioteca (calcula su huella si el archivo existe)."""
    path = Path(path)
    size, digest = 0, ""
    try:
        size, digest = path.stat().st_size, sha256_of(path)
    except OSError:
        log.warning("No se pudo leer la foto para el proyecto: %s", path)
    mask = None
    if mask_file is not None:
        try:
            mask = Path(mask_file).read_bytes() if Path(mask_file).is_file() else None
        except OSError:
            mask = None
    return PhotoEntry(str(path), path.name, size, digest, source, project, depth, mask, list(crop) if crop else None,
                      exports=list(exports or []))


# ---------------------------------------------------------------------------
# Abrir
# ---------------------------------------------------------------------------
def load_project(path: Path) -> LoadedProject:
    """Lee un proyecto. ProjectFileError si no es válido."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if MANIFEST not in names:
                raise ProjectFileError(QCoreApplication.translate("ProjectFile", "No es un proyecto de Deepshot Studio (falta el manifiesto)."))
            manifest = json.loads(z.read(MANIFEST).decode("utf-8"))
            if not isinstance(manifest, dict) or "deepshot_project" not in manifest:
                raise ProjectFileError(QCoreApplication.translate("ProjectFile", "No es un proyecto de Deepshot Studio."))
            if int(manifest["deepshot_project"]) > PROJECT_VERSION:
                raise ProjectFileError(QCoreApplication.translate("ProjectFile", "El proyecto es de una versión más reciente de la aplicación."))
            out = LoadedProject(current=int(manifest.get("current", 0)),
                                app_version=str(manifest.get("app_version", "")),
                                saved=str(manifest.get("saved", "")))
            for item in manifest.get("photos", []):
                entry = PhotoEntry(path=str(item["path"]), name=str(item.get("name", "")),
                                   size=int(item.get("size", 0)), sha256=str(item.get("sha256", "")),
                                   source=str(item.get("source", "")), project=item.get("project"),
                                   crop=item.get("crop") if isinstance(item.get("crop"), list) else None,
                                   exports=[str(p) for p in item.get("exports", []) if isinstance(p, str)])
                if item.get("photo") in names:
                    entry.embedded = str(item["photo"])
                if item.get("depth") in names:
                    entry.depth = _png_to_depth(z.read(item["depth"]))
                if item.get("mask") in names:
                    entry.mask = z.read(item["mask"])
                out.photos.append(entry)
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        if isinstance(exc, ProjectFileError):
            raise
        raise ProjectFileError(QCoreApplication.translate("ProjectFile", "No se pudo leer el proyecto: {0}").format(exc)) from exc
    out.current = max(0, min(out.current, max(len(out.photos) - 1, 0)))
    return out


def extract_embedded(project_file: Path, entry: PhotoEntry) -> Path | None:
    """Saca de un proyecto la copia de una foto a `<datos>/embedded/<huella>/<nombre>` y devuelve su ruta (None si el
    proyecto no la lleva, no se puede leer o no coincide con su huella). Si ya estaba sacada, se reutiliza."""
    if not entry.embedded:
        return None
    from deepshot.paths import data_subdir

    name = Path(entry.name or Path(entry.path).name).name            # solo el nombre: nunca una ruta del ZIP
    folder = data_subdir("embedded") / (entry.sha256[:16] or "sin-huella")
    target = folder / name
    try:
        if target.is_file() and (not entry.sha256 or sha256_of(target) == entry.sha256):
            return target
        with zipfile.ZipFile(project_file) as z:
            if entry.embedded not in z.namelist():
                return None
            folder.mkdir(parents=True, exist_ok=True)
            with z.open(entry.embedded) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        if entry.sha256 and sha256_of(target) != entry.sha256:
            target.unlink(missing_ok=True)
            return None
        return target
    except (OSError, zipfile.BadZipFile, KeyError):
        log.warning("No se pudo sacar la foto incluida en el proyecto: %s", entry.embedded)
        return None


def locate_photo(entry: PhotoEntry, project_file: Path) -> tuple[Path | None, bool]:
    """Busca la foto de una entrada: primero su ruta original y luego la carpeta del proyecto.

    Devuelve (ruta o None, cambiada): `cambiada` es True si el contenido ya no coincide con la huella guardada.
    """
    candidates = [Path(entry.path)]
    if entry.name:
        candidates.append(Path(project_file).parent / entry.name)
        candidates.append(Path(project_file).parent / "photos" / entry.name)       # carpeta de «Recopilar archivos»
    for candidate in candidates:
        if candidate.is_file():
            changed = False
            if entry.sha256:
                try:
                    changed = sha256_of(candidate) != entry.sha256
                except OSError:
                    changed = True
            return candidate, changed
    return None, False
