"""Recopilar archivos: todo lo del trabajo en una carpeta ordenada, para guardarlo o llevarlo a otro equipo. Sin interfaz.

Como «Collect Files» de After Effects o el «Save Project + Media» de MadMapper:

    <nombre>/
      <nombre>.deepshot      el proyecto, que encuentra las fotos en photos/ aunque se mueva la carpeta
      photos/                las fotos originales
      maps/                  el mapa de profundidad de cada foto (PNG de 16 bits, cerca = 0)
      fx/<tipo>/*.json       los presets propios (rampas, estilos, bokeh, looks y exportaciones), si los hay
      exports/<photo>/  lo que se exportó de cada foto, si se exportó algo
      README.txt              qué hay en cada sitio y qué módulos de Look usa el trabajo

No cambia nada del proyecto abierto: solo copia. Los módulos descargables no se copian (son código): en README.txt
se anota cuáles usa el trabajo, para instalarlos en el otro equipo desde el gestor.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QCoreApplication

from deepshot import presets, project_file
from deepshot.project_file import PhotoEntry

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]
PHOTOS, MAPS, FX, EXPORTS = "photos", "maps", "fx", "exports"


@dataclass
class CollectResult:
    folder: Path
    project: Path
    photos: int = 0
    maps: int = 0
    presets: int = 0
    exports: int = 0
    missing: list[str] = field(default_factory=list)          # fotos o exportaciones que ya no existían
    modules: list[str] = field(default_factory=list)          # módulos de Look que usa el trabajo


def _free_dir(parent: Path, name: str) -> Path:
    safe = presets.safe_name(name) or "proyecto"
    folder, n = parent / safe, 2
    while folder.exists():
        folder = parent / f"{safe}_{n}"
        n += 1
    return folder


def _unique(folder: Path, name: str, taken: set[str]) -> Path:
    """Un nombre de archivo que no se repite dentro de la carpeta (fotos de distintas carpetas con el mismo nombre)."""
    path, n = folder / name, 2
    while str(path).lower() in taken:
        path = folder / f"{Path(name).stem}_{n}{Path(name).suffix}"
        n += 1
    taken.add(str(path).lower())
    return path


def used_modules(entries: list[PhotoEntry]) -> list[str]:
    """Los módulos de Look (capas «module.<id>») que usan los ajustes de las fotos, sin repetir."""
    found: list[str] = []
    for e in entries:
        for stack in ("pre", "look"):
            for layer in (e.project or {}).get(stack, []) or []:
                kind = str(layer.get("kind", ""))
                if kind.startswith("module.") and kind.removeprefix("module.") not in found:
                    found.append(kind.removeprefix("module."))
    return found


def collect_files(parent: Path, name: str, entries: list[PhotoEntry], current: int = 0, app_version: str = "",
                  progress: Progress = lambda _m, _f: None) -> CollectResult:
    """Crea `parent/name/` con todo (ver arriba). `entries` ya lleva los mapas, los ajustes, las máscaras y, en
    `exports`, lo exportado de cada foto. Devuelve lo que se copió y lo que faltaba."""
    folder = _free_dir(Path(parent), name)
    for sub in (PHOTOS, MAPS):
        (folder / sub).mkdir(parents=True)
    result = CollectResult(folder, folder / f"{folder.name}{project_file.EXTENSION}", modules=used_modules(entries))
    total = max(len(entries), 1)
    taken_photos: set[str] = set()
    taken_maps: set[str] = set()
    new_entries: list[PhotoEntry] = []
    for i, e in enumerate(entries):
        progress(QCoreApplication.translate("Collect", "Copiando {0}…").format(Path(e.path).name), i / total * 0.8)
        source = Path(e.path)
        copied = e
        if source.is_file():
            target = _unique(folder / PHOTOS, source.name, taken_photos)
            shutil.copy2(source, target)
            copied = PhotoEntry(**{**e.__dict__, "path": str(target), "name": target.name, "embedded": ""})
            result.photos += 1
        else:
            result.missing.append(source.name)
        if e.depth is not None:
            target = _unique(folder / MAPS, f"{source.stem}_depth.png", taken_maps)
            target.write_bytes(project_file._depth_to_png(e.depth))
            result.maps += 1
        exports = [Path(p) for p in e.exports if Path(p).is_file()]
        result.missing += [Path(p).name for p in e.exports if not Path(p).is_file()]
        if exports:
            out = folder / EXPORTS / (presets.safe_name(source.stem) or f"foto_{i + 1}")
            out.mkdir(parents=True, exist_ok=True)
            taken: set[str] = set()
            for path in exports:
                shutil.copy2(path, _unique(out, path.name, taken))
                result.exports += 1
        new_entries.append(PhotoEntry(**{**copied.__dict__, "exports": []}))      # lo exportado ya va en exports/
    progress(QCoreApplication.translate("Collect", "Copiando los presets propios…"), 0.85)
    for kind in presets.KINDS:
        for preset_name in presets.list_user(kind):
            target = folder / FX / kind
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(presets.user_path(kind, preset_name), target / presets.user_path(kind, preset_name).name)
            result.presets += 1
    progress(QCoreApplication.translate("Collect", "Guardando el proyecto…"), 0.92)
    project_file.save_project(result.project, new_entries, current, app_version)
    _write_readme(folder, result)
    progress(QCoreApplication.translate("Collect", "Listo"), 1.0)
    log.info("Archivos recopilados en %s", folder)
    return result


def _write_readme(folder: Path, result: CollectResult) -> None:
    lines = [QCoreApplication.translate("Collect", "Deepshot Studio · archivos recopilados"), "",
             QCoreApplication.translate("Collect", "{0}: el proyecto (ábrelo con Archivo › Abrir proyecto).").format(result.project.name),
             QCoreApplication.translate("Collect", "{0}/: las fotos originales ({1}).").format(PHOTOS, result.photos),
             QCoreApplication.translate("Collect", "{0}/: el mapa de profundidad de cada foto, PNG de 16 bits ({1}).").format(MAPS, result.maps),
             QCoreApplication.translate("Collect", "{0}/: los presets propios ({1}); cópialos a la carpeta de presets del otro equipo.")
             .format(FX, result.presets),
             QCoreApplication.translate("Collect", "{0}/: lo exportado de cada foto ({1}).").format(EXPORTS, result.exports)]
    if result.modules:
        lines += ["", QCoreApplication.translate("Collect", "Módulos de Look que usa el trabajo (instálalos en el gestor si faltan): {0}")
                  .format(", ".join(result.modules))]
    if result.missing:
        lines += ["", QCoreApplication.translate("Collect", "No se encontraron (ya no estaban en su sitio): {0}").format(", ".join(result.missing))]
    (folder / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
