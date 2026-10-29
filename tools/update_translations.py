"""Actualiza las traducciones: extrae los textos del código y compila el inglés.

Uso (desde la carpeta del proyecto, con el entorno activado):
    python tools/update_translations.py            # extrae (lupdate) y compila (lrelease)
    python tools/update_translations.py --extract  # solo extrae: deja el .ts al día para traducir
    python tools/update_translations.py --compile  # solo compila el .qm desde el .ts

`resources/translations/deepshot_en.ts` es el archivo fuente (se edita con Qt Linguist o a mano); el `.qm` es el
que carga la aplicación. Los textos nuevos aparecen en el .ts sin traducir ("unfinished") hasta que se traducen.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS = ROOT / "resources" / "translations" / "deepshot_en.ts"
QM = TS.with_suffix(".qm")


def tool(name: str) -> str:
    """Ejecutable de PySide6 (lupdate, lrelease) del entorno que ejecuta este script."""
    found = shutil.which(name, path=str(Path(sys.executable).parent)) or shutil.which(name)
    if not found:
        raise SystemExit(f"No se encuentra «{name}». ¿Está instalado PySide6 en este entorno?")
    return found


def extract() -> None:
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "deepshot").rglob("*.py"))
    TS.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("pyside6-lupdate"), *files, "-ts", str(TS.relative_to(ROOT)), "-no-obsolete"], cwd=ROOT, check=True)


def compile_qm() -> None:
    subprocess.run([tool("pyside6-lrelease"), str(TS.relative_to(ROOT)), "-qm", str(QM.relative_to(ROOT))],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if not args or "--extract" in args:
        extract()
    if not args or "--compile" in args:
        compile_qm()
