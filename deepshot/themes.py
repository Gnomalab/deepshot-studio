"""Temas de color y aspecto de la interfaz (sin interfaz gráfica: solo datos y texto).

Un tema es una paleta con nombre (`colors`): la hoja de estilos `resources/theme/theme.qss` es una plantilla con
marcas ${nombre} que se rellenan con esos colores y con las medidas del «aspecto» (redondeo y densidad).

De fábrica hay tres temas: «dark» (oscuro, estilo Blender), «light» (claro) y «midnight» (oscuro azulado). El usuario
puede añadir los suyos como archivos JSON en `<datos>/themes/`:

    {"name": "Mi tema", "base": "dark", "colors": {"accent": "#c0503a", "window": "#262626"}}

`base` es el tema de fábrica del que parte (los colores que falten salen de él); si falta, «dark».
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from PySide6.QtCore import QT_TRANSLATE_NOOP

from deepshot.paths import data_subdir, resource

log = logging.getLogger(__name__)

# Los colores que tiene que dar todo tema (la plantilla y los widgets pintados a mano los usan).
COLOR_KEYS = (
    "window", "text", "text_dim", "text_faint", "text_strong", "text_disabled", "line", "panel", "header", "popup", "base",
    "base_hover", "button", "button_hover", "button_disabled", "outline", "accent", "accent_hover", "accent_disabled",
    "accent_disabled_text", "on_accent", "section_hover", "item_hover", "scroll", "scroll_hover", "link", "ok", "warn", "bad",
    "icon_idle", "icon_active", "icon_disabled", "viewer",
)

DARK = {
    "window": "#303030", "text": "#e5e5e5", "text_dim": "#9a9a9a", "text_faint": "#8a8a8a", "text_strong": "#f0f0f0",
    "text_disabled": "#7a7a7a", "line": "#1d1d1d", "panel": "#3d3d3d", "header": "#2b2b2b", "popup": "#181818",
    "base": "#282828", "base_hover": "#2f2f2f", "button": "#545454", "button_hover": "#656565",
    "button_disabled": "#474747", "outline": "#3d3d3d", "accent": "#4772b3", "accent_hover": "#5680c2",
    "accent_disabled": "#3a4a63", "accent_disabled_text": "#8a93a3", "on_accent": "#ffffff", "section_hover": "#454545",
    "item_hover": "#333333", "scroll": "#4f4f4f", "scroll_hover": "#606060", "link": "#7fa6e0", "ok": "#9bd29b",
    "warn": "#d9a03f", "bad": "#e06c5f", "icon_idle": "#aeaeae", "icon_active": "#ffffff", "icon_disabled": "#6a6a6a", "viewer": "#1d1d1d",
}
LIGHT = {
    "window": "#dcdcdc", "text": "#202020", "text_dim": "#555555", "text_faint": "#666666", "text_strong": "#000000",
    "text_disabled": "#9a9a9a", "line": "#b9b9b9", "panel": "#ececec", "header": "#cdcdcd", "popup": "#f6f6f6",
    "base": "#ffffff", "base_hover": "#f1f1f1", "button": "#d0d0d0", "button_hover": "#c0c0c0",
    "button_disabled": "#e3e3e3", "outline": "#b4b4b4", "accent": "#4772b3", "accent_hover": "#3a62a0",
    "accent_disabled": "#a9bad6", "accent_disabled_text": "#f0f3f9", "on_accent": "#ffffff", "section_hover": "#e0e0e0",
    "item_hover": "#e6e6e6", "scroll": "#a8a8a8", "scroll_hover": "#8f8f8f", "link": "#2f5da8", "ok": "#2f7d32",
    "warn": "#a56a00", "bad": "#b3261e", "icon_idle": "#4a4a4a", "icon_active": "#000000", "icon_disabled": "#a8a8a8", "viewer": "#c4c4c4",
}
MIDNIGHT = {
    **DARK, "window": "#232a36", "panel": "#2d3647", "header": "#1c222c", "popup": "#151a22", "base": "#1b212b",
    "base_hover": "#222a36", "button": "#3b475c", "button_hover": "#4a5873", "button_disabled": "#333d4f",
    "outline": "#2d3647", "line": "#141922", "section_hover": "#36415a", "item_hover": "#2a3342", "scroll": "#46536b",
    "scroll_hover": "#586887", "accent": "#4a78c2", "accent_hover": "#5a88d2", "accent_disabled": "#3c4f73",
    "viewer": "#141922",
}


@dataclass(frozen=True)
class Theme:
    key: str                          # identificador (clave de fábrica o nombre del archivo del usuario)
    label: str                        # nombre que se ve
    colors: dict[str, str] = field(default_factory=dict)
    factory: bool = True

    def color(self, name: str) -> str:
        return self.colors.get(name) or DARK[name]


FACTORY_THEMES: dict[str, Theme] = {
    "dark": Theme("dark", QT_TRANSLATE_NOOP("Themes", "Oscuro (estilo Blender)"), DARK),
    "light": Theme("light", QT_TRANSLATE_NOOP("Themes", "Claro"), LIGHT),
    "midnight": Theme("midnight", QT_TRANSLATE_NOOP("Themes", "Medianoche (oscuro azulado)"), MIDNIGHT),
}
DEFAULT_THEME = "dark"


@dataclass(frozen=True)
class Appearance:
    """Aspecto de los controles, independiente del tema de color."""

    radius: int = 3                   # redondeo de botones, campos y listas, en píxeles (0 = esquinas rectas)
    density: str = "standard"         # "compact" | "standard" | "comfortable": cuánto aire llevan los controles
    icons: str = "color"              # "color": un color por tipo de icono | "mono": una sola tinta, más sobrio

    def validated(self) -> "Appearance":
        return Appearance(radius=min(max(int(self.radius), 0), 10),
                          density=self.density if self.density in DENSITIES else "standard",
                          icons=self.icons if self.icons in ICON_STYLES else "color")


ICON_STYLES = ("color", "mono")

# Medidas por densidad: relleno vertical de botones y campos, alto mínimo, relleno de filas y de menús.
DENSITIES = {
    "compact": {"pad_v": 1, "field_v": 1, "min_h": 14, "row_pad": 1, "menu_pad": 2, "step_v": 2},
    "standard": {"pad_v": 3, "field_v": 3, "min_h": 16, "row_pad": 3, "menu_pad": 4, "step_v": 4},
    "comfortable": {"pad_v": 6, "field_v": 6, "min_h": 20, "row_pad": 6, "menu_pad": 7, "step_v": 7},
}
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def tokens(theme: Theme, appearance: Appearance) -> dict[str, str]:
    """Todas las marcas de la plantilla: colores del tema y medidas del aspecto."""
    a = appearance.validated()
    out = {k: theme.color(k) for k in COLOR_KEYS}
    out.update({k: str(v) for k, v in DENSITIES[a.density].items()})
    out.update(radius=str(a.radius), radius_big=str(a.radius + 1), pad_h="10", field_h="6")
    return out


def build_qss(theme: Theme, appearance: Appearance | None = None) -> str:
    """La hoja de estilos de la aplicación para este tema y aspecto."""
    template = resource("theme", "theme.qss").read_text(encoding="utf-8")
    return Template(template).substitute(tokens(theme, appearance or Appearance()))


# ---------------------------------------------------------------------------
# Temas del usuario (archivos JSON)
# ---------------------------------------------------------------------------
def themes_dir() -> Path:
    return data_subdir("themes")


def _theme_from_file(path: Path) -> Theme | None:
    """Un tema del usuario o None si el archivo no vale (se registra, no se rompe la lista)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        base = FACTORY_THEMES.get(str(data.get("base", DEFAULT_THEME)), FACTORY_THEMES[DEFAULT_THEME])
        colors = dict(base.colors)
        for key, value in dict(data.get("colors", {})).items():
            if key in COLOR_KEYS and isinstance(value, str) and HEX.match(value):
                colors[key] = value
        name = str(data.get("name") or path.stem).strip() or path.stem
        return Theme(f"user:{path.stem}", name, colors, factory=False)
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        log.warning("Tema no válido %s: %s", path, exc)
        return None


def user_themes() -> list[Theme]:
    folder = themes_dir()
    return [t for t in (_theme_from_file(p) for p in sorted(folder.glob("*.json"))) if t is not None]


def all_themes() -> list[Theme]:
    return [*FACTORY_THEMES.values(), *user_themes()]


def get_theme(key: str) -> Theme:
    """El tema con esa clave (de fábrica o del usuario); si ya no existe, el de por defecto."""
    for theme in all_themes():
        if theme.key == key:
            return theme
    return FACTORY_THEMES[DEFAULT_THEME]


def save_user_theme(name: str, base: str, colors: dict[str, str]) -> Path:
    """Guarda un tema propio como JSON en `<datos>/themes/`."""
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")[:60]
    if not safe:
        raise ValueError("El nombre del tema está vacío.")
    path = themes_dir() / f"{safe}.json"
    clean = {k: v for k, v in colors.items() if k in COLOR_KEYS and HEX.match(v)}
    path.write_text(json.dumps({"name": name.strip(), "base": base, "colors": clean}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path
