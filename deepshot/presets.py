"""Presets de rampa y de estilo: los de fábrica y los que guarda el usuario.

Cuatro tipos (`kind`):
  - "look":  la pila de efectos del paso Look (niveles, contraste, grano, resplandor…). Los de fábrica usan solo los
             efectos de serie (numpy), nunca módulos descargables: así valen en cualquier instalación.
  - "lens":  solo el carácter del bokeh del desenfoque de lente (apertura, palas, ojo de gato…). No toca el foco ni
             el destino, que dependen de cada foto, y al aplicarlo enciende el desenfoque.
  - "ramp":  solo la rampa de color (paradas e interpolación).
  - "style": un aspecto completo (rampa, base, bandas, mezcla con la foto y, si lo traen,
             áreas). No toca el relieve, la cámara ni la máscara pintada de la foto.

Los de usuario son archivos JSON en `<datos>/presets/<kind>/<nombre>.json`: se pueden copiar
a otro equipo o compartir. Los de fábrica viven aquí, en código, y no se pueden borrar.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication

from deepshot.areas import Area
from deepshot.dof import LensBlur
from deepshot.paths import data_subdir
from deepshot.tonemap import BLACK, WHITE, ColorRamp, RampStop, ramp_preset

log = logging.getLogger(__name__)

KINDS = ("ramp", "style", "lens", "look")
PRESET_VERSION = 1


def _ramp(stops: list[tuple[float, tuple[float, float, float]]], interpolation: str = "linear") -> ColorRamp:
    return ColorRamp([RampStop(p, c) for p, c in stops], interpolation)


# ---------------------------------------------------------------------------
# De fábrica
# ---------------------------------------------------------------------------
# (clave, nombre visible, rampa). Los nombres se traducen al mostrarlos (ver `builtin_*`).
_RAMPS: list[tuple[str, str, ColorRamp]] = [
    ("linear", QT_TRANSLATE_NOOP("Presets", "Lineal"), ramp_preset("linear")),
    ("haze", QT_TRANSLATE_NOOP("Presets", "Neblina (haze)"), ramp_preset("haze")),
    ("contrast", QT_TRANSLATE_NOOP("Presets", "Contraste fuerte"), ramp_preset("contrast")),
    ("color", QT_TRANSLATE_NOOP("Presets", "Color (rojo / azul)"), ramp_preset("color")),
    ("xray", QT_TRANSLATE_NOOP("Presets", "Inverso (cerca oscuro)"), _ramp([(0.0, BLACK), (1.0, WHITE)])),
    ("silhouette", QT_TRANSLATE_NOOP("Presets", "Silueta (cerca negro, lejos luz)"),
     _ramp([(0.0, (0.02, 0.02, 0.02)), (0.45, (0.10, 0.10, 0.10)), (0.8, (0.85, 0.85, 0.85)),
            (1.0, WHITE)], "ease")),
    # Mapas de profundidad «de calle»: lo cercano casi negro, el plano medio muy luminoso y el fondo en niebla clara.
    ("street", QT_TRANSLATE_NOOP("Presets", "Calle (cerca negro, fondo en niebla)"),
     _ramp([(0.0, (0.0, 0.0, 0.0)), (0.10, (0.02, 0.02, 0.02)), (0.35, (0.40, 0.40, 0.40)),
            (0.65, (0.68, 0.68, 0.68)), (1.0, (0.90, 0.90, 0.90))], "ease")),
    ("soft_map", QT_TRANSLATE_NOOP("Presets", "Mapa suave (cerca oscuro)"),
     _ramp([(0.0, (0.03, 0.03, 0.03)), (0.5, (0.55, 0.55, 0.55)), (1.0, (0.96, 0.96, 0.96))], "ease")),
    ("dense_fog", QT_TRANSLATE_NOOP("Presets", "Niebla densa (lejos blanco)"),
     _ramp([(0.0, (0.05, 0.05, 0.05)), (0.3, (0.25, 0.25, 0.25)), (0.6, (0.80, 0.80, 0.80)),
            (1.0, WHITE)], "ease")),
    ("sunset", QT_TRANSLATE_NOOP("Presets", "Atardecer (cálido)"),
     _ramp([(0.0, (0.08, 0.03, 0.10)), (0.35, (0.60, 0.15, 0.25)), (0.7, (0.98, 0.55, 0.20)),
            (1.0, (1.0, 0.92, 0.65))], "ease")),
    ("thermal", QT_TRANSLATE_NOOP("Presets", "Térmico"),
     _ramp([(0.0, (1.0, 1.0, 0.85)), (0.25, (0.98, 0.75, 0.15)), (0.55, (0.85, 0.15, 0.20)),
            (0.8, (0.30, 0.05, 0.45)), (1.0, (0.02, 0.0, 0.10))])),
    ("ocean", QT_TRANSLATE_NOOP("Presets", "Océano"),
     _ramp([(0.0, (0.85, 0.97, 1.0)), (0.4, (0.20, 0.65, 0.80)), (1.0, (0.02, 0.10, 0.30))], "ease")),
    ("posterize", QT_TRANSLATE_NOOP("Presets", "Cartel (3 tonos)"),
     _ramp([(0.0, WHITE), (0.34, (0.55, 0.55, 0.55)), (0.67, (0.20, 0.20, 0.20)), (1.0, BLACK)], "constant")),
]

# Estilos completos: diccionarios parciales de Project (ver project.Project.to_dict).
_STYLES: list[tuple[str, str, dict]] = [
    ("haze", QT_TRANSLATE_NOOP("Presets", "Neblina atmosférica"), {"ramp": "haze", "base": "ramp"}),
    ("contrast", QT_TRANSLATE_NOOP("Presets", "Contraste fuerte (B/N)"), {"ramp": "contrast", "base": "ramp"}),
    ("color", QT_TRANSLATE_NOOP("Presets", "Color de profundidad"), {"ramp": "color", "base": "ramp"}),
    ("silhouette", QT_TRANSLATE_NOOP("Presets", "Silueta fotográfica"), {
        "ramp": "silhouette", "base": "ramp",
        "composite": {"mode": "soft_light", "opacity": 0.25, "photo_grayscale": True}}),
    ("photo_bw", QT_TRANSLATE_NOOP("Presets", "Fotografía B/N con profundidad"), {
        "ramp": "haze", "base": "ramp",
        "composite": {"mode": "multiply", "opacity": 0.65, "photo_grayscale": True}}),
    ("photo_color", QT_TRANSLATE_NOOP("Presets", "Foto en color con niebla"), {
        "ramp": "haze", "base": "ramp",
        "composite": {"mode": "overlay", "opacity": 0.55, "photo_grayscale": False}}),
    ("studio_light", QT_TRANSLATE_NOOP("Presets", "Foco de luz sobre negro"), {
        "ramp": "linear", "base": "black",
        "areas": [{"name": "Foco", "shape": "ellipsoid", "center": [0.0, 0.1, 0.35], "scale": [0.7, 0.7, 0.5],
                   "strength": 1.0, "edge_softness": 0.9, "falloff": 0.6}],
        "composite": {"mode": "soft_light", "opacity": 0.4, "photo_grayscale": True}}),
    # Mapas de profundidad en gris con el aspecto de las referencias de calle (cerca negro, lejos en niebla).
    ("depth_street", QT_TRANSLATE_NOOP("Presets", "Mapa de calle (cerca negro)"), {"ramp": "street", "base": "ramp"}),
    ("depth_street_detail", QT_TRANSLATE_NOOP("Presets", "Mapa de calle con detalle de la foto"), {
        "ramp": "street", "base": "ramp",
        "composite": {"mode": "soft_light", "opacity": 0.35, "photo_grayscale": True}}),
    ("depth_soft", QT_TRANSLATE_NOOP("Presets", "Mapa de profundidad suave"), {"ramp": "soft_map", "base": "ramp"}),
    ("depth_fog", QT_TRANSLATE_NOOP("Presets", "Mapa con niebla densa"), {"ramp": "dense_fog", "base": "ramp"}),
    ("depth_bands", QT_TRANSLATE_NOOP("Presets", "Bandas de profundidad"), {
        "ramp": "contrast", "base": "ramp",
        "bands": [{"position": 0.25, "width": 0.06, "softness": 0.04, "color": [1.0, 1.0, 1.0], "strength": 1.0},
                  {"position": 0.55, "width": 0.06, "softness": 0.04, "color": [1.0, 1.0, 1.0], "strength": 0.8}]}),
    # Los siguientes enseñan funciones que los anteriores no usan: color de banda tomado de la rampa, áreas en caja y
    # con fuerza negativa, base blanca, desenfoque de lente y máscara retocada.
    ("color_bands", QT_TRANSLATE_NOOP("Presets", "Bandas con el color de la rampa"), {
        "ramp": "sunset", "base": "black",
        "bands": [{"position": 0.2, "width": 0.07, "softness": 0.05, "strength": 1.0, "color_from_ramp": True},
                  {"position": 0.5, "width": 0.07, "softness": 0.05, "strength": 1.0, "color_from_ramp": True},
                  {"position": 0.8, "width": 0.07, "softness": 0.05, "strength": 1.0, "color_from_ramp": True}]}),
    ("dark_spot", QT_TRANSLATE_NOOP("Presets", "Sombra central (área negativa)"), {
        "ramp": "haze", "base": "ramp",
        "areas": [{"name": "Sombra", "shape": "ellipsoid", "center": [0.0, 0.0, 0.4], "scale": [0.6, 0.6, 0.6],
                   "strength": -0.7, "edge_softness": 1.0, "falloff": 0.8}]}),
    ("light_slab", QT_TRANSLATE_NOOP("Presets", "Losa de luz (área en caja)"), {
        "ramp": "linear", "base": "black",
        "areas": [{"name": "Losa", "shape": "box", "center": [0.0, 0.0, 0.5], "scale": [2.0, 2.0, 0.12],
                   "strength": 1.0, "edge_softness": 0.35, "falloff": 0.0}],
        "composite": {"mode": "overlay", "opacity": 0.3, "photo_grayscale": True}}),
    ("white_veil", QT_TRANSLATE_NOOP("Presets", "Velo blanco con bandas oscuras"), {
        "ramp": "soft_map", "base": "white",
        "bands": [{"position": 0.3, "width": 0.1, "softness": 0.08, "color": [0.0, 0.0, 0.0], "strength": 0.7},
                  {"position": 0.7, "width": 0.1, "softness": 0.08, "color": [0.0, 0.0, 0.0], "strength": 0.7}]}),
    ("poster", QT_TRANSLATE_NOOP("Presets", "Cartel de profundidad (3 tonos)"), {"ramp": "posterize", "base": "ramp"}),
    ("sunset_photo", QT_TRANSLATE_NOOP("Presets", "Atardecer sobre la foto"), {
        "ramp": "sunset", "base": "ramp",
        "composite": {"mode": "soft_light", "opacity": 0.6, "photo_grayscale": False}}),
    ("portrait_depth", QT_TRANSLATE_NOOP("Presets", "Retrato: sujeto nítido, fondo desenfocado"), {
        "ramp": "haze", "base": "ramp",
        "composite": {"mode": "overlay", "opacity": 0.6, "photo_grayscale": False, "mask": "depth_range",
                      "invert_mask": True, "depth_near": 0.0, "depth_far": 0.4, "depth_softness": 0.1,
                      "mask_feather": 0.3, "mask_snap": 0.6},
        "lens": {"enabled": True, "target": "photo", "focus": 0.2, "sharp_range": 0.06, "aperture": 1.8,
                 "near": 0.8, "far": 1.2, "blades": 0, "highlights": 0.45, "softness": 0.4}}),
    # Para entender el desenfoque de lente: la foto sola (sin mezcla con el efecto) con un foco claro y mucho desenfoque.
    ("lens_demo", QT_TRANSLATE_NOOP("Presets", "Desenfoque de lente: foto con foco y fondo borroso"), {
        "ramp": "linear", "base": "ramp",
        "composite": {"mode": "normal", "opacity": 1.0, "photo_grayscale": False},
        "lens": {"enabled": True, "target": "photo", "focus": 0.25, "sharp_range": 0.05, "aperture": 1.4,
                 "near": 1.2, "far": 1.5, "blades": 6, "highlights": 0.6, "softness": 0.25}}),
]


# Bokeh de fábrica: solo el carácter del desenfoque (el foco y el destino se quedan como los tenga la foto).
_LENS: list[tuple[str, str, dict]] = [
    ("portrait", QT_TRANSLATE_NOOP("Presets", "Retrato f/1.8 (fondo suave)"),
     {"aperture": 1.8, "sharp_range": 0.05, "near": 0.8, "far": 1.2, "blades": 0, "rotation": 0.0,
      "highlights": 0.45, "softness": 0.4, "cat_eye": 0.0}),
    ("portrait_classic", QT_TRANSLATE_NOOP("Presets", "Retrato clásico (ojo de gato)"),
     {"aperture": 1.4, "sharp_range": 0.04, "near": 0.9, "far": 1.3, "blades": 0, "rotation": 0.0,
      "highlights": 0.55, "softness": 0.2, "cat_eye": 0.6}),
    ("macro", QT_TRANSLATE_NOOP("Presets", "Macro (muy poca profundidad)"),
     {"aperture": 2.0, "sharp_range": 0.02, "near": 1.5, "far": 1.5, "blades": 0, "rotation": 0.0,
      "highlights": 0.3, "softness": 0.35, "cat_eye": 0.0}),
    ("cinema", QT_TRANSLATE_NOOP("Presets", "Cine (diafragma de 7 palas)"),
     {"aperture": 2.0, "sharp_range": 0.06, "near": 1.0, "far": 1.2, "blades": 7, "rotation": 12.0,
      "highlights": 0.6, "softness": 0.15, "cat_eye": 0.35}),
    ("hard_discs", QT_TRANSLATE_NOOP("Presets", "Discos nítidos (bokeh duro)"),
     {"aperture": 2.8, "sharp_range": 0.06, "near": 1.0, "far": 1.0, "blades": 0, "rotation": 0.0,
      "highlights": 0.9, "softness": 0.0, "cat_eye": 0.0}),
    ("landscape", QT_TRANSLATE_NOOP("Presets", "Paisaje f/8 (desenfoque ligero)"),
     {"aperture": 8.0, "sharp_range": 0.15, "near": 0.6, "far": 0.8, "blades": 6, "rotation": 0.0,
      "highlights": 0.3, "softness": 0.3, "cat_eye": 0.0}),
]
# Looks de fábrica: pilas de efectos de serie (deepshot/look.py), de arriba abajo; se aplican de abajo arriba. Solo tipos de
# serie: un test vigila que ninguno use un módulo descargable.
_LOOKS: list[tuple[str, str, list[dict]]] = [
    ("cinema_warm", QT_TRANSLATE_NOOP("Presets", "Cine cálido"), [
        {"kind": "grain", "params": {"amount": 0.2, "size": 1.2}},
        {"kind": "vignette", "params": {"amount": 0.3}},
        {"kind": "contrast", "params": {"contrast": 0.2}},
        {"kind": "temperature", "params": {"temp": 0.35, "tint": 0.05}}]),
    ("cold_night", QT_TRANSLATE_NOOP("Presets", "Noche fría"), [
        {"kind": "vignette", "params": {"amount": 0.4}},
        {"kind": "glow", "params": {"amount": 0.4, "threshold": 0.75, "radius": 3.0}},
        {"kind": "saturation", "params": {"amount": -0.2}},
        {"kind": "levels", "params": {"black": 0.04, "gamma": 0.9}},
        {"kind": "temperature", "params": {"temp": -0.5}}]),
    ("hard_bw", QT_TRANSLATE_NOOP("Presets", "Blanco y negro duro"), [
        {"kind": "vignette", "params": {"amount": 0.25}},
        {"kind": "grain", "params": {"amount": 0.3}},
        {"kind": "clarity", "params": {"amount": 0.6}},
        {"kind": "contrast", "params": {"contrast": 0.35}},
        {"kind": "saturation", "params": {"amount": -1.0}}]),
    ("dreamy", QT_TRANSLATE_NOOP("Presets", "Onírico suave"), [
        {"kind": "contrast", "params": {"contrast": -0.1}},
        {"kind": "saturation", "params": {"amount": 0.15}},
        {"kind": "clarity", "params": {"amount": -0.4, "radius": 3.0}},
        {"kind": "glow", "params": {"amount": 0.8, "threshold": 0.6, "radius": 4.0, "warmth": 0.2}}]),
    ("far_fog", QT_TRANSLATE_NOOP("Presets", "Niebla en la distancia"), [
        {"kind": "temperature", "params": {"temp": -0.1}},
        {"kind": "contrast", "params": {"contrast": 0.1}},
        {"kind": "haze", "params": {"amount": 0.6, "start": 0.35, "tone": 0.85}}]),
    ("old_film", QT_TRANSLATE_NOOP("Presets", "Película antigua"), [
        {"kind": "vignette", "params": {"amount": 0.45}},
        {"kind": "grain", "params": {"amount": 0.45, "size": 1.6}},
        {"kind": "chromatic", "params": {"amount": 0.4}},
        {"kind": "temperature", "params": {"temp": 0.25, "tint": 0.08}},
        {"kind": "saturation", "params": {"amount": -0.25}},
        {"kind": "levels", "params": {"black": 0.05, "white": 0.95}}]),
    ("crisp", QT_TRANSLATE_NOOP("Presets", "Nitidez de revista"), [
        {"kind": "contrast", "params": {"contrast": 0.15}},
        {"kind": "saturation", "params": {"amount": 0.15}},
        {"kind": "sharpen", "params": {"amount": 0.8, "radius": 0.15}},
        {"kind": "clarity", "params": {"amount": 0.8}}]),
    ("matte", QT_TRANSLATE_NOOP("Presets", "Lavado mate"), [
        {"kind": "saturation", "params": {"amount": -0.3}},
        {"kind": "contrast", "params": {"contrast": -0.15}},
        {"kind": "levels", "params": {"black": 0.08, "gamma": 1.1}}]),
]

LENS_PRESET_FIELDS = ("aperture", "sharp_range", "near", "far", "blades", "rotation", "highlights", "softness",
                      "cat_eye")        # lo que guarda y aplica un preset de bokeh


def builtin_names(kind: str) -> list[tuple[str, str]]:
    """[(clave, nombre visible)] de los presets de fábrica."""
    table = {"ramp": _RAMPS, "style": _STYLES, "lens": _LENS, "look": _LOOKS}[kind]
    return [(k, name) for k, name, *_ in table]


def builtin(kind: str, key: str) -> dict:
    """Datos de un preset de fábrica como diccionario listo para aplicar."""
    if kind == "ramp":
        for k, _name, ramp in _RAMPS:
            if k == key:
                return {"ramp": ramp_to_dict(ramp)}
    elif kind == "lens":
        for k, _name, data in _LENS:
            if k == key:
                return {"lens": dict(data)}
    elif kind == "look":
        for k, _name, layers in _LOOKS:
            if k == key:
                return {"look": [json.loads(json.dumps(layer)) for layer in layers]}
    elif kind == "style":
        for k, _name, data in _STYLES:
            if k == key:
                out = dict(data)
                ramp_key = out.pop("ramp", None)
                if ramp_key:
                    out["ramp"] = builtin("ramp", ramp_key)["ramp"]
                return out
    raise KeyError(f"Preset desconocido: {kind}/{key}")


def ramp_to_dict(ramp: ColorRamp) -> dict:
    return {"stops": [{"position": s.position, "color": list(s.color)} for s in ramp.stops],
            "interpolation": ramp.interpolation}


# ---------------------------------------------------------------------------
# De usuario (archivos JSON)
# ---------------------------------------------------------------------------
def _dir(kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(QCoreApplication.translate("Presets", "Tipo de preset desconocido: {0}").format(kind))
    return data_subdir("presets") / kind


def safe_name(name: str) -> str:
    """Nombre de archivo seguro a partir del texto que escribe el usuario."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return cleaned[:80]


def user_path(kind: str, name: str) -> Path:
    return _dir(kind) / f"{safe_name(name)}.json"


def list_user(kind: str) -> list[str]:
    """Nombres de los presets del usuario, por orden alfabético."""
    folder = _dir(kind)
    if not folder.is_dir():
        return []
    return sorted((f.stem for f in folder.glob("*.json")), key=str.casefold)


def exists_user(kind: str, name: str) -> bool:
    return bool(safe_name(name)) and user_path(kind, name).is_file()


def save_user(kind: str, name: str, data: dict) -> Path:
    """Guarda (o sobrescribe) un preset del usuario. `data` = lo que devuelve `from_project`."""
    if not safe_name(name):
        raise ValueError(QCoreApplication.translate("Presets", "El nombre del preset está vacío."))
    folder = _dir(kind)
    folder.mkdir(parents=True, exist_ok=True)
    path = user_path(kind, name)
    payload = {"deepshot_preset": PRESET_VERSION, "kind": kind, "name": name.strip(), **data}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Preset guardado: %s", path)
    return path


def save_file(path: Path, kind: str, name: str, data: dict) -> None:
    """Escribe un preset en un archivo elegido por el usuario (para compartirlo o archivarlo)."""
    if kind not in KINDS:
        raise ValueError(QCoreApplication.translate("Presets", "Tipo de preset desconocido: {0}").format(kind))
    payload = {"deepshot_preset": PRESET_VERSION, "kind": kind, "name": name.strip(), **data}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Preset guardado en archivo: %s", path)


def load_file(path: Path, kind: str) -> dict:
    """Lee un archivo de preset. ValueError si no es válido o es de otro tipo."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(QCoreApplication.translate("Presets", "No se pudo leer el archivo: {0}").format(exc)) from exc
    if not isinstance(data, dict) or "deepshot_preset" not in data or data.get("kind") != kind:
        raise ValueError(QCoreApplication.translate("Presets", "El archivo no es un preset de tipo «{0}».").format(kind))
    return data


def load_user(kind: str, name: str) -> dict:
    """Lee un preset del usuario. ValueError si el archivo no es válido."""
    path = user_path(kind, name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(QCoreApplication.translate("Presets", "No se pudo leer el preset «{0}»: {1}").format(name, exc)) from exc
    if not isinstance(data, dict) or data.get("kind") != kind:
        raise ValueError(QCoreApplication.translate("Presets", "«{0}» no es un preset de tipo {1}.").format(name, kind))
    return data


def read_importable(path: Path, kind: str) -> tuple[str, dict]:
    """Lee y comprueba un archivo de preset que alguien nos ha pasado: (nombre, datos). ValueError si no vale.

    Además de que sea de este tipo, se prueba a aplicarlo sobre un proyecto vacío: un archivo a medias, con
    campos de otro tipo o con valores que no encajan se rechaza aquí y no al usarlo."""
    from deepshot.project import Project

    data = load_file(path, kind)
    try:
        Project.from_dict(apply_to_project_dict(kind, Project().to_dict(), data))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(QCoreApplication.translate("Presets", "El archivo no contiene un preset válido: {0}").format(exc)) from exc
    name = str(data.get("name") or Path(path).stem).strip()
    if not safe_name(name):
        name = safe_name(Path(path).stem) or "preset"
    return name, data


def unique_name(kind: str, name: str) -> str:
    """«nombre», o «nombre (2)», «nombre (3)»… el primero que no exista ya entre los presets del usuario."""
    if not exists_user(kind, name):
        return name
    n = 2
    while exists_user(kind, f"{name} ({n})"):
        n += 1
    return f"{name} ({n})"


def import_user(kind: str, path: Path, name: str | None = None) -> str:
    """Guarda en los presets del usuario el archivo `path` (ya comprobado con `read_importable`). Con `name`
    sobrescribe ese; sin él usa el del archivo, con número si ya existe. Devuelve el nombre con que quedó."""
    file_name, data = read_importable(path, kind)
    final = name or unique_name(kind, file_name)
    extra = {k: v for k, v in data.items() if k not in ("deepshot_preset", "kind", "name")}
    save_user(kind, final, extra)
    return final


def export_user(kind: str, name: str, path: Path) -> None:
    """Copia un preset del usuario a un archivo elegido, para compartirlo."""
    data = load_user(kind, name)
    extra = {k: v for k, v in data.items() if k not in ("deepshot_preset", "kind", "name")}
    save_file(path, kind, name, extra)


def delete_user(kind: str, name: str) -> bool:
    path = user_path(kind, name)
    try:
        path.unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# De proyecto a preset y viceversa
# ---------------------------------------------------------------------------
def from_project(kind: str, project, stack: str = "look") -> dict:
    """Extrae de un `Project` lo que guarda cada tipo de preset. `stack` (solo para «look»): la pila de efectos de la
    que sale, «look» (Post FX) o «pre» (Pre FX); un look guardado vale para las dos."""
    d = project.to_dict()
    ramp = {"stops": [{"position": s["position"], "color": list(s["color"])} for s in d["ramp"]["stops"]],
            "interpolation": d["ramp"]["interpolation"]}
    if kind == "ramp":
        return {"ramp": ramp}
    if kind == "lens":
        return {"lens": {k: d["lens"][k] for k in LENS_PRESET_FIELDS}}
    if kind == "look":
        return {"look": d["pre" if stack == "pre" else "look"]}
    composite = {k: v for k, v in d["composite"].items() if k != "painted_mask"}
    areas = [{k: v for k, v in a.items() if k != "from_style"} for a in d["areas"]]     # la marca no se guarda en el estilo
    return {"ramp": ramp, "base": d["base"], "bands": d["bands"], "areas": areas, "composite": composite,
            "lens": d["lens"]}


def area_signature(area: dict) -> str:
    """Huella del contenido de un área (sin la marca de estilo): cambia en cuanto se edita cualquier valor."""
    content = {k: v for k, v in area.items() if k != "from_style"}
    return hashlib.sha1(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _style_areas(preset_areas: list[dict]) -> list[dict]:
    """Las áreas de un estilo completas (con los valores por defecto) y marcadas como suyas con su huella."""
    out = []
    for raw in preset_areas:
        known = {k: v for k, v in raw.items() if k in Area.__dataclass_fields__ and k != "from_style"}
        full = json.loads(json.dumps(asdict(Area(**{k: tuple(v) if isinstance(v, list) else v
                                                    for k, v in known.items()}))))
        full["from_style"] = area_signature(full)
        out.append(full)
    return out


def _areas_without_old_style(areas: list[dict]) -> list[dict]:
    """Las áreas del proyecto sin las que trajo el estilo anterior y siguen tal cual. Las hechas a mano se quedan, y las
    de estilo que el usuario editó también (pasan a ser suyas: se les quita la marca)."""
    kept = []
    for area in areas:
        mark = area.get("from_style", "")
        if not mark:
            kept.append(area)
        elif area_signature(json.loads(json.dumps(area))) != mark:       # editada: ya es del usuario
            kept.append({**area, "from_style": ""})
    return kept


def apply_to_project_dict(kind: str, project_dict: dict, preset: dict, stack: str = "look") -> dict:
    """Devuelve una copia del diccionario del proyecto con el preset aplicado.

    La rampa conserva su rango (el "ajustar al relieve" de la foto). Las áreas: se quitan las que trajo el estilo
    anterior (si no se editaron), se conservan las hechas a mano y las de estilo editadas, y se añaden las del estilo
    nuevo, marcadas como suyas. Lo demás de un estilo se restablece. La máscara pintada de la foto nunca se toca.
    """
    out = json.loads(json.dumps(project_dict))   # copia profunda barata (solo tipos JSON)
    if kind == "lens":
        # Solo el carácter del bokeh; el foco, la zona nítida del foco y el destino siguen siendo los de la foto.
        known = {k: v for k, v in preset.get("lens", {}).items() if k in LENS_PRESET_FIELDS}
        out["lens"] = {**out["lens"], **known, "enabled": True}
        return out
    if kind == "look":
        # Solo la pila de efectos (se sustituye entera); el resto del proyecto no se toca. Las capas que traiga un
        # módulo que no está instalado se conservan y aparecen como «módulo no instalado». `stack` elige la pila:
        # «look» (Post FX, por defecto) o «pre» (Pre FX).
        out["pre" if stack == "pre" else "look"] = json.loads(json.dumps(preset.get("look", [])))
        return out
    if "ramp" in preset:
        out["ramp"]["stops"] = [{"position": float(s["position"]), "color": list(s["color"])}
                                for s in preset["ramp"]["stops"]]
        out["ramp"]["interpolation"] = preset["ramp"].get("interpolation", "linear")
    if kind == "style":
        # Un estilo define el aspecto completo: lo que no trae vuelve a su valor por defecto.
        out["base"] = preset.get("base", "ramp")
        out["bands"] = json.loads(json.dumps(preset.get("bands", [])))
        out["areas"] = _areas_without_old_style(out.get("areas", [])) + _style_areas(preset.get("areas", []))
        keep = out["composite"].get("painted_mask", "")
        out["composite"] = {**asdict_default_composite(), **preset.get("composite", {}), "painted_mask": keep}
        out["lens"] = {**asdict(LensBlur()), **preset.get("lens", {})}      # un estilo sin desenfoque lo apaga
    return out


def asdict_default_composite() -> dict:
    """Valores por defecto de la mezcla (un estilo parte siempre de ellos)."""
    from deepshot.project import Composite

    return asdict(Composite())
