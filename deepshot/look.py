"""Look: pila de capas de ajuste que se aplica al final, como las capas de ajuste de Photoshop (sin interfaz).

Cada capa recibe la imagen que dejó la anterior, así que el ORDEN importa: niveles y luego saturación no da lo mismo que
saturación y luego niveles. Cada capa tiene ojo (enabled), opacidad, modo de fusión y un límite opcional por rango de
profundidad («solo en el fondo»).

De serie hay tres capas: niveles, contraste y saturación. El registro `KINDS` admite más: los módulos descargables
(deepshot/modules.py) añaden los suyos. Cada tipo da un nombre, la lista de sus ajustes (`ParamSpec`, con rango y valor
inicial: la interfaz dibuja los controles a partir de ella) y la función que los aplica.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PySide6.QtCore import QT_TRANSLATE_NOOP

from deepshot.compose import BLEND_MODES, blend, depth_range_mask

log = logging.getLogger(__name__)

LIMITS = ("all", "depth")                       # a qué parte de la imagen afecta una capa
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# ---------------------------------------------------------------------------
# Funciones de ajuste (imagen float32 alto x ancho x 3 en 0..1 -> misma forma)
# ---------------------------------------------------------------------------
def levels(image: np.ndarray, black: float = 0.0, white: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Niveles: negros y blancos de entrada y gamma (>1 aclara los medios tonos, como en Photoshop)."""
    span = max(white - black, 1e-4)
    out = np.clip((image - black) / span, 0.0, 1.0)
    if gamma != 1.0:
        out = out ** (1.0 / max(gamma, 1e-3))
    return out.astype(np.float32)


def brightness_contrast(image: np.ndarray, brightness: float = 0.0, contrast: float = 0.0) -> np.ndarray:
    """Brillo y contraste, ambos en -1..1. El contraste gira los tonos alrededor del gris medio."""
    c = float(np.clip(contrast, -1.0, 1.0))
    factor = 1.0 + c if c < 0 else 1.0 / max(1.0 - c, 1e-3)
    out = (image - 0.5) * factor + 0.5 + brightness * 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def saturation(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Saturación en -1..1: -1 = blanco y negro, 0 = igual, 1 = el doble de color."""
    gray = (image @ LUMA)[..., None]
    out = gray + (image - gray) * (1.0 + amount)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class ParamSpec:
    """Un ajuste de una capa: nombre, rango y valor inicial. `label` y `tooltip` se traducen con el contexto «Look»
    (los de un módulo van tal cual, en el idioma en que los escribió su autor)."""

    name: str
    label: str
    default: float = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    decimals: int = 2
    step: float = 0.01
    tooltip: str = ""
    drag_range: float | None = None             # cuánto cambia al arrastrar todo el ancho (por defecto, el rango)


@dataclass(frozen=True)
class LayerKind:
    """Un tipo de capa: su nombre, sus ajustes y cómo se aplica."""

    key: str
    title: str                                   # texto traducible (contexto «Look»)
    params: tuple[ParamSpec, ...]
    apply: Callable[[np.ndarray, "np.ndarray | None", dict], np.ndarray]    # apply(imagen, profundidad, ajustes)
    module_id: str = ""                          # "" = de serie; si no, el id del módulo que lo trae

    @property
    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}


KINDS: dict[str, LayerKind] = {k.key: k for k in (
    LayerKind("levels", QT_TRANSLATE_NOOP("Look", "Niveles"), (
        ParamSpec("black", QT_TRANSLATE_NOOP("Look", "Negros"), 0.0, 0.0, 1.0, 3, 0.005,
                  QT_TRANSLATE_NOOP("Look", "Valor de entrada que pasa a negro puro: sube los negros")),
        ParamSpec("gamma", QT_TRANSLATE_NOOP("Look", "Gamma"), 1.0, 0.2, 4.0, 2, 0.02,
                  QT_TRANSLATE_NOOP("Look", "Medios tonos: más de 1 aclara, menos de 1 oscurece, sin tocar negros ni blancos"),
                  drag_range=3.0),
        ParamSpec("white", QT_TRANSLATE_NOOP("Look", "Blancos"), 1.0, 0.0, 1.0, 3, 0.005,
                  QT_TRANSLATE_NOOP("Look", "Valor de entrada que pasa a blanco puro: baja los blancos")),
    ), lambda img, _depth, p: levels(img, p["black"], p["white"], p["gamma"])),
    LayerKind("contrast", QT_TRANSLATE_NOOP("Look", "Contraste"), (
        ParamSpec("brightness", QT_TRANSLATE_NOOP("Look", "Brillo"), 0.0, -1.0, 1.0, 2, 0.01,
                  QT_TRANSLATE_NOOP("Look", "Aclara (+) u oscurece (−) toda la imagen")),
        ParamSpec("contrast", QT_TRANSLATE_NOOP("Look", "Contraste"), 0.0, -1.0, 1.0, 2, 0.01,
                  QT_TRANSLATE_NOOP("Look", "Separa (+) o junta (−) los tonos alrededor del gris medio")),
    ), lambda img, _depth, p: brightness_contrast(img, p["brightness"], p["contrast"])),
    LayerKind("saturation", QT_TRANSLATE_NOOP("Look", "Saturación"), (
        ParamSpec("amount", QT_TRANSLATE_NOOP("Look", "Saturación"), 0.0, -1.0, 1.0, 2, 0.01,
                  QT_TRANSLATE_NOOP("Look", "−1 = blanco y negro, 0 = como está, +1 = el doble de color")),
    ), lambda img, _depth, p: saturation(img, p["amount"])),
)}


def register_kind(kind: LayerKind) -> None:
    """Añade (o sustituye) un tipo de capa. Los tipos de serie no se pueden pisar."""
    current = KINDS.get(kind.key)
    if current is not None and not current.module_id:
        raise ValueError(f"No se puede sustituir el tipo de capa de serie: {kind.key}")
    KINDS[kind.key] = kind


def unregister_module_kinds(module_id: str | None = None) -> None:
    """Quita los tipos que trajo un módulo (o todos los de módulos si no se da id)."""
    for key in [k for k, v in KINDS.items() if v.module_id and (module_id is None or v.module_id == module_id)]:
        del KINDS[key]


@dataclass
class LookLayer:
    kind: str = "levels"
    enabled: bool = True
    opacity: float = 1.0
    blend: str = "normal"                   # ver compose.BLEND_MODES
    limit: str = "all"                      # "all" | "depth"
    depth_near: float = 0.0                 # para limit = "depth": rango de profundidad al que afecta
    depth_far: float = 1.0
    depth_softness: float = 0.1
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Los ajustes que falten (proyecto antiguo o capa nueva) toman su valor inicial. Si el tipo no está instalado
        # (un módulo que se quitó), la capa se conserva tal cual para no perder el trabajo, y no hace nada.
        if self.kind in KINDS:
            self.params = {**KINDS[self.kind].defaults, **{k: v for k, v in self.params.items()
                                                           if k in KINDS[self.kind].defaults}}

    @classmethod
    def new(cls, kind: str) -> "LookLayer":
        if kind not in KINDS:
            raise ValueError(f"Tipo de capa desconocido: {kind}")
        return cls(kind=kind)

    @property
    def missing(self) -> bool:
        """True si el tipo de la capa no está disponible (su módulo no está instalado)."""
        return self.kind not in KINDS

    def is_identity(self) -> bool:
        """True si la capa no cambia nada (apagada, opacidad 0, sin su módulo o ajustes en su valor inicial)."""
        if not self.enabled or self.opacity <= 0 or self.missing:
            return True
        return self.params == KINDS[self.kind].defaults

    def validate(self) -> None:
        if self.blend not in BLEND_MODES:
            raise ValueError(f"Modo de fusión desconocido: {self.blend}")
        if self.limit not in LIMITS:
            raise ValueError(f"Límite desconocido: {self.limit}")


def layers_from_list(data: list | None) -> list[LookLayer]:
    """Capas desde el JSON del proyecto. Se ignoran las claves desconocidas; los tipos que no están instalados se
    conservan (aparecen como «módulo no instalado») para no perder el trabajo al volver a guardar."""
    names = LookLayer.__dataclass_fields__
    out = []
    for item in data or []:
        if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
            continue
        out.append(LookLayer(**{k: v for k, v in item.items() if k in names}))
    return out


def apply_look(image: np.ndarray, layers: list[LookLayer], depth: np.ndarray | None = None) -> np.ndarray:
    """Aplica la pila en orden (la primera de la lista va primero). `depth` = profundidad ya movida por el control."""
    for layer in layers:
        if layer.is_identity():
            continue
        try:
            adjusted = KINDS[layer.kind].apply(image, depth, dict(layer.params))
            adjusted = np.clip(np.asarray(adjusted, dtype=np.float32), 0.0, 1.0)
            if adjusted.shape != image.shape:
                raise ValueError("la capa devolvió una imagen de otro tamaño")
        except Exception:                           # un módulo que falla no debe romper la vista previa ni la exportación
            log.exception("La capa «%s» falló y se omite", layer.kind)
            continue
        mask = None
        if layer.limit == "depth" and depth is not None:
            mask = depth_range_mask(depth, layer.depth_near, layer.depth_far, layer.depth_softness)
        image = blend(image, adjusted, layer.blend, layer.opacity, mask)
    return image
