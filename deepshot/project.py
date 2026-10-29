"""Proyecto: todos los ajustes de una imagen en JSON, y el procesado completo.

Orden del procesado (`render`):
  0. Control de profundidad: las zonas cerca / medio / lejos se mueven (deepshot/depth_control.py); lo demás lee ya ese mapa.
  1. Base: la rampa aplicada a la profundidad, o negro, o blanco (`base`).
  2. Bandas: franjas de profundidad pintadas sobre la base.
  3. Áreas: elipsoides o cajas 3D que aclaran u oscurecen.
  4. Mezcla con la foto: modo de fusión, opacidad y máscara opcional.
  5. Desenfoque de lente por profundidad (opcional): sobre el resultado final o solo sobre la foto antes de mezclarla.
  6. Look: pila de capas de ajuste (niveles, contraste, saturación…) en orden (deepshot/look.py).

Nota de diseño: el add-on de referencia tiene un ajuste "added on" para bandas y otro
para áreas. Aquí hay una sola base para toda la pila: es más predecible y equivale al
caso habitual. Se puede ampliar si hace falta.

El mismo JSON produce siempre el mismo resultado (tests/test_project.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
from PIL import Image

from deepshot.areas import Area, Scene, apply_areas, combined_mask
from deepshot.compose import BLEND_MODES, blend, depth_range_mask, load_mask, refine_mask
from deepshot.depth_control import DepthControl
from deepshot.dof import LensBlur, lens_blur
from deepshot.look import LookLayer, apply_look, layers_from_list
from deepshot.tonemap import BASES, Band, ColorRamp, RampStop, apply_bands, base_image

log = logging.getLogger(__name__)

PROJECT_VERSION = 1
MASK_SOURCES = ("none", "areas", "depth_range", "painted")


@dataclass
class Composite:
    """Mezcla del resultado con la foto original."""

    mode: str = "normal"            # ver compose.BLEND_MODES
    opacity: float = 0.0            # 0 = solo el resultado, 1 = foto con el modo elegido
    photo_grayscale: bool = False   # usar la foto en blanco y negro
    mask: str = "none"              # "none" | "areas" | "depth_range" | "painted"
    invert_mask: bool = False
    depth_near: float = 0.0         # para mask = "depth_range"
    depth_far: float = 0.5
    depth_softness: float = 0.05
    painted_mask: str = ""          # ruta del PNG pintado (relativa al JSON si es posible)
    mask_feather: float = 0.0       # 0..1: difumina la máscara (cualquiera de las tres)
    mask_snap: float = 0.0          # 0..1: pega el borde de la máscara a los bordes de la foto (guided filter)


@dataclass
class Project:
    ramp: ColorRamp = field(default_factory=ColorRamp)
    base: str = "ramp"                              # "ramp" | "black" | "white"
    bands: list[Band] = field(default_factory=list)
    scene: Scene = field(default_factory=Scene)
    areas: list[Area] = field(default_factory=list)
    composite: Composite = field(default_factory=Composite)
    lens: LensBlur = field(default_factory=LensBlur)
    depth_control: DepthControl = field(default_factory=DepthControl)
    look: list[LookLayer] = field(default_factory=list)   # pila de capas de ajuste, la primera va primero (deepshot/look.py)
    version: int = PROJECT_VERSION

    # --- JSON -----------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Reconstruye el proyecto. Las claves desconocidas se ignoran (compatibilidad)."""
        ramp_data = dict(data.get("ramp", {}))
        stops = [RampStop(s["position"], tuple(s["color"])) for s in ramp_data.pop("stops", [])]
        ramp = ColorRamp(**_known(ColorRamp, ramp_data))
        if stops:
            ramp.stops = stops
        bands = [Band(**_tuples(_known(Band, b), "color")) for b in data.get("bands", [])]
        areas = [Area(**_tuples(_known(Area, a), "center", "scale", "rotation", "ui_color")) for a in data.get("areas", [])]
        return cls(
            ramp=ramp,
            base=data.get("base", "ramp"),
            bands=bands,
            scene=Scene(**_known(Scene, data.get("scene", {}))),
            areas=areas,
            composite=Composite(**_known(Composite, data.get("composite", {}))),
            lens=LensBlur(**_known(LensBlur, data.get("lens", {}))),
            depth_control=DepthControl.from_dict(data.get("depth_control")),
            look=layers_from_list(data.get("look")),
            version=data.get("version", PROJECT_VERSION),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Project":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        """Comprueba valores con nombre (modos, formas, bases)."""
        if self.base not in BASES:
            raise ValueError(f"Base desconocida: {self.base}")
        if self.composite.mode not in BLEND_MODES:
            raise ValueError(f"Modo de fusión desconocido: {self.composite.mode}")
        if self.composite.mask not in MASK_SOURCES:
            raise ValueError(f"Máscara desconocida: {self.composite.mask}")
        self.lens.validate()
        for layer in self.look:
            layer.validate()


def _known(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _tuples(data: dict, *keys: str) -> dict:
    """JSON guarda las tuplas como listas: se vuelven a convertir."""
    return {k: (tuple(v) if k in keys and isinstance(v, list) else v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Procesado
# ---------------------------------------------------------------------------
def photo_to_float(photo: np.ndarray | Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Foto a float32 0..1 (alto, ancho, 3), reescalada a `size` = (ancho, alto) si hace falta."""
    img = photo if isinstance(photo, Image.Image) else Image.fromarray(np.asarray(photo))
    img = img.convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def render(project: Project, depth: np.ndarray, photo: np.ndarray | Image.Image | None = None,
           mask_base_dir: Path | None = None, points: np.ndarray | None = None,
           painted_mask: np.ndarray | None = None, with_look: bool = True) -> np.ndarray:
    """Procesa la imagen completa. Devuelve float32 (alto, ancho, 3) en 0..1.

    photo:  PIL o array uint8/float del mismo tamaño que `depth` (si no, se reescala).
    points: nube de puntos ya calculada para la profundidad CON el control de profundidad aplicado
            y `project.scene` (la vista previa la guarda en caché para no recalcularla en cada cambio).
    painted_mask: máscara pintada ya cargada (float32 0..1 del tamaño de `depth`); si no se da,
            se lee el PNG de `project.composite.painted_mask`.
    with_look: aplicar la pila de Look al final (la exportación «Efecto» la deja fuera; «Imagen final» la lleva).
    """
    project.validate()
    depth = np.clip(depth.astype(np.float32), 0.0, 1.0)
    depth = project.depth_control.apply(depth)       # zonas cerca / medio / lejos movidas (si las hay)

    image = base_image(project.base, project.ramp, depth)
    if project.bands:
        image = apply_bands(image, project.bands, project.ramp, depth, project.scene.relief)

    if any(a.enabled for a in project.areas):
        if points is None:
            points = project.scene.points(depth)
        image = apply_areas(image, project.areas, points)

    comp = project.composite
    if photo is not None and comp.opacity > 0:
        h, w = depth.shape
        if isinstance(photo, np.ndarray) and photo.dtype == np.float32 and photo.shape[:2] == (h, w):
            top = photo  # ya preparada (vista previa)
        else:
            top = photo_to_float(photo, (w, h))
        if project.lens.enabled and project.lens.target == "photo":
            top = lens_blur(top, depth, project.lens)            # solo la foto se desenfoca, antes de mezclarla
        guide = top                                              # la foto en color: guía del ajuste de la máscara a los bordes
        if comp.photo_grayscale:
            gray = top @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            top = np.repeat(gray[..., None], 3, axis=-1)
        mask = None
        if comp.mask == "areas":
            mask = combined_mask(project.areas, points if points is not None else project.scene.points(depth))
        elif comp.mask == "depth_range":
            mask = depth_range_mask(depth, comp.depth_near, comp.depth_far, comp.depth_softness)
        elif comp.mask == "painted" and painted_mask is not None:
            mask = painted_mask
        elif comp.mask == "painted" and comp.painted_mask:
            mask_path = Path(comp.painted_mask)
            if not mask_path.is_absolute() and mask_base_dir is not None:
                mask_path = mask_base_dir / mask_path
            try:
                mask = load_mask(mask_path, (w, h))
            except OSError:
                log.warning("No se encuentra la máscara pintada: %s", mask_path)
        if mask is not None and (comp.mask_feather > 0 or comp.mask_snap > 0):
            mask = refine_mask(mask, guide, comp.mask_feather, comp.mask_snap)
        if mask is not None and comp.invert_mask:
            mask = 1.0 - mask
        image = blend(image, top, comp.mode, comp.opacity, mask)

    if project.lens.enabled and project.lens.target == "result":
        image = lens_blur(image, depth, project.lens)

    if with_look and project.look:                   # Look: capas de ajuste al final, en el orden de la pila
        image = apply_look(image, project.look, depth)

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def to_uint16(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image, 0, 1) * 65535).astype(np.uint16)
