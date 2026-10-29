"""Modo sencillo: de una foto a un mapa de profundidad o un mapa de normales listo para exportar (sin interfaz).

Es el «gancho» gratuito de la aplicación: sin rampas, áreas ni desenfoque, solo el mapa. Convención interna del
resto de la app: 0 = cerca, 1 = lejos. Aquí se prepara para guardar:
  - `finish_depth`: normalizar (aprovechar todo el rango) e invertir (cerca = claro o cerca = oscuro).
  - `normal_map`: mapa de normales derivado de la profundidad, con intensidad y convención OpenGL o DirectX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

NORMAL_CONVENTIONS = ("opengl", "directx")     # OpenGL: verde hacia arriba · DirectX: verde hacia abajo
MIN_STRENGTH, MAX_STRENGTH, DEFAULT_STRENGTH = 0.1, 10.0, 1.0
RELIEF_FRACTION = 0.05          # con intensidad 1, el rango completo de profundidad equivale al 5 % del lado largo


def normalize_range(depth: np.ndarray, low_pct: float = 0.0, high_pct: float = 100.0) -> np.ndarray:
    """Estira el mapa para que ocupe todo el rango 0..1. Con percentiles distintos de 0 y 100 se ignoran los
    píxeles extremos (útil si un punto suelto aplasta el resto del contraste). Un mapa plano devuelve ceros."""
    d = np.asarray(depth, dtype=np.float32)
    lo, hi = (float(d.min()), float(d.max())) if (low_pct, high_pct) == (0.0, 100.0) \
        else (float(np.percentile(d, low_pct)), float(np.percentile(d, high_pct)))
    if hi <= lo:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def finish_depth(depth: np.ndarray, normalize: bool = True, near_bright: bool = True) -> np.ndarray:
    """Mapa listo para guardar en gris: float32 0..1. `near_bright`: lo cercano claro (lo habitual, como la vista)."""
    d = normalize_range(depth) if normalize else np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    return 1.0 - d if near_bright else d


def normal_map(depth: np.ndarray, strength: float = DEFAULT_STRENGTH, convention: str = "opengl",
               smooth: float = 1.0) -> np.ndarray:
    """Mapa de normales (alto, ancho, 3) float32 en 0..1 (RGB = XYZ·0,5 + 0,5), calculado a partir de la profundidad.

    La altura es «cerca = alto». `strength` multiplica el relieve (1 = el rango completo de profundidad vale el 5 %
    del lado largo de la imagen): más intensidad = normales más inclinadas. `smooth` es un desenfoque previo en píxeles
    (a la escala de una imagen de 1500 px de lado largo; crece con la imagen) para que el escalonado del mapa no
    salga como estrías. `convention`: «opengl» (Y+ arriba, Blender y Unity) o «directx» (Y+ abajo, Unreal)."""
    if convention not in NORMAL_CONVENTIONS:
        raise ValueError(f"Convención de normales desconocida: {convention}")
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape
    long_side = float(max(h, w))
    height = 1.0 - np.clip(d, 0.0, 1.0)
    sigma = smooth * long_side / 1500.0
    if sigma > 0.3:
        height = gaussian_filter(height, sigma)
    height = height * (float(strength) * RELIEF_FRACTION * long_side)        # altura en píxeles
    gy_down, gx = np.gradient(height)                                         # d(altura)/d(píxel): abajo y derecha
    nx, ny, nz = -gx, gy_down, np.ones_like(gx)       # ny hacia arriba = -(-gy_down): la superficie sube hacia arriba
    if convention == "directx":
        ny = -ny
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1) * 0.5 + 0.5
    return out.astype(np.float32)


@dataclass
class SimpleOptions:
    """Lo que se pide en el modo sencillo."""
    depth_map: bool = True
    normal: bool = False
    normalize: bool = True
    near_bright: bool = True
    strength: float = DEFAULT_STRENGTH
    convention: str = "opengl"
    format: str = "png16"                 # «png8» | «png16» | «exr32»
    folder: Path = Path(".")
    tag: str = ""                         # va al final del nombre: «foto_depth_<tag>» (p. ej. modelo y calidad)
    save_json: bool = False               # guarda junto a cada imagen un .json con cómo se hizo
    info: dict | None = None              # datos que se escriben en ese .json (modelo, calidad…)

    def validate(self) -> None:
        if not (self.depth_map or self.normal):
            raise ValueError("Elige al menos un resultado: mapa de profundidad o mapa de normales.")
        if self.convention not in NORMAL_CONVENTIONS:
            raise ValueError(f"Convención de normales desconocida: {self.convention}")
        self.strength = min(max(float(self.strength), MIN_STRENGTH), MAX_STRENGTH)


def file_tag(model: str, quality: str, resolution: int | None = None) -> str:
    """Texto para el final del nombre del archivo: modelo, calidad y, si se eligió a mano, la resolución.
    Solo letras, números, guiones y guiones bajos, para que valga en cualquier sistema."""
    parts = [model, quality] + ([f"{int(resolution)}px"] if resolution else [])
    return "_".join("".join(c if c.isalnum() or c in "-." else "-" for c in p) for p in parts if p)


def _write_json(path: Path, photo_name: str, kind: str, options: SimpleOptions) -> Path:
    """Escribe «<archivo>.json» junto a la imagen con cómo se hizo (para repetirla o saber de qué modelo salió)."""
    import json
    from datetime import datetime

    data = {"photo": photo_name, "result": kind, "file": path.name, "format": options.format,
            "normalize": options.normalize, "created": datetime.now().isoformat(timespec="seconds"),
            **(options.info or {})}
    if kind == "depth":
        data["near_bright"] = options.near_bright
    else:
        data.update(strength=options.strength, convention=options.convention)
    target = path.with_name(path.name + ".json")           # «x.png.json»: nunca choca con el de otra imagen
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def export_simple(depth: np.ndarray, photo_name: str, options: SimpleOptions) -> list[Path]:
    """Guarda lo pedido (mapa de profundidad y/o de normales) a la resolución del mapa; devuelve las rutas escritas.
    Los nombres son «<foto>_depth» y «<foto>_normal» (más «_<tag>» si se da), sin pisar archivos que ya existan.
    Con `save_json` se escribe también un .json por imagen (se devuelve en la lista)."""
    from deepshot.export import EXTENSIONS, free_path, save_image

    options.validate()
    options.folder.mkdir(parents=True, exist_ok=True)
    stem = Path(photo_name).stem
    suffix = f"_{options.tag}" if options.tag else ""
    written: list[Path] = []
    if options.depth_map:
        path = free_path(options.folder, f"{stem}_depth{suffix}", EXTENSIONS[options.format])
        save_image(finish_depth(depth, options.normalize, options.near_bright), path, options.format)
        written.append(path)
        if options.save_json:
            written.append(_write_json(path, photo_name, "depth", options))
    if options.normal:
        base = normalize_range(depth) if options.normalize else depth
        path = free_path(options.folder, f"{stem}_normal{suffix}", EXTENSIONS[options.format])
        save_image(normal_map(base, options.strength, options.convention), path, options.format)
        written.append(path)
        if options.save_json:
            written.append(_write_json(path, photo_name, "normal", options))
    return written
