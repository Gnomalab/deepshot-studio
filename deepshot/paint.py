"""Máscara pintada a mano: una capa en gris que se retoca con un pincel redondo.

Funciona como el pincel de un programa de retoque: tamaño, dureza (borde duro o difuminado) y
opacidad, y un modo borrador. La máscara decide dónde se aplica la foto original (blanco = se
aplica, negro = no). Se guarda a menos resolución que la foto (lado largo `MAX_SIDE`, 8 bits) y se
reescala al procesar; los bordes de un pincel son suaves, así que no se nota.

Sin interfaz: solo numpy y Pillow, para poder probarlo.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from PIL import Image

MAX_SIDE = 2048          # resolución máxima de la capa (lado largo)
DAB_SPACING = 0.18       # separación entre toques de un trazo, como fracción del radio


def layer_size(photo_size: tuple[int, int]) -> tuple[int, int]:
    """Tamaño (ancho, alto) de la capa para una foto de `photo_size` = (ancho, alto)."""
    w, h = photo_size
    k = min(1.0, MAX_SIDE / max(w, h))
    return max(1, round(w * k)), max(1, round(h * k))


class MaskLayer:
    """Máscara de 8 bits (0 = negro, 255 = blanco) con las coordenadas de la foto como referencia."""

    def __init__(self, photo_size: tuple[int, int], data: np.ndarray | None = None) -> None:
        self.photo_size = photo_size
        self.width, self.height = layer_size(photo_size)
        self.data = np.zeros((self.height, self.width), dtype=np.uint8) if data is None else data
        self._last: tuple[float, float] | None = None     # último toque del trazo en curso (px de la foto)
        self.version = 0                                  # sube con cada cambio (para cachés)

    # --- Pincel ---------------------------------------------------------------------------
    @property
    def scale(self) -> float:
        """Píxeles de la capa por píxel de la foto."""
        return self.width / self.photo_size[0]

    def begin_stroke(self) -> None:
        self._last = None

    def stroke_to(self, x: float, y: float, radius: float, hardness: float, opacity: float,
                  erase: bool = False) -> tuple[int, int, int, int] | None:
        """Pinta desde el toque anterior hasta (x, y), en píxeles de la foto. `radius` también.
        Devuelve el rectángulo (x0, y0, x1, y1) de la capa que cambió, o None."""
        points = [(x, y)]
        if self._last is not None:
            lx, ly = self._last
            dist = float(np.hypot(x - lx, y - ly))
            step = max(radius * DAB_SPACING, 1.0)
            n = int(dist // step)
            points = [(lx + (x - lx) * i / (n + 1), ly + (y - ly) * i / (n + 1)) for i in range(1, n + 1)] + [(x, y)]
        self._last = (x, y)
        dirty = None
        for px, py in points:
            rect = self.dab(px, py, radius, hardness, opacity, erase)
            if rect is not None:
                dirty = rect if dirty is None else (min(dirty[0], rect[0]), min(dirty[1], rect[1]),
                                                    max(dirty[2], rect[2]), max(dirty[3], rect[3]))
        return dirty

    def dab(self, x: float, y: float, radius: float, hardness: float, opacity: float,
            erase: bool = False) -> tuple[int, int, int, int] | None:
        """Un toque del pincel en (x, y) (píxeles de la foto).

        La influencia vale 1 hasta `hardness · radio` y baja suavemente a 0 en el borde. Pintar
        acerca la máscara a blanco en proporción a la opacidad (los toques se acumulan, como un
        aerógrafo); borrar la acerca a negro.
        """
        k = self.scale
        cx, cy, r = x * k, y * k, max(radius * k, 0.5)
        x0, x1 = max(int(np.floor(cx - r)), 0), min(int(np.ceil(cx + r)) + 1, self.width)
        y0, y1 = max(int(np.floor(cy - r)), 0), min(int(np.ceil(cy + r)) + 1, self.height)
        if x0 >= x1 or y0 >= y1:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / r                    # 0 en el centro, 1 en el borde
        h = min(max(hardness, 0.0), 0.999)
        t = np.clip((1.0 - d) / (1.0 - h), 0.0, 1.0)
        alpha = (t * t * (3.0 - 2.0 * t)) * min(max(opacity, 0.0), 1.0)   # suavizado tipo smoothstep
        region = self.data[y0:y1, x0:x1].astype(np.float32)
        target = 0.0 if erase else 255.0
        region = region + (target - region) * alpha
        self.data[y0:y1, x0:x1] = np.clip(np.rint(region), 0, 255).astype(np.uint8)
        self.version += 1
        return x0, y0, x1, y1

    # --- Operaciones sobre toda la capa -------------------------------------------------------
    def fill(self, value: float) -> None:
        self.data[:] = int(round(min(max(value, 0.0), 1.0) * 255))
        self.version += 1

    def invert(self) -> None:
        self.data[:] = 255 - self.data
        self.version += 1

    def is_empty(self) -> bool:
        return not self.data.any()

    # --- Lectura, copias y deshacer ------------------------------------------------------------------
    def to_float(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Máscara 0..1 float32, reescalada a `size` = (ancho, alto) si se pide."""
        if size is None or tuple(size) == (self.width, self.height):
            return self.data.astype(np.float32) / 255.0
        img = Image.fromarray(self.data, mode="L").resize(tuple(size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def snapshot(self) -> bytes:
        """Copia comprimida (para el paso de deshacer): una máscara casi vacía ocupa muy poco."""
        return zlib.compress(self.data.tobytes(), 1)

    def restore(self, blob: bytes) -> None:
        raw = zlib.decompress(blob)
        self.data = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width).copy()
        self.version += 1

    # --- Archivos --------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.data, mode="L").save(path)

    @classmethod
    def load(cls, path: Path, photo_size: tuple[int, int]) -> "MaskLayer":
        """Lee un PNG gris (8 o 16 bits, cualquier tamaño) y lo adapta a la foto."""
        layer = cls(photo_size)
        with Image.open(path) as img:
            if img.mode.startswith("I"):
                arr = np.clip(np.asarray(img, dtype=np.float32) / 65535.0, 0, 1)
                img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            else:
                img = img.convert("L")
            img = img.resize((layer.width, layer.height), Image.BILINEAR)
            layer.data = np.asarray(img, dtype=np.uint8).copy()
        return layer


def mask_path_for(photo_path: str) -> Path:
    """Archivo PNG donde se guarda la máscara pintada de una foto (en la carpeta de datos).

    El nombre lleva un trozo de la huella de la ruta para que dos fotos con el mismo nombre no se pisen.
    """
    import hashlib

    from deepshot.paths import data_subdir

    key = hashlib.sha256(photo_path.encode("utf-8")).hexdigest()[:10]
    return data_subdir("masks") / f"{Path(photo_path).stem[:40]}-{key}.png"
