"""Áreas locales de enfoque y luz en 3D (elipsoide o caja), como el add-on de referencia.

Modelo de escena (ROADMAP, decisión 2): cada píxel se convierte en un punto 3D.
  - x: horizontal, de -aspecto a +aspecto (aspecto = ancho / alto).
  - y: vertical, de +1 (arriba) a -1 (abajo).
  - z: profundidad × relieve (0 = cerca). "Relieve" es la escala de profundidad del add-on.
Con `fov_deg` > 0 se usa una cámara en perspectiva: lo lejano se "abre" en x, y como
en una foto real. Con 0, proyección ortográfica (más fácil de manejar).

Un área es un elipsoide o una caja con centro, escala y rotación en ese espacio. Por eso
selecciona por posición en la imagen Y por distancia: puede iluminar a una persona sin
tocar la pared que tiene detrás.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from deepshot.tonemap import smoothstep

Vec3 = tuple[float, float, float]

# Colores con los que se dibujan las áreas sobre la imagen (solo en pantalla: no afectan al
# resultado). Cada área toma uno según su posición en la lista; el usuario puede cambiarlo.
AREA_COLORS: list[Vec3] = [
    (1.00, 0.81, 0.35),   # ámbar
    (0.30, 0.79, 0.94),   # cian
    (0.95, 0.36, 0.71),   # magenta
    (0.62, 0.94, 0.10),   # lima
    (1.00, 0.55, 0.26),   # naranja
    (0.70, 0.53, 1.00),   # violeta
    (0.35, 0.90, 0.70),   # turquesa
    (1.00, 0.45, 0.45),   # coral
]


# ---------------------------------------------------------------------------
# Escena
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    relief: float = 1.0       # escala de profundidad (z = depth × relief)
    fov_deg: float = 0.0      # 0 = ortográfica; p. ej. 50 = cámara normal

    def points(self, depth: np.ndarray) -> np.ndarray:
        """Nube de puntos (alto, ancho, 3) a partir del mapa de profundidad."""
        h, w = depth.shape
        aspect = w / h
        # Centros de píxel en coordenadas normalizadas.
        xs = ((np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0) * aspect
        ys = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0
        x, y = np.meshgrid(xs, ys)
        z = depth.astype(np.float32) * self.relief
        if self.fov_deg > 0:
            # Perspectiva: la cámara está a distancia d0 del plano z = 0, con el fov vertical dado.
            d0 = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
            factor = (d0 + z) / d0
            x, y = x * factor, y * factor
        return np.stack([x, y, z], axis=-1)

    def point_at(self, depth: np.ndarray, u: float, v: float) -> Vec3:
        """Punto 3D bajo el píxel (u, v) en fracciones 0..1 de ancho y alto (para colocar áreas)."""
        h, w = depth.shape
        col = min(max(int(u * w), 0), w - 1)
        row = min(max(int(v * h), 0), h - 1)
        # Misma fórmula que points(), solo para un píxel.
        x = ((col + 0.5) / w * 2.0 - 1.0) * (w / h)
        y = 1.0 - (row + 0.5) / h * 2.0
        z = float(depth[row, col]) * self.relief
        if self.fov_deg > 0:
            d0 = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
            factor = (d0 + z) / d0
            x, y = x * factor, y * factor
        return (float(x), float(y), z)


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matriz de rotación a partir de ángulos en grados (orden X, luego Y, luego Z)."""
    ax, ay, az = (math.radians(a) for a in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return mz @ my @ mx


# ---------------------------------------------------------------------------
# Áreas
# ---------------------------------------------------------------------------
@dataclass
class Area:
    """Área local en 3D.

    shape:          "ellipsoid" o "box".
    center:         centro (x, y, z) en el espacio de la escena.
    scale:          semiejes (x, y, z).
    rotation:       grados alrededor de X, Y, Z.
    strength:       > 0 aclara hacia blanco, < 0 oscurece hacia negro (-1..1 habitual).
    edge_softness:  0 = borde duro; 1 = transición desde el centro hasta el borde.
    falloff:        0 = intensidad uniforme dentro; 1 = cae linealmente del centro al borde.
    ui_color:       color con el que se dibuja el área en pantalla (None = automático por posición).
    from_style:     huella del área tal como la trajo un estilo ("" = hecha por el usuario). Al aplicar otro estilo se
                    quitan las áreas cuya huella sigue coincidiendo con su contenido (no se tocaron); si el usuario
                    las edita, la huella ya no coincide y pasan a ser suyas. Ver `presets.apply_to_project_dict`.
    """

    name: str = "Área"
    shape: str = "ellipsoid"
    center: Vec3 = (0.0, 0.0, 0.5)
    scale: Vec3 = (0.4, 0.4, 0.4)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    strength: float = 1.0
    edge_softness: float = 0.5
    falloff: float = 0.5
    enabled: bool = True
    ui_color: Vec3 | None = None
    from_style: str = ""

    def display_color(self, index: int) -> Vec3:
        """Color de la interfaz para esta área (el elegido o el automático según su posición)."""
        return self.ui_color if self.ui_color is not None else AREA_COLORS[index % len(AREA_COLORS)]

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distancia normalizada al centro: 1 = superficie del área, < 1 = dentro."""
        local = points - np.asarray(self.center, dtype=np.float32)
        # Al sistema del área: rotación inversa (= traspuesta) y división por la escala.
        local = local @ rotation_matrix(*self.rotation)
        local = local / np.maximum(np.abs(np.asarray(self.scale, dtype=np.float32)), 1e-6)
        if self.shape == "box":
            return np.max(np.abs(local), axis=-1)
        if self.shape == "ellipsoid":
            return np.sqrt(np.sum(local * local, axis=-1))
        raise ValueError(f"Forma desconocida: {self.shape}")

    def mask(self, points: np.ndarray) -> np.ndarray:
        """Influencia 0..1 de cada punto (sin signo)."""
        r = self.distance(points)
        soft = min(max(self.edge_softness, 0.0), 1.0)
        inside = smoothstep(1.0, 1.0 - soft, r) if soft > 0 else (r <= 1.0).astype(np.float32)
        fall = 1.0 - min(max(self.falloff, 0.0), 1.0) * np.clip(r, 0.0, 1.0)
        return (inside * fall).astype(np.float32)


def apply_areas(image: np.ndarray, areas: list[Area], points: np.ndarray) -> np.ndarray:
    """Aplica las áreas una tras otra. Fuerza positiva aclara, negativa oscurece.

    Se mezcla hacia blanco o negro según la fuerza, así el resultado nunca sale de 0..1.
    """
    out = image.copy()
    for area in areas:
        if not area.enabled or area.strength == 0:
            continue
        amount = (area.mask(points) * min(abs(area.strength), 1.0))[..., None]
        target = 1.0 if area.strength > 0 else 0.0
        out = out + (target - out) * amount
    return out.astype(np.float32)


def combined_mask(areas: list[Area], points: np.ndarray) -> np.ndarray:
    """Unión de las máscaras de varias áreas (útil para mezclar por máscara)."""
    m = np.zeros(points.shape[:2], dtype=np.float32)
    for area in areas:
        if area.enabled:
            m = np.maximum(m, area.mask(points))
    return m
