"""Edición de áreas desde la imagen: de movimientos del ratón a valores del área.

Sin interfaz (solo números), para poder probarlo. Trabaja en píxeles de la foto, con el
origen arriba a la izquierda y la Y hacia abajo, y los traduce al espacio de la escena
(ver `areas.py`): x ∈ ±aspecto, y ∈ ±1 con +1 arriba, z = profundidad × relieve.

Cómo se ve un área sobre la imagen (misma proyección que el contorno que dibuja el lienzo):
    factor f = (d0 + z) / d0 con perspectiva (d0 = 1 / tan(fov / 2)), o 1 sin ella;
    centro en píxeles: cx = (x / f / aspecto + 1) / 2 · ancho, cy = (1 − y / f) / 2 · alto;
    semiejes en píxeles: rx = escala_x / f · alto / 2, ry = escala_y / f · alto / 2.
El área gira `rotación_z` grados en sentido antihorario sobre la imagen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from deepshot.areas import Area, Scene, Vec3

MIN_SCALE = 0.005     # semieje mínimo: por debajo el área desaparece y no se puede volver a agarrar


@dataclass
class ImageGeometry:
    """Tamaño de la foto, cámara y mapa de profundidad (para fijar el área a la superficie)."""

    width: int
    height: int
    scene: Scene
    depth: np.ndarray | None = None     # mapa 0..1 (puede ser reducido: se usa en fracciones)

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def factor(self, z: float) -> float:
        """Cuánto se achica lo que está a profundidad z (1 = sin perspectiva)."""
        if self.scene.fov_deg <= 0:
            return 1.0
        d0 = 1.0 / math.tan(math.radians(self.scene.fov_deg) / 2.0)
        return (d0 + z) / d0

    # --- Conversiones imagen <-> escena -----------------------------------------------------
    def to_scene_xy(self, px: float, py: float, z: float) -> tuple[float, float]:
        f = self.factor(z)
        return ((px / self.width * 2.0 - 1.0) * self.aspect * f, (1.0 - py / self.height * 2.0) * f)

    def to_pixels(self, x: float, y: float, z: float) -> tuple[float, float]:
        f = self.factor(z)
        return ((x / f / self.aspect + 1.0) / 2.0 * self.width, (1.0 - y / f) / 2.0 * self.height)

    def units_to_pixels(self, length: float, z: float) -> float:
        """Longitud de la escena (a profundidad z) -> píxeles."""
        return length / self.factor(z) * self.height / 2.0

    def pixels_to_units(self, pixels: float, z: float) -> float:
        return pixels * 2.0 / self.height * self.factor(z)

    def surface_point(self, px: float, py: float) -> Vec3 | None:
        """Punto 3D de la superficie bajo el píxel (None si no hay mapa de profundidad)."""
        if self.depth is None:
            return None
        u = min(max(px / self.width, 0.0), 1.0)
        v = min(max(py / self.height, 0.0), 1.0)
        return self.scene.point_at(self.depth, u, v)


def axes(angle_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Ejes locales del área en pantalla (X hacia la derecha, Y hacia abajo), ya girados."""
    a = math.radians(angle_deg)
    return (math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))


def _wrap_angle(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _snap(deg: float, step: float) -> float:
    return round(deg / step) * step


# ---------------------------------------------------------------------------
# Gestos. Cada función recibe el área tal como era al empezar (`area0`) y devuelve una copia
# modificada; así se puede arrastrar de ida y vuelta sin acumular errores.
# ---------------------------------------------------------------------------
def move(area0: Area, press: tuple[float, float], pointer: tuple[float, float], geo: ImageGeometry,
         snap_to_surface: bool = False, axis: str = "") -> Area:
    """Mueve el centro. Sin modificadores: en el plano de la imagen (conserva la profundidad).
    snap_to_surface: el centro se pega a la superficie bajo el puntero (cambia z).
    axis "x" / "y": solo ese eje de pantalla; "z": la profundidad (movimiento horizontal)."""
    x0, y0, z0 = area0.center
    if snap_to_surface:
        hit = geo.surface_point(*pointer)
        if hit is not None:
            return replace(area0, center=hit)
    dx, dy = pointer[0] - press[0], pointer[1] - press[1]
    if axis == "z":
        return replace(area0, center=(x0, y0, z0 + geo.pixels_to_units(dx, z0)))
    if axis == "x":
        dy = 0.0
    elif axis == "y":
        dx = 0.0
    cx, cy = geo.to_pixels(x0, y0, z0)
    nx, ny = geo.to_scene_xy(cx + dx, cy + dy, z0)
    return replace(area0, center=(nx, ny, z0))


def scale_handle(area0: Area, handle: str, pointer: tuple[float, float], geo: ImageGeometry,
                 uniform: bool = False) -> Area:
    """Arrastrar el tirador de un semieje ("x" o "y"): su longitud pasa a ser la distancia del
    puntero al centro medida sobre ese eje. uniform: escala también el otro eje y Z a la vez."""
    cx, cy = geo.to_pixels(*area0.center)
    ux, uy = axes(area0.rotation[2])
    vx, vy = pointer[0] - cx, pointer[1] - cy
    along = abs(vx * (ux[0] if handle == "x" else uy[0]) + vy * (ux[1] if handle == "x" else uy[1]))
    new = max(geo.pixels_to_units(along, area0.center[2]), MIN_SCALE)
    sx, sy, sz = area0.scale
    old = sx if handle == "x" else sy
    if uniform:
        k = new / max(old, 1e-9)
        return replace(area0, scale=(max(sx * k, MIN_SCALE), max(sy * k, MIN_SCALE), max(sz * k, MIN_SCALE)))
    return replace(area0, scale=(new, sy, sz) if handle == "x" else (sx, new, sz))


def scale_all(area0: Area, press: tuple[float, float], pointer: tuple[float, float], geo: ImageGeometry,
              axis: str = "") -> Area:
    """Escalado como la tecla S de Blender: la razón entre la distancia actual y la inicial del
    puntero al centro multiplica la escala. axis "x" / "y" / "z": solo ese eje."""
    cx, cy = geo.to_pixels(*area0.center)
    d0 = math.hypot(press[0] - cx, press[1] - cy)
    d1 = math.hypot(pointer[0] - cx, pointer[1] - cy)
    k = d1 / d0 if d0 > 1e-6 else 1.0
    sx, sy, sz = area0.scale
    if axis == "x":
        sx *= k
    elif axis == "y":
        sy *= k
    elif axis == "z":
        sz *= k
    else:
        sx, sy, sz = sx * k, sy * k, sz * k
    return replace(area0, scale=(max(sx, MIN_SCALE), max(sy, MIN_SCALE), max(sz, MIN_SCALE)))


def rotate_handle(area0: Area, pointer: tuple[float, float], geo: ImageGeometry, snap: bool = False) -> Area:
    """El tirador de giro está sobre el eje Y superior: el área gira hasta apuntarlo al puntero."""
    cx, cy = geo.to_pixels(*area0.center)
    angle = math.degrees(math.atan2(-(pointer[1] - cy), pointer[0] - cx)) - 90.0
    angle = _wrap_angle(_snap(angle, 15.0) if snap else angle)
    rx, ry, _ = area0.rotation
    return replace(area0, rotation=(rx, ry, angle))


def rotate_free(area0: Area, press: tuple[float, float], pointer: tuple[float, float], geo: ImageGeometry,
                axis: str = "z", snap: bool = False) -> Area:
    """Giro como la tecla R de Blender: el ángulo que recorre el puntero alrededor del centro se
    suma a la rotación (eje Z por defecto). Con axis "x" / "y" el giro sigue el movimiento
    horizontal del puntero (1 px = 0,5°), porque esos giros no se ven en el plano de la imagen."""
    rx, ry, rz = area0.rotation
    cx, cy = geo.to_pixels(*area0.center)
    if axis in ("x", "y"):
        delta = (pointer[0] - press[0]) * 0.5
        if snap:
            delta = _snap(delta, 15.0)
        return replace(area0, rotation=(_wrap_angle(rx + delta), ry, rz) if axis == "x"
                       else (rx, _wrap_angle(ry + delta), rz))
    a0 = math.atan2(-(press[1] - cy), press[0] - cx)
    a1 = math.atan2(-(pointer[1] - cy), pointer[0] - cx)
    delta = math.degrees(a1 - a0)
    if snap:
        delta = _snap(delta, 15.0)
    return replace(area0, rotation=(rx, ry, _wrap_angle(rz + delta)))


def pan_tilt(area0: Area, kind: str, press: tuple[float, float], pointer: tuple[float, float],
             snap: bool = False) -> Area:
    """Giro «paneo / inclinación» con los tiradores de los lados: kind «pan» gira sobre el eje vertical (rotación Y)
    con el movimiento horizontal del puntero; «tilt» gira sobre el horizontal (rotación X) con el vertical.
    1 px = 0,5°; con `snap`, de 15° en 15°. Así se orienta hacia dónde actúa el área sin salir de la vista 2D."""
    rx, ry, rz = area0.rotation
    if kind == "pan":
        delta = (pointer[0] - press[0]) * 0.5
        delta = _snap(delta, 15.0) if snap else delta
        return replace(area0, rotation=(rx, _wrap_angle(ry + delta), rz))
    delta = (pointer[1] - press[1]) * 0.5
    delta = _snap(delta, 15.0) if snap else delta
    return replace(area0, rotation=(_wrap_angle(rx + delta), ry, rz))


def draw(area0: Area, press: tuple[float, float], pointer: tuple[float, float], geo: ImageGeometry,
         free: bool = False) -> Area:
    """«Dibujar» un área: el centro es donde se pulsó (pegado a la superficie) y el radio es lo que
    se arrastra. free: caja rectangular desde el centro (semiejes X e Y por separado)."""
    hit = geo.surface_point(*press)
    center = hit if hit is not None else area0.center
    z = center[2]
    dx, dy = abs(pointer[0] - press[0]), abs(pointer[1] - press[1])
    if free:
        rx, ry = dx, dy
    else:
        rx = ry = math.hypot(dx, dy)
    sx = max(geo.pixels_to_units(rx, z), MIN_SCALE)
    sy = max(geo.pixels_to_units(ry, z), MIN_SCALE)
    sz = max(min(sx, sy) * 0.6, MIN_SCALE)       # espesor en profundidad proporcional al tamaño
    return replace(area0, center=center, scale=(sx, sy, sz))


def place(area0: Area, pointer: tuple[float, float], geo: ImageGeometry) -> Area:
    """«Colocar»: el centro del área sigue al puntero, pegado a la superficie de la escena."""
    hit = geo.surface_point(*pointer)
    return replace(area0, center=hit) if hit is not None else area0


# ---------------------------------------------------------------------------
# Contornos para dibujar
# ---------------------------------------------------------------------------
def _hull(points: np.ndarray) -> np.ndarray:
    """Envolvente convexa (cadena monótona) de puntos (N, 2), en orden de contorno."""
    pts = sorted({(round(float(x), 3), round(float(y), 3)) for x, y in points})
    if len(pts) < 3:
        return np.asarray(pts, dtype=np.float64)

    def half(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) \
                    - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


TILT_SILHOUETTE_DEG = 0.5      # a partir de esta inclinación (rotación X o Y) el contorno ya no es una elipse


def silhouette(area: Area, geo: ImageGeometry) -> list[tuple[float, float]] | None:
    """Contorno real del área en píxeles cuando está inclinada (rotación X o Y): la envolvente de su forma 3D vista
    desde la cámara de la imagen. None si no está inclinada (vale la elipse o la caja de `outline`)."""
    if abs(area.rotation[0]) < TILT_SILHOUETTE_DEG and abs(area.rotation[1]) < TILT_SILHOUETTE_DEG:
        return None
    from deepshot import area3d

    pts = np.concatenate(area3d.wireframe(area), axis=0)
    pixels = np.array([geo.to_pixels(float(x), float(y), float(z)) for x, y, z in pts], dtype=np.float64)
    hull = _hull(pixels)
    return [(float(x), float(y)) for x, y in hull] if len(hull) >= 3 else None


def outline(area: Area, geo: ImageGeometry) -> dict:
    """Elipse (o caja) del área en píxeles: centro, semiejes y ángulo, como los usa el lienzo. Si el área está inclinada,
    lleva además `poly`: el contorno real (lista de puntos) que el lienzo dibuja en vez de la elipse."""
    x, y, z = area.center
    cx, cy = geo.to_pixels(x, y, z)
    out = {
        "cx": cx, "cy": cy,
        "rx": geo.units_to_pixels(abs(area.scale[0]), z), "ry": geo.units_to_pixels(abs(area.scale[1]), z),
        "angle": area.rotation[2], "box": area.shape == "box",
    }
    poly = silhouette(area, geo)
    if poly is not None:
        out["poly"] = poly
    return out
