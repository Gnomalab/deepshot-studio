"""Edición de áreas en la vista 3D: geometría de los contornos y de los tiradores, y de ratón a valores. Sin interfaz.

Las áreas viven en la misma escena que el relieve (`areas.Scene`): x derecha, y arriba, z al fondo. Un punto local
`u` (en unidades de semieje) está en el mundo en `centro + R @ (escala * u)`, con R = `rotation_matrix(*rotación)`.

Los gestos reciben el área tal como era al pulsar (`area0`) y dos rayos de ratón (al pulsar y ahora), y devuelven
valores nuevos: igual que `area_edit.py` hace en la vista 2D, así un gesto entero es un solo paso de deshacer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from deepshot.areas import Area, rotation_matrix
from deepshot.view3d import OrbitCamera

MIN_SCALE = 0.005                  # semieje mínimo, igual que en la vista 2D
GIZMO_FRACTION = 0.16              # largo de las flechas, en fracción de la distancia de la cámara (tamaño fijo en pantalla)
AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Geometría del área
# ---------------------------------------------------------------------------
def matrix(area: Area) -> np.ndarray:
    return rotation_matrix(*area.rotation).astype(np.float64)


def to_world(area: Area, local: np.ndarray) -> np.ndarray:
    """Puntos locales (N, 3), en unidades de semieje, al mundo."""
    return np.asarray(area.center, dtype=np.float64) + (np.asarray(local, dtype=np.float64)
                                                          * np.asarray(area.scale, dtype=np.float64)) @ matrix(area).T


def wireframe(area: Area, segments: int = 40) -> list[np.ndarray]:
    """Líneas que dibujan el área: tres elipses (elipsoide) o las doce aristas (caja). Cada una (N, 3) en el mundo."""
    if area.shape == "box":
        corners = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], dtype=np.float64)
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
        return [to_world(area, corners[[a, b]]) for a, b in edges]
    t = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    c, s, z = np.cos(t), np.sin(t), np.zeros_like(t)
    return [to_world(area, np.stack(ring, axis=-1)) for ring in ((c, s, z), (c, z, s), (z, c, s))]


def local_axes(area: Area) -> np.ndarray:
    """Columnas = ejes locales del área en el mundo."""
    return matrix(area)


def scale_handle_points(area: Area) -> dict[str, np.ndarray]:
    """Extremos + de cada semieje (donde van los cubos de escala)."""
    r = matrix(area)
    c = np.asarray(area.center, dtype=np.float64)
    return {name: c + r[:, i] * float(area.scale[i]) for name, i in AXIS_INDEX.items()}


def gizmo_length(camera: OrbitCamera) -> float:
    return GIZMO_FRACTION * camera.distance


def ring_points(center: np.ndarray, axis: str, radius: float, segments: int = 48) -> np.ndarray:
    """Circunferencia alrededor del eje del mundo `axis`, centrada en `center`."""
    t = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    c, s, z = np.cos(t) * radius, np.sin(t) * radius, np.zeros_like(t)
    local = {"x": np.stack([z, c, s], -1), "y": np.stack([c, z, s], -1), "z": np.stack([c, s, z], -1)}[axis]
    return np.asarray(center, dtype=np.float64) + local


# ---------------------------------------------------------------------------
# Rayos y planos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray


def ray_through(camera: OrbitCamera, x: float, y: float, width: int, height: int) -> Ray:
    """Rayo del ojo de la cámara que atraviesa el píxel (x, y) de la pantalla."""
    right, up, forward = camera.basis()
    focal = (height / 2.0) / math.tan(math.radians(camera.fov_deg) / 2.0)
    d = forward + right * ((x - width / 2.0) / focal) - up * ((y - height / 2.0) / focal)
    return Ray(camera.position(), d / np.linalg.norm(d))


def param_on_axis(ray: Ray, origin: np.ndarray, axis: np.ndarray) -> float | None:
    """Parámetro s del punto `origin + s·axis` (eje unitario) más cercano al rayo; None si son casi paralelos."""
    w0 = origin - ray.origin
    b = float(axis @ ray.direction)
    d, e = float(axis @ w0), float(ray.direction @ w0)
    denom = 1.0 - b * b
    if denom < 1e-6:
        return None
    return (b * e - d) / denom


def hit_plane(ray: Ray, point: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    """Punto donde el rayo corta el plano (None si es paralelo o queda detrás de la cámara)."""
    denom = float(normal @ ray.direction)
    if abs(denom) < 1e-6:
        return None
    t = float(normal @ (point - ray.origin)) / denom
    return ray.origin + ray.direction * t if t > 0 else None


# ---------------------------------------------------------------------------
# Rotaciones
# ---------------------------------------------------------------------------
def axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis].astype(np.float64)


def euler_from_matrix(r: np.ndarray) -> tuple[float, float, float]:
    """Inversa de `rotation_matrix` (R = Rz·Ry·Rx): grados (x, y, z)."""
    sy = -float(np.clip(r[2, 0], -1.0, 1.0))
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(r[2, 1], r[2, 2])
        rz = math.atan2(r[1, 0], r[0, 0])
    else:                                  # bloqueo de cardán: se reparte todo en X
        rx = math.atan2(-r[1, 2], r[1, 1])
        rz = 0.0
    return tuple(math.degrees(v) for v in (rx, ry, rz))


# ---------------------------------------------------------------------------
# Gestos: de rayos de ratón a valores nuevos del área
# ---------------------------------------------------------------------------
def move_along_axis(area0: Area, axis: str, ray0: Ray, ray1: Ray) -> Area:
    """Mueve el centro sobre un eje del mundo siguiendo al ratón."""
    a = AXES[axis]
    c0 = np.asarray(area0.center, dtype=np.float64)
    s0, s1 = param_on_axis(ray0, c0, a), param_on_axis(ray1, c0, a)
    if s0 is None or s1 is None:
        return area0
    return replace(area0, center=tuple(float(v) for v in c0 + a * (s1 - s0)))


def move_in_view_plane(area0: Area, camera: OrbitCamera, ray0: Ray, ray1: Ray) -> Area:
    """Mueve el centro en el plano paralelo a la pantalla que pasa por él (arrastre libre)."""
    c0 = np.asarray(area0.center, dtype=np.float64)
    normal = camera.basis()[2]
    p0, p1 = hit_plane(ray0, c0, normal), hit_plane(ray1, c0, normal)
    if p0 is None or p1 is None:
        return area0
    return replace(area0, center=tuple(float(v) for v in c0 + (p1 - p0)))


def scale_along_axis(area0: Area, axis: str, ray1: Ray, uniform: bool = False) -> Area:
    """Escala un semieje: queda a la distancia del centro a la que está el ratón sobre ese eje local.
    Con `uniform` (Mayús) se escalan los tres semiejes en la misma proporción."""
    i = AXIS_INDEX[axis]
    c0 = np.asarray(area0.center, dtype=np.float64)
    s = param_on_axis(ray1, c0, matrix(area0)[:, i])
    if s is None:
        return area0
    new = max(abs(s), MIN_SCALE)
    if uniform:
        k = new / max(float(area0.scale[i]), 1e-9)
        scale = tuple(max(float(v) * k, MIN_SCALE) for v in area0.scale)
    else:
        scale = tuple(new if j == i else float(v) for j, v in enumerate(area0.scale))
    return replace(area0, scale=scale)


def scale_by_factor(area0: Area, factor: float, axis: str = "") -> Area:
    """Escala del modo S (teclado): multiplica los tres semiejes por `factor`, o solo el del eje local dado."""
    factor = max(float(factor), 1e-6)
    if axis:
        i = AXIS_INDEX[axis]
        scale = tuple(max(float(v) * factor, MIN_SCALE) if j == i else float(v) for j, v in enumerate(area0.scale))
    else:
        scale = tuple(max(float(v) * factor, MIN_SCALE) for v in area0.scale)
    return replace(area0, scale=scale)


def rotate_about_axis(area0: Area, axis: str, ray0: Ray, ray1: Ray, snap: bool = False) -> Area:
    """Gira el área alrededor de un eje del mundo: el ángulo es el que barre el ratón sobre el anillo."""
    c0 = np.asarray(area0.center, dtype=np.float64)
    n = AXES[axis]
    p0, p1 = hit_plane(ray0, c0, n), hit_plane(ray1, c0, n)
    if p0 is None or p1 is None:
        return area0
    v0, v1 = p0 - c0, p1 - c0
    if np.linalg.norm(v0) < 1e-9 or np.linalg.norm(v1) < 1e-9:
        return area0
    angle = math.atan2(float(n @ np.cross(v0, v1)), float(v0 @ v1))
    if snap:
        step = math.radians(15.0)
        angle = round(angle / step) * step
    return replace(area0, rotation=euler_from_matrix(axis_rotation(axis, angle) @ matrix(area0)))


# ---------------------------------------------------------------------------
# Bandas: tiradores del plano de una banda y gestos de ratón a valores
# ---------------------------------------------------------------------------
BAND_MAX_ANGLE = 80.0              # pan y tilt, igual que en el panel
BAND_MIN_SPAN = 0.05               # el ratón tiene que estar al menos así de lejos del centro para fijar un ángulo


@dataclass(frozen=True)
class BandFrame:
    """Una banda y la escena donde vive, lo justo para sus tiradores. El plano es z = z0 + x·tan(pan) + y·tan(tilt) (ver
    `view3d.band_plane`) y la parte plena de la franja es un grueso de ±`width`/2 en la escala de la rampa."""

    position: float
    width: float
    pan: float
    tilt: float
    ramp_min: float
    ramp_max: float
    relief: float
    aspect: float

    @property
    def z_span(self) -> float:
        """Cuánto mide la rampa entera en profundidad de la escena."""
        return max((self.ramp_max - self.ramp_min) * self.relief, 1e-6)

    @property
    def z0(self) -> float:
        return (self.ramp_min + self.position * (self.ramp_max - self.ramp_min)) * self.relief

    @property
    def half_z(self) -> float:
        return 0.5 * self.width * self.z_span

    def z_at(self, x: float, y: float) -> float:
        return self.z0 + x * math.tan(math.radians(self.pan)) + y * math.tan(math.radians(self.tilt))

    def position_of(self, z0: float) -> float:
        return float(np.clip((z0 / max(self.relief, 1e-6) - self.ramp_min) / max(self.ramp_max - self.ramp_min, 1e-6),
                             0.0, 1.0))


def band_handles(f: BandFrame) -> dict[str, np.ndarray]:
    """Puntos de los tiradores: «move» (centro, flecha en profundidad), «pan» (borde derecho), «tilt» (borde de arriba)
    y «width» (borde izquierdo, en la cara lejana del grueso)."""
    return {"move": np.array([0.0, 0.0, f.z0]),
            "pan": np.array([f.aspect, 0.0, f.z_at(f.aspect, 0.0)]),
            "tilt": np.array([0.0, 1.0, f.z_at(0.0, 1.0)]),
            "width": np.array([-f.aspect, 0.0, f.z_at(-f.aspect, 0.0) + f.half_z])}


def band_slab_corners(f: BandFrame, sign: int) -> np.ndarray:
    """Esquinas (4, 3) del plano a un lado (sign = -1) o al otro (+1) del grueso de la banda."""
    xs = np.array([-f.aspect, f.aspect, f.aspect, -f.aspect])
    ys = np.array([1.0, 1.0, -1.0, -1.0])
    return np.stack([xs, ys, np.array([f.z_at(x, y) for x, y in zip(xs, ys)]) + sign * f.half_z], axis=-1)


def band_move(f0: BandFrame, ray0: Ray, ray1: Ray) -> BandFrame:
    """Flecha del centro: mueve la banda en profundidad (cambia su posición en la rampa)."""
    origin = np.array([0.0, 0.0, f0.z0])
    s0, s1 = param_on_axis(ray0, origin, AXES["z"]), param_on_axis(ray1, origin, AXES["z"])
    if s0 is None or s1 is None:
        return f0
    return replace(f0, position=f0.position_of(f0.z0 + (s1 - s0)))


def band_pan(f0: BandFrame, ray1: Ray) -> BandFrame:
    """Tirador del borde derecho: el plano gira sobre el eje vertical para pasar por donde está el ratón."""
    hit = hit_plane(ray1, np.array([0.0, 0.0, f0.z0]), AXES["y"])
    if hit is None or abs(hit[0]) < BAND_MIN_SPAN:
        return f0
    angle = math.degrees(math.atan2(hit[2] - f0.z0, hit[0]))
    if hit[0] < 0:                                   # al otro lado del centro: el mismo plano visto desde el otro extremo
        angle = math.degrees(math.atan2(-(hit[2] - f0.z0), -hit[0]))
    return replace(f0, pan=float(np.clip(angle, -BAND_MAX_ANGLE, BAND_MAX_ANGLE)))


def band_tilt(f0: BandFrame, ray1: Ray) -> BandFrame:
    """Tirador del borde de arriba: el plano gira sobre el eje horizontal para pasar por donde está el ratón."""
    hit = hit_plane(ray1, np.array([0.0, 0.0, f0.z0]), AXES["x"])
    if hit is None or abs(hit[1]) < BAND_MIN_SPAN:
        return f0
    dz, dy = hit[2] - f0.z0, hit[1]
    angle = math.degrees(math.atan2(dz if dy > 0 else -dz, abs(dy)))
    return replace(f0, tilt=float(np.clip(angle, -BAND_MAX_ANGLE, BAND_MAX_ANGLE)))


def band_width(f0: BandFrame, ray0: Ray, ray1: Ray) -> BandFrame:
    """Tirador del grueso: arrastrado en profundidad cambia la anchura de la franja (0..1 de la rampa)."""
    origin = band_handles(f0)["width"]
    s0, s1 = param_on_axis(ray0, origin, AXES["z"]), param_on_axis(ray1, origin, AXES["z"])
    if s0 is None or s1 is None:
        return f0
    half = max(f0.half_z + (s1 - s0), 0.0)
    return replace(f0, width=float(np.clip(2.0 * half / f0.z_span, 0.0, 1.0)))


BAND_GESTURES = {"move": lambda f0, r0, r1: band_move(f0, r0, r1),
                 "pan": lambda f0, r0, r1: band_pan(f0, r1),
                 "tilt": lambda f0, r0, r1: band_tilt(f0, r1),
                 "width": lambda f0, r0, r1: band_width(f0, r0, r1)}
