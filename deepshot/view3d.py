"""Vista 3D del relieve: cámara orbital y dibujo de la nube de puntos con numpy. Sin interfaz.

Convención (la misma que `areas.Scene`): x a la derecha, y arriba, z hacia el fondo (z = profundidad × relieve).
La cámara de la vista 2D mira desde z < 0 hacia +z; con yaw = 0 y pitch = 0 la vista 3D coincide con ella.

El dibujo es por "salpicado" de puntos (splatting): se proyecta cada punto de una rejilla del mapa, se ordenan de lejos
a cerca y se pintan en un búfer; los más cercanos se pintan los últimos y tapan a los lejanos. No usa GPU ni
dependencias nuevas, vale igual en Windows y Mac y se puede probar sin ventana.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAX_POINTS = 90_000          # tope de puntos de la rejilla por defecto (nivel «Media»)

# Niveles de «Calidad 3D»: puntos de la rejilla (la foto tiene tantos o más; la rejilla nunca pasa de su tamaño).
# Medido a 1000 × 700 px con una foto de 1372 × 771 (2026-10-19): dibujo completo 0,14 s con 40.000 puntos,
# 0,17 s con 90.000, 0,27 s con 250.000, 0,40 s con 600.000 y 0,66 s con 1.000.000.
QUALITY_POINTS = {"low": 40_000, "medium": 90_000, "high": 250_000, "ultra": 600_000, "max": 1_000_000}
AUTO = "auto"
AUTO_BUDGET_S = 0.45         # en modo automático: el dibujo completo no debe pasar de este tiempo
AUTO_FIRST = "medium"        # nivel con el que se mide el coste antes de elegir
FAST_POINTS = 120_000        # puntos que se dibujan mientras se mueve la cámara (fluido sea cual sea la calidad)


def auto_points(seconds_per_point: float, available: int) -> int:
    """Nivel del modo automático: el más alto cuyo dibujo completo estimado cabe en `AUTO_BUDGET_S`.

    seconds_per_point: coste medido por punto en esta máquina · available: puntos que tiene la foto.
    Nunca baja de «Media» (el nivel de partida) ni sube de «Muy alta»."""
    best = QUALITY_POINTS[AUTO_FIRST]
    for level in ("high", "ultra"):
        points = QUALITY_POINTS[level]
        if points <= available and points * seconds_per_point <= AUTO_BUDGET_S:
            best = points
    return best
MIN_PITCH, MAX_PITCH = -89.0, 89.0
BACKGROUND = (29, 29, 29, 255)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


@dataclass
class OrbitCamera:
    """Cámara que gira alrededor de un punto (`target`), como la órbita de Blender."""

    yaw: float = 28.0          # grados; 0 = de frente (como la vista 2D), positivo = hacia la derecha
    pitch: float = 16.0        # grados; positivo = mirando desde arriba
    distance: float = 3.6      # distancia al objetivo, en unidades de la escena
    fov_deg: float = 40.0      # campo de visión vertical de la cámara 3D
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5], dtype=np.float64))

    # --- Geometría -----------------------------------------------------------
    def position(self) -> np.ndarray:
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        offset = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
        return self.target + self.distance * offset

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(derecha, arriba, adelante) de la cámara, en coordenadas de la escena."""
        forward = _unit(self.target - self.position())
        right = _unit(np.cross(np.array([0.0, 1.0, 0.0]), forward))
        up = np.cross(forward, right)
        return right, up, forward

    def project(self, points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Puntos (N, 3) → (x, y) en píxeles de la pantalla y z de cámara (distancia por delante; > 0 = visible)."""
        right, up, forward = self.basis()
        rel = points.reshape(-1, 3).astype(np.float64) - self.position()
        zc = rel @ forward
        xc, yc = rel @ right, rel @ up
        focal = (height / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        safe = np.where(zc > 1e-6, zc, np.nan)           # detrás de la cámara: no se dibuja
        return width / 2.0 + xc / safe * focal, height / 2.0 - yc / safe * focal, zc

    # --- Movimientos del ratón --------------------------------------------------
    def orbit(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw = (self.yaw + d_yaw + 180.0) % 360.0 - 180.0
        self.pitch = min(max(self.pitch + d_pitch, MIN_PITCH), MAX_PITCH)

    def zoom(self, factor: float) -> None:
        """factor > 1 acerca la cámara."""
        self.distance = min(max(self.distance / factor, 0.4), 40.0)

    def pan(self, dx_px: float, dy_px: float, height: int) -> None:
        """Desplaza el objetivo en el plano de la pantalla (arrastrar la escena con el ratón)."""
        right, up, _ = self.basis()
        world_per_px = 2.0 * self.distance * math.tan(math.radians(self.fov_deg) / 2.0) / max(height, 1)
        self.target = self.target - right * dx_px * world_per_px + up * dy_px * world_per_px

    def set_view(self, yaw: float, pitch: float) -> None:
        self.yaw, self.pitch = yaw, min(max(pitch, MIN_PITCH), MAX_PITCH)

    def reset(self, depth_center: float = 0.5) -> None:
        self.yaw, self.pitch, self.distance = 28.0, 16.0, 3.6
        self.target = np.array([0.0, 0.0, depth_center], dtype=np.float64)

    def copy(self) -> "OrbitCamera":
        return OrbitCamera(self.yaw, self.pitch, self.distance, self.fov_deg, self.target.copy())

    def key(self) -> tuple:
        """Valor comparable del estado, para saber si hay que volver a dibujar."""
        return (round(self.yaw, 3), round(self.pitch, 3), round(self.distance, 4), self.fov_deg,
                tuple(np.round(self.target, 4)))


# ---------------------------------------------------------------------------
# Rejilla del relieve
# ---------------------------------------------------------------------------
def band_plane(position: float, pan: float, tilt: float, ramp_min: float, ramp_max: float, relief: float,
               aspect: float) -> np.ndarray:
    """Las cuatro esquinas (4, 3) del plano de una banda en la escena: z = z0 + x·tan(pan) + y·tan(tilt), con z0 la
    profundidad del centro de la banda (su posición en la escala de la rampa, por el relieve). Es el mismo plano que
    usa `tonemap.tilted_depth` para colorear la franja, así que lo que se ve coincide con el efecto."""
    z0 = (ramp_min + float(position) * (ramp_max - ramp_min)) * float(relief)
    tp, tt = np.tan(np.radians(pan)), np.tan(np.radians(tilt))
    xs = np.array([-aspect, aspect, aspect, -aspect], dtype=np.float64)
    ys = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    return np.stack([xs, ys, z0 + xs * tp + ys * tt], axis=-1)


def grid_size(width: int, height: int, max_points: int = MAX_POINTS) -> tuple[int, int]:
    """(columnas, filas) de la rejilla con la misma proporción que la foto y como mucho `max_points`."""
    scale = min(1.0, math.sqrt(max_points / max(width * height, 1)))
    return max(8, int(round(width * scale))), max(8, int(round(height * scale)))


def shade(colors: np.ndarray, z: np.ndarray, aspect: float, strength: float = 0.4) -> np.ndarray:
    """Sombreado Lambert del relieve para que se lean las formas aunque la textura sea plana.

    colors: (filas, columnas, 3) uint8 · z: (filas, columnas) en unidades de la escena.
    Una superficie plana mirando a la cámara queda como está; las inclinadas se oscurecen o aclaran.
    """
    rows, cols = z.shape
    if strength <= 0 or rows < 2 or cols < 2:
        return colors
    step_x, step_y = 2.0 * aspect / cols, 2.0 / rows
    d_row, d_col = np.gradient(z.astype(np.float64))
    # La normal apunta hacia la cámara (-z); y crece hacia arriba y las filas hacia abajo.
    normal = np.stack([d_col / step_x, -d_row / step_y, -np.ones_like(z, dtype=np.float64)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = _unit(np.array([-0.4, 0.6, -0.7]))
    lam = np.clip(normal @ light, 0.0, 1.0)
    flat = float(np.dot(np.array([0.0, 0.0, -1.0]), light))
    factor = np.clip(1.0 - strength + strength * lam / flat, 0.0, 1.5)
    return np.clip(colors.astype(np.float64) * factor[..., None], 0, 255).astype(np.uint8)


def cliff_points(points: np.ndarray, colors: np.ndarray, max_total: int = 120_000) -> tuple[np.ndarray, np.ndarray]:
    """Puntos intermedios para tapar los «acantilados» del relieve.

    Entre dos puntos vecinos con mucha diferencia de profundidad (el borde de una persona sobre el fondo) la
    superficie real es una pared casi vertical que, con solo los puntos de la rejilla, queda llena de agujeros.
    Se añaden puntos a lo largo de esa pared, con el color del extremo más cercano a la cámara (el primer plano
    «chorrea» hacia el fondo, como en un relieve de papel).

    Devuelve (puntos (M, 3), colores (M, 3) uint8); M = 0 si el relieve no tiene saltos.
    """
    rows, cols = points.shape[:2]
    step = max(2.0 * points[..., 0].max() / max(cols, 1), 1e-6)       # separación x entre columnas ≈ separación y
    pts_out, col_out = [], []
    for a, b, ca, cb in (
        (points[:, :-1], points[:, 1:], colors[:, :-1], colors[:, 1:]),     # vecinos horizontales
        (points[:-1, :], points[1:, :], colors[:-1, :], colors[1:, :]),     # vecinos verticales
    ):
        gap = np.linalg.norm((b - a).reshape(-1, 3), axis=-1)
        n = np.minimum(np.ceil(gap / (1.1 * step)) - 1, 24).astype(np.int64)    # puntos nuevos por par
        n[gap <= 1.1 * step] = 0
        pairs = np.flatnonzero(n > 0)
        if pairs.size == 0:
            continue
        counts = n[pairs]
        total = int(counts.sum())
        if total > max_total // 2:                    # demasiados saltos (foto muy ruidosa): se aclara
            keep = np.random.default_rng(0).random(pairs.size) < (max_total // 2) / total
            pairs, counts = pairs[keep], counts[keep]
            total = int(counts.sum())
            if total == 0:
                continue
        pa, pb = a.reshape(-1, 3)[pairs], b.reshape(-1, 3)[pairs]
        which = np.repeat(np.arange(pairs.size), counts)
        # Posición de cada punto nuevo a lo largo del par: 1/(n+1), 2/(n+1), ...
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        t = ((offsets + 1) / (counts[which] + 1))[:, None]
        pts_out.append(pa[which] * (1 - t) + pb[which] * t)
        near_is_a = pa[:, 2] <= pb[:, 2]               # z menor = más cerca
        chosen = np.where(near_is_a[:, None], ca.reshape(-1, 3)[pairs], cb.reshape(-1, 3)[pairs])
        col_out.append(chosen[which])
    if not pts_out:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(pts_out), np.concatenate(col_out).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def render_cloud(points: np.ndarray, colors: np.ndarray, camera: OrbitCamera, width: int, height: int,
                 background: tuple[int, int, int, int] = BACKGROUND,
                 extra: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Dibuja la rejilla de puntos vista desde la cámara y devuelve una imagen RGBA (alto, ancho, 4) uint8.

    points: (filas, columnas, 3) · colors: (filas, columnas, 3) uint8.
    Cada punto se pinta como un cuadrado del tamaño del hueco medio entre puntos vecinos en pantalla,
    así la superficie no se llena de agujeros al acercarse. `extra` = (puntos (M, 3), colores (M, 3)) sueltos
    que se dibujan además (ver `cliff_points`).
    """
    width, height = max(int(width), 1), max(int(height), 1)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:] = background
    rows, cols = points.shape[:2]
    sx, sy, zc = camera.project(points, width, height)
    flat_colors = colors.reshape(-1, 3)
    n_grid = sx.size
    if extra is not None and len(extra[0]):
        ex, ey, ez = camera.project(extra[0], width, height)
        sx, sy, zc = np.concatenate([sx, ex]), np.concatenate([sy, ey]), np.concatenate([zc, ez])
        flat_colors = np.concatenate([flat_colors, extra[1]])
    visible = np.isfinite(sx) & np.isfinite(sy)
    if not visible.any():
        return out

    # Tamaño del salpicado: separación en pantalla entre puntos vecinos de la rejilla. Se toma un valor algo
    # mayor que la mediana (las zonas inclinadas se abren más) pero sin pasar del doble, para no engordar los bordes.
    gx = sx[:n_grid].reshape(rows, cols)
    gy = sy[:n_grid].reshape(rows, cols)
    gaps = [np.hypot(np.diff(gx, axis=ax), np.diff(gy, axis=ax)).ravel() for ax in (1, 0) if gx.shape[ax] > 1]
    gaps = np.concatenate(gaps) if gaps else np.array([1.0])
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size:
        median = float(np.median(gaps))
        gap = min(float(np.percentile(gaps, 85)), 1.5 * median)
    else:
        gap = 1.0
    size = int(min(max(math.ceil(gap * 1.1), 1), 8))

    # De lejos a cerca: al repetirse un píxel gana el último escrito, es decir, el punto más cercano.
    idx = np.flatnonzero(visible)
    idx = idx[np.argsort(-zc[idx], kind="stable")]
    base_x = np.floor(sx[idx] - size / 2.0 + 0.5).astype(np.int64)
    base_y = np.floor(sy[idx] - size / 2.0 + 0.5).astype(np.int64)
    z, rgb = zc[idx], flat_colors[idx]
    zbuf = np.full(width * height, np.inf)           # profundidad ya pintada en cada píxel
    flat_out = out.reshape(-1, 4)
    for dy in range(size):                            # cada punto cubre un cuadrado size × size de píxeles
        for dx in range(size):
            x, y = base_x + dx, base_y + dy
            ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            pix = y[ok] * width + x[ok]
            near = z[ok] < zbuf[pix]                  # solo si está más cerca de lo que ya hay en ese píxel
            pix = pix[near]
            zbuf[pix] = z[ok][near]
            flat_out[pix, :3] = rgb[ok][near]
            flat_out[pix, 3] = 255
    return out
