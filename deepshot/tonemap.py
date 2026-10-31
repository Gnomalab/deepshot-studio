"""Rampa artística y bandas: convierten la profundidad en tonos o colores.

Convención (igual que depth.py): profundidad 0 = cerca, 1 = lejos.
La rampa va de izquierda (posición 0 = cerca) a derecha (posición 1 = lejos), como
en el panel del add-on de referencia ("Left = near · right = far").

Todas las imágenes son float32 en 0..1; las de color tienen forma (alto, ancho, 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Color = tuple[float, float, float]
BLACK: Color = (0.0, 0.0, 0.0)
WHITE: Color = (1.0, 1.0, 1.0)

INTERPOLATIONS = ("linear", "constant", "ease")
BASES = ("ramp", "black", "white")   # sobre qué se aplican bandas y áreas


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Transición suave (curva S) de 0 a 1 entre edge0 y edge1. Admite edge0 > edge1."""
    if edge0 == edge1:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Rampa
# ---------------------------------------------------------------------------
@dataclass
class RampStop:
    position: float          # 0 = cerca, 1 = lejos
    color: Color             # RGB 0..1 (gris: los tres iguales)


@dataclass
class ColorRamp:
    stops: list[RampStop] = field(default_factory=lambda: [RampStop(0.0, WHITE), RampStop(1.0, BLACK)])
    interpolation: str = "linear"   # "linear" | "constant" | "ease"
    # Rango de profundidad que ocupa la rampa ("Fit ramp to relief" lo ajusta al real).
    range_min: float = 0.0
    range_max: float = 1.0

    def sorted_stops(self) -> list[RampStop]:
        return sorted(self.stops, key=lambda s: s.position)

    def normalize(self, depth: np.ndarray) -> np.ndarray:
        """Profundidad -> posición en la rampa (0..1) según el rango."""
        span = max(self.range_max - self.range_min, 1e-6)
        return np.clip((depth - self.range_min) / span, 0.0, 1.0).astype(np.float32)

    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Aplica la rampa: (alto, ancho) -> (alto, ancho, 3)."""
        t = self.normalize(depth)
        stops = self.sorted_stops()
        if not stops:
            return np.zeros(t.shape + (3,), dtype=np.float32)
        pos = np.array([s.position for s in stops], dtype=np.float32)
        cols = np.array([s.color for s in stops], dtype=np.float32)       # (n, 3)
        if len(stops) == 1:
            return np.broadcast_to(cols[0], t.shape + (3,)).astype(np.float32).copy()

        # Tramo de cada píxel: índice de la parada a su izquierda.
        idx = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(pos) - 2)
        p0, p1 = pos[idx], pos[idx + 1]
        local = np.clip((t - p0) / np.maximum(p1 - p0, 1e-6), 0.0, 1.0)
        # Antes de la primera parada y después de la última: color de esa parada.
        local = np.where(t <= pos[0], 0.0, np.where(t >= pos[-1], 1.0, local)).astype(np.float32)
        if self.interpolation == "constant":
            local = (local >= 1.0).astype(np.float32)
        elif self.interpolation == "ease":
            local = local * local * (3.0 - 2.0 * local)
        c0, c1 = cols[idx], cols[idx + 1]
        return (c0 + (c1 - c0) * local[..., None]).astype(np.float32)

    def fit_to_relief(self, depth: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> None:
        """Ajusta el rango de la rampa al rango real de la profundidad (ignora extremos aislados)."""
        lo, hi = np.percentile(depth, [low_pct, high_pct])
        self.range_min, self.range_max = float(lo), float(max(hi, lo + 1e-6))


def ramp_preset(name: str) -> ColorRamp:
    """Presets de partida inspirados en los estilos de referencia."""
    presets = {
        # Neblina: cerca casi blanco, lejos gris claro, transición suave.
        "haze": ColorRamp([RampStop(0.0, (0.95, 0.95, 0.95)), RampStop(0.55, (0.75, 0.75, 0.75)),
                           RampStop(1.0, (0.45, 0.45, 0.45))], "ease"),
        # Contraste fuerte: cerca blanco, caída rápida a negro.
        "contrast": ColorRamp([RampStop(0.0, WHITE), RampStop(0.35, (0.85, 0.85, 0.85)),
                               RampStop(0.55, BLACK), RampStop(1.0, BLACK)], "linear"),
        # Color: cerca rojo, lejos azul.
        "color": ColorRamp([RampStop(0.0, (0.95, 0.20, 0.08)), RampStop(0.5, (0.55, 0.15, 0.45)),
                            RampStop(1.0, (0.10, 0.25, 0.85))], "linear"),
        # Lineal simple: cerca blanco, lejos negro.
        "linear": ColorRamp(),
    }
    if name not in presets:
        raise KeyError(f"Preset de rampa desconocido: {name}")
    return presets[name]


# ---------------------------------------------------------------------------
# Bandas
# ---------------------------------------------------------------------------
@dataclass
class Band:
    """Franja de profundidad que se ilumina (o se oscurece) sobre la base.

    position:  centro de la franja en la escala de la rampa (0 = cerca, 1 = lejos).
    width:     anchura total de la parte plena (0..1).
    softness:  anchura de la transición a cada lado (0 = borde duro).
    color:     color de la banda (blanco = luz).
    strength:  0..1 opacidad de la banda.
    pan:       giro de la franja sobre el eje vertical, en grados: en vez de un plano de profundidad constante
               («paralelo a la cámara»), la franja es un plano que se aleja hacia la derecha (pan > 0) o hacia
               la izquierda (pan < 0). La posición sigue siendo la del centro de la imagen.
    tilt:      igual sobre el eje horizontal: arriba más lejos (tilt > 0) o más cerca (tilt < 0).
    """

    position: float = 0.5
    width: float = 0.1
    softness: float = 0.05
    color: Color = WHITE
    strength: float = 1.0
    enabled: bool = True
    pan: float = 0.0
    tilt: float = 0.0
    name: str = ""                      # nombre elegido por el usuario ("" = «Banda N»)
    color_from_ramp: bool = False       # el color de la banda es el de la rampa en su posición (en vez de `color`)

    def mask(self, t: np.ndarray) -> np.ndarray:
        """Máscara 0..1 de la banda sobre posiciones de rampa t (0..1)."""
        dist = np.abs(t - self.position)
        half = self.width / 2.0
        if self.softness <= 0:
            return (dist <= half).astype(np.float32)
        return smoothstep(half + self.softness, half, dist)


def base_image(base: str, ramp: ColorRamp, depth: np.ndarray) -> np.ndarray:
    """Imagen sobre la que se aplican bandas o áreas: la rampa, negro o blanco."""
    if base == "ramp":
        return ramp.apply(depth)
    value = 0.0 if base == "black" else 1.0
    if base not in BASES:
        raise ValueError(f"Base desconocida: {base}")
    return np.full(depth.shape + (3,), value, dtype=np.float32)


MAX_BAND_ANGLE = 80.0           # límite del pan y el tilt: más allá la tangente se dispara


def tilted_depth(depth: np.ndarray, pan: float, tilt: float, relief: float) -> np.ndarray:
    """Profundidad vista desde un plano girado (pan sobre el eje vertical, tilt sobre el horizontal).

    Escena: x ∈ ±aspecto, y ∈ ±1 (arriba +), z = profundidad × relieve. El plano que pasa por el centro de la imagen
    es z = z0 + x·tan(pan) + y·tan(tilt); se le resta a la profundidad (en unidades de profundidad) para que una
    banda de profundidad constante en el resultado sea ese plano inclinado en la escena."""
    h, w = depth.shape
    pan = min(max(pan, -MAX_BAND_ANGLE), MAX_BAND_ANGLE)
    tilt = min(max(tilt, -MAX_BAND_ANGLE), MAX_BAND_ANGLE)
    x = ((np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0) * (w / h)
    y = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0
    shift = x[None, :] * np.float32(np.tan(np.radians(pan))) + y[:, None] * np.float32(np.tan(np.radians(tilt)))
    return depth - shift / max(float(relief), 1e-6)


def ramp_color_at(ramp: ColorRamp, position: float) -> tuple[float, float, float]:
    """Color de la rampa en una posición de su escala (0 = cerca, 1 = lejos)."""
    value = ramp.range_min + float(position) * (ramp.range_max - ramp.range_min)
    rgb = ramp.apply(np.array([[value]], dtype=np.float32))[0, 0]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def band_color(band: "Band", ramp: ColorRamp) -> tuple[float, float, float]:
    """El color con que se pinta una banda: el suyo, o el de la rampa en su posición si así se pidió."""
    return ramp_color_at(ramp, band.position) if band.color_from_ramp else tuple(band.color)


def apply_bands(image: np.ndarray, bands: list[Band], ramp: ColorRamp, depth: np.ndarray,
                relief: float = 1.0) -> np.ndarray:
    """Pinta las bandas una tras otra sobre `image` (mezcla normal por opacidad). `relief` es el de la escena:
    solo cuenta para las bandas con pan o tilt (cuánto se inclina el plano respecto a la profundidad)."""
    out = image.copy()
    t = ramp.normalize(depth)   # las bandas usan la misma escala que la rampa
    for band in bands:
        if not band.enabled or band.strength <= 0:
            continue
        if band.pan == 0.0 and band.tilt == 0.0:
            tb = t
        else:        # sin recortar a 0..1: lo que el plano inclinado deja fuera del rango no debe pegarse a los extremos
            span = max(ramp.range_max - ramp.range_min, 1e-6)
            tb = ((tilted_depth(depth, band.pan, band.tilt, relief) - ramp.range_min) / span).astype(np.float32)
        a = (band.mask(tb) * band.strength)[..., None]
        out = out + (np.asarray(band_color(band, ramp), dtype=np.float32) - out) * a
    return out.astype(np.float32)
