"""Control de profundidad por zonas (cerca / medio / lejos). Sin interfaz.

Qué hace: reasigna los valores del mapa de profundidad para mover cada zona por separado. Una zona tiene un
valor central (`center`, p. ej. 0,5 = lo que está a media distancia) y un valor de destino (`target`): todo lo
que hay en esa zona se desplaza lo que haga falta para que el centro quede en el destino. Así se puede acercar
el fondo sin tocar el primer plano, o separar más lo que está a media distancia.

Qué cambia: todo lo que lee la profundidad. La rampa, las bandas, el desenfoque de lente, las áreas 3D (sus
puntos se calculan con la profundidad ya movida) y la vista 3D. El mapa original no se modifica.

Cómo (resumen):
  1. Cada zona ocupa un tramo del eje de profundidad. En automático los tramos se reparten entre las zonas
     activas: el límite entre dos zonas está a medio camino entre sus centros, la primera llega a 0 y la última a 1.
     En manual el tramo se da con `range_min` y `range_max`.
  2. Dentro de su tramo la zona pesa 1 y fuera pesa 0. Los bordes se suavizan con `feather` (anchura de la
     transición, centrada en el límite), de modo que no se ven escalones.
  3. profundidad nueva = profundidad + Σ (destino − centro) × peso de cada zona, y se recorta a 0..1.
Con los pesos de un reparto automático la suma es 1 en todo el eje: una zona desplaza «su parte» y las vecinas la suya.

Zonas extra: además de cerca / medio / lejos se pueden añadir hasta `MAX_EXTRA` zonas más (nombres
«extra0», «extra1»…), con el mismo comportamiento: todas se reparten el eje por orden de centro.

Convención: igual que el mapa de la app, 0 = lo más cercano y 1 = lo más lejano.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

ZONE_NAMES = ("near", "mid", "far")
MAX_EXTRA = 3                      # zonas añadidas por el usuario además de las tres de fábrica


@dataclass
class DepthZone:
    """Una zona del eje de profundidad."""

    enabled: bool = True
    center: float = 0.5             # valor de profundidad que representa a la zona (0 = cerca, 1 = lejos)
    target: float = 0.5             # valor al que se lleva ese centro (si es igual al centro, la zona no se mueve)
    feather: float = 0.05           # anchura de la transición en los límites de la zona (0..0,5)
    auto_range: bool = True         # True: los límites salen de los centros vecinos
    range_min: float = 0.0          # límites manuales (solo con auto_range = False)
    range_max: float = 1.0

    @property
    def shift(self) -> float:
        return self.target - self.center


def _default_zones() -> dict[str, DepthZone]:
    return {
        "near": DepthZone(center=0.1, target=0.1),
        "mid": DepthZone(center=0.5, target=0.5),
        "far": DepthZone(center=0.9, target=0.9),
    }


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class DepthControl:
    """Tres zonas de profundidad editables por separado (se guardan en el proyecto)."""

    enabled: bool = False
    near: DepthZone = field(default_factory=lambda: _default_zones()["near"])
    mid: DepthZone = field(default_factory=lambda: _default_zones()["mid"])
    far: DepthZone = field(default_factory=lambda: _default_zones()["far"])
    extra: list[DepthZone] = field(default_factory=list)       # zonas añadidas (hasta MAX_EXTRA)

    # --- JSON -----------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict | None) -> "DepthControl":
        """Reconstruye desde el JSON del proyecto. Lo que falte o sobre se ignora (compatibilidad)."""
        data = data or {}
        zone_fields = {f.name for f in fields(DepthZone)}
        zones = {}
        for name in ZONE_NAMES:
            raw = data.get(name, {})
            zones[name] = DepthZone(**{**_default_zones()[name].__dict__,
                                       **{k: v for k, v in raw.items() if k in zone_fields}})
        extra = []
        raw_extra = data.get("extra")
        for raw in (raw_extra if isinstance(raw_extra, list) else [])[:MAX_EXTRA]:      # lo que no sea una lista se ignora
            if isinstance(raw, dict):
                extra.append(DepthZone(**{k: v for k, v in raw.items() if k in zone_fields}))
        return cls(enabled=bool(data.get("enabled", False)), extra=extra, **zones)

    def zones(self) -> list[DepthZone]:
        return [self.near, self.mid, self.far, *self.extra]

    def zone_names(self) -> list[str]:
        """Nombres de todas las zonas, en el orden de `zones()`."""
        return [*ZONE_NAMES, *(f"extra{i}" for i in range(len(self.extra)))]

    def zone(self, name: str) -> DepthZone:
        """Zona por nombre («near», «mid», «far» o «extraN»). KeyError si no existe."""
        if name in ZONE_NAMES:
            return getattr(self, name)
        if name.startswith("extra") and name[5:].isdigit() and int(name[5:]) < len(self.extra):
            return self.extra[int(name[5:])]
        raise KeyError(name)

    def add_zone(self) -> str | None:
        """Añade una zona con el centro en el hueco más ancho entre los centros actuales (sin cambiar la imagen:
        el destino es el centro). Devuelve su nombre, o None si ya están las `MAX_EXTRA`."""
        if len(self.extra) >= MAX_EXTRA:
            return None
        centers = sorted([0.0, 1.0] + [z.center for z in self.zones()])
        gap, center = max(((b - a), (a + b) / 2.0) for a, b in zip(centers, centers[1:]))
        self.extra.append(DepthZone(center=round(center, 4), target=round(center, 4)))
        return f"extra{len(self.extra) - 1}"

    def remove_zone(self, name: str) -> bool:
        """Quita una zona añadida. Las de fábrica (cerca / medio / lejos) no se quitan."""
        if name in ZONE_NAMES or not name.startswith("extra") or not name[5:].isdigit():
            return False
        index = int(name[5:])
        if index >= len(self.extra):
            return False
        del self.extra[index]
        return True

    def key(self) -> tuple:
        """Valor comparable de los ajustes (para saber si hay que recalcular una caché)."""
        return (self.enabled,) + tuple(
            (z.enabled, round(z.center, 5), round(z.target, 5), round(z.feather, 5), z.auto_range,
             round(z.range_min, 5), round(z.range_max, 5)) for z in self.zones())

    def is_identity(self) -> bool:
        """True si no cambia nada (apagado, sin zonas activas o con todos los destinos en su centro)."""
        return (not self.enabled) or all((not z.enabled) or abs(z.shift) < 1e-6 for z in self.zones())

    # --- Tramos -------------------------------------------------------------------
    def ranges(self) -> list[tuple[DepthZone, float, float]]:
        """(zona, desde, hasta) de cada zona activa. Las automáticas se reparten el eje por los puntos medios."""
        active = sorted((z for z in self.zones() if z.enabled), key=lambda z: z.center)
        out = []
        for i, z in enumerate(active):
            if z.auto_range:
                lo = 0.0 if i == 0 else (active[i - 1].center + z.center) / 2.0
                hi = 1.0 if i == len(active) - 1 else (z.center + active[i + 1].center) / 2.0
            else:
                lo, hi = sorted((z.range_min, z.range_max))
            out.append((z, lo, hi))
        return out

    # --- Aplicar ------------------------------------------------------------------
    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Devuelve la profundidad con las zonas movidas (float32, 0..1). Sin cambios, devuelve la misma matriz."""
        if self.is_identity():
            return depth
        d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
        out = d.copy()
        for zone, lo, hi in self.ranges():
            if abs(zone.shift) < 1e-6:
                continue
            out += np.float32(zone.shift) * self._weight(zone, lo, hi, d)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _weight(zone: DepthZone, lo: float, hi: float, d: np.ndarray) -> np.ndarray | float:
        """Peso de una zona en cada píxel: 1 dentro de su tramo, 0 fuera y suave en los bordes."""
        f = max(zone.feather, 1e-4)
        rise = 1.0 if lo <= 0.0 else _smoothstep(lo - f / 2.0, lo + f / 2.0, d)       # el borde en 0 o 1 está abierto
        fall = 1.0 if hi >= 1.0 else 1.0 - _smoothstep(hi - f / 2.0, hi + f / 2.0, d)
        return rise * fall

    def zone_weights(self, depth: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """(nombre, peso 0..1 por píxel) de cada zona activa, para mostrar qué parte del mapa es de cada una.
        Incluye las zonas cuyo destino es igual al centro (no mueven nada pero ocupan su tramo)."""
        d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
        names = {id(z): n for z, n in zip(self.zones(), self.zone_names())}
        return [(names[id(z)], np.broadcast_to(self._weight(z, lo, hi, d), d.shape).astype(np.float32))
                for z, lo, hi in self.ranges()]

    def set_centers_from_depth(self, depth: np.ndarray) -> None:
        """Pone los centros donde de verdad hay profundidad en este mapa (percentiles 10, 50 y 90) y los
        destinos en el mismo sitio: es el punto de partida para empezar a mover cada zona."""
        p10, p50, p90 = (float(v) for v in np.percentile(depth, (10, 50, 90)))
        for zone, c in zip(self.zones(), (p10, p50, p90)):
            zone.center = zone.target = round(c, 4)
