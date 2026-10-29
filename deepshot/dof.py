"""Desenfoque de lente por profundidad (DOF), como el de una cámara real. Sin interfaz.

Qué hace: elige una profundidad «en foco» y desenfoca el resto según lo lejos que esté de ella, con un núcleo
con la forma del diafragma (círculo o polígono de palas), no con un gaussiano. La apertura se da en pasos f
como en una cámara: cuanto más abierta (f pequeño), mayor el desenfoque.

Cómo (resumen):
  1. Círculo de confusión (CoC) de cada píxel = distancia a la profundidad de enfoque (fuera de la zona nítida),
     escalada por la apertura, en píxeles y relativa al lado largo de la imagen (el aspecto no depende de la
     resolución: la vista previa se parece a la exportación).
  2. Se calculan varias capas con radios crecientes (0 … radio máximo). Cada capa es una convolución por FFT
     con el núcleo del diafragma, ponderada para que un primer plano nítido no «manche» de color el fondo
     desenfocado (sin halos) y para que las altas luces formen discos de bokeh más brillantes.
  3. Cada píxel mezcla las dos capas vecinas a su CoC.
Ojo de gato (`cat_eye`): en una lente real el borde de la imagen recorta los discos de bokeh y se vuelven almendras
(la lente los «corta» por la montura). Se imita con dos núcleos más: un disco recortado por los lados (para los bordes
izquierdo y derecho) y otro recortado arriba y abajo; cada píxel mezcla el núcleo redondo y los recortados según lo cerca
que esté de cada borde. Con `cat_eye` = 0 el resultado es idéntico al de siempre.
Las capas se calculan como mucho a `WORK_LONG_SIDE` píxeles y se amplían: el desenfoque es suave, no pierde nada,
y una exportación de 24 megapíxeles no tarda minutos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import fft as sfft

TARGETS = ("result", "photo")
APERTURES = (1.2, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0)      # pasos f habituales (referencia para la interfaz)
RADIUS_AT_F1 = 0.06            # radio máximo del desenfoque a f/1, como fracción del lado largo (f/2.8 → ~2 %)
WORK_LONG_SIDE = 1536          # lado largo máximo al calcular las capas de desenfoque
MIN_RADIUS_PX = 0.6            # por debajo de esto el desenfoque no se nota: se devuelve la imagen tal cual
MAX_LEVELS = 8


@dataclass
class LensBlur:
    """Ajustes del desenfoque de lente (se guardan en el proyecto)."""

    enabled: bool = False
    target: str = "result"          # "result" = la imagen final · "photo" = solo la foto, antes de mezclarla
    focus: float = 0.35             # profundidad en foco: 0 = lo más cercano, 1 = lo más lejano
    sharp_range: float = 0.08       # mitad de la zona nítida alrededor del foco (en profundidad 0..1)
    aperture: float = 2.8           # número f: menor = más abierta = más desenfoque
    near: float = 1.0               # multiplicador del desenfoque por delante del foco (0 = nada, 2 = el doble)
    far: float = 1.0                # ... y por detrás
    blades: int = 0                 # 0 = diafragma redondo · 3..12 = polígono con ese número de palas
    rotation: float = 0.0           # giro del polígono, en grados
    highlights: float = 0.4         # 0..1: cuánto brillan más las altas luces en el bokeh
    softness: float = 0.25          # 0..1: borde del disco (0 = duro y definido, 1 = difuso)
    cat_eye: float = 0.0            # 0..1: los discos de los bordes de la imagen se recortan en almendra («ojo de gato»)

    def validate(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"Destino del desenfoque desconocido: {self.target}")


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------
def max_radius(params: LensBlur, long_side: float) -> float:
    """Radio máximo del círculo de confusión, en píxeles, para una imagen con ese lado largo."""
    return RADIUS_AT_F1 / max(params.aperture, 0.5) * long_side


def coc_map(depth: np.ndarray, params: LensBlur) -> np.ndarray:
    """Radio del círculo de confusión de cada píxel, en píxeles (0 = nítido)."""
    long_side = float(max(depth.shape))
    focus = min(max(params.focus, 0.0), 1.0)
    sharp = min(max(params.sharp_range, 0.0), 0.95)
    # El lado más lejano al foco llega al radio máximo; el otro, proporcionalmente menos.
    norm = max(max(focus, 1.0 - focus) - sharp, 1e-3)
    signed = depth.astype(np.float32) - focus
    excess = np.clip(np.abs(signed) - sharp, 0.0, None) / norm
    side = np.where(signed >= 0, params.far, params.near).astype(np.float32)
    return np.clip(excess, 0.0, 1.0) * side * max_radius(params, long_side)


def focus_map(depth: np.ndarray, params: LensBlur, max_side: int = 512) -> np.ndarray:
    """Mapa de nitidez para enseñar qué zona queda en foco: (alto, ancho) uint8, 255 = nítido, 0 = muy desenfocado.

    Sirve de ayuda visual (se dibuja en verde sobre la imagen, sin exportarse). Se calcula a baja resolución y
    solo depende del foco, la zona nítida, la apertura y los multiplicadores delante / detrás."""
    step = max(1, int(math.ceil(max(depth.shape) / max_side)))
    small = np.asarray(depth, dtype=np.float32)[::step, ::step]
    coc = coc_map(small, params)
    reference = max(0.35 * max_radius(params, float(max(small.shape))), 1e-6)       # un 35 % del desenfoque máximo = nada de verde
    return np.round(255.0 * np.clip(1.0 - coc / reference, 0.0, 1.0)).astype(np.uint8)


CAT_EYE_SQUEEZE = 0.9           # cuánto se recorta el disco en el borde con el ojo de gato al máximo (1 = hasta una raya)


def aperture_kernel(radius: float, blades: int = 0, rotation_deg: float = 0.0, softness: float = 0.25,
                    squeeze: float = 0.0, axis: str = "x") -> np.ndarray:
    """Núcleo del diafragma: disco (o polígono de `blades` lados) de ese radio en píxeles, normalizado a suma 1.

    squeeze (0..1) recorta la forma como un ojo de gato: es la intersección de dos copias del diafragma separadas
    a lo largo de `axis` ("x" = recortado por los lados, "y" = por arriba y abajo)."""
    if radius < MIN_RADIUS_PX:
        return np.ones((1, 1), dtype=np.float32)
    half = int(math.ceil(radius + 1.0))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    width = max(1.0, min(max(softness, 0.0), 1.0) * radius * 0.6)         # ancho de la transición del borde

    def shape(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        dist = np.hypot(px, py)
        if blades >= 3:
            sector = 2.0 * math.pi / blades
            angle = np.mod(np.arctan2(py, px) - math.radians(rotation_deg), sector) - sector / 2.0
            edge = radius * math.cos(math.pi / blades) / np.cos(angle)   # distancia al lado del polígono
        else:
            edge = np.full_like(dist, radius)
        return np.clip((edge - dist) / width + 0.5, 0.0, 1.0)

    squeeze = min(max(squeeze, 0.0), 1.0)
    if squeeze > 0.0:
        shift = squeeze * radius / 2.0
        dx, dy = (shift, 0.0) if axis == "x" else (0.0, shift)
        kernel = np.minimum(shape(x - dx, y - dy), shape(x + dx, y + dy)).astype(np.float32)
    else:
        kernel = shape(x, y).astype(np.float32)
    total = float(kernel.sum())
    return kernel / total if total > 0 else np.ones((1, 1), dtype=np.float32)


def edge_weights(size: tuple[int, int], amount: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peso de cada núcleo en cada píxel para el ojo de gato: (redondo, recortado por los lados, recortado arriba y abajo).

    size = (alto, ancho). Los tres pesos suman 1. Cerca del centro todo es redondo; hacia los bordes izquierdo y derecho
    crece el recortado por los lados y hacia arriba y abajo el otro. `amount` (0..1) es la fuerza del efecto."""
    h, w = size
    amount = min(max(amount, 0.0), 1.0)
    xn = np.abs(2.0 * (np.arange(w, dtype=np.float32) + 0.5) / w - 1.0)[None, :]
    yn = np.abs(2.0 * (np.arange(h, dtype=np.float32) + 0.5) / h - 1.0)[:, None]

    def ramp(t: np.ndarray) -> np.ndarray:
        u = np.clip((t - 0.2) / 0.8, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    a_x = amount * ramp(xn) * np.ones((h, 1), dtype=np.float32)
    a_y = amount * ramp(yn) * np.ones((1, w), dtype=np.float32)
    total = a_x + a_y
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, 1e-6), 1.0).astype(np.float32)    # en las esquinas se reparten
    a_x, a_y = a_x * scale, a_y * scale
    return (1.0 - a_x - a_y).astype(np.float32), a_x.astype(np.float32), a_y.astype(np.float32)


def _scatter(stack: np.ndarray, radius: float, params: LensBlur,
             edge: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Convolución de `stack` con el núcleo del diafragma; con ojo de gato, mezcla el redondo y los recortados
    según el peso de cada uno en cada píxel de salida (`edge`, ver `edge_weights`)."""
    round_kernel = aperture_kernel(radius, params.blades, params.rotation, params.softness)
    if edge is None:
        return _convolve(stack, round_kernel)
    w_round, w_x, w_y = edge
    out = np.zeros_like(stack)
    for weight, kernel in ((w_round, round_kernel),
                           (w_x, aperture_kernel(radius, params.blades, params.rotation, params.softness,
                                                 CAT_EYE_SQUEEZE, "x")),
                           (w_y, aperture_kernel(radius, params.blades, params.rotation, params.softness,
                                                 CAT_EYE_SQUEEZE, "y"))):
        if weight.any():                                   # cada núcleo solo donde hace falta
            out += _convolve(stack, kernel) * weight[None]
    return out


def _resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Reescala (alto, ancho) o (alto, ancho, canales) float32 a size = (ancho, alto)."""
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32), mode="F").resize(size, Image.BILINEAR))
    return np.stack([_resize(arr[..., c], size) for c in range(arr.shape[-1])], axis=-1)


def _convolve(stack: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolución de cada plano de `stack` (C, H, W) con `kernel` por FFT, del mismo tamaño y sin desplazar."""
    c, h, w = stack.shape
    kh, kw = kernel.shape
    pad = (sfft.next_fast_len(h + kh - 1, real=True), sfft.next_fast_len(w + kw - 1, real=True))
    k_fft = sfft.rfft2(kernel, s=pad)
    out = sfft.irfft2(sfft.rfft2(stack, s=pad, axes=(1, 2), workers=-1) * k_fft[None], s=pad, axes=(1, 2), workers=-1)
    return out[:, kh // 2: kh // 2 + h, kw // 2: kw // 2 + w].astype(np.float32)


def _highlight_weight(image: np.ndarray, amount: float) -> np.ndarray:
    """Peso extra de las altas luces (hasta 25×): hacen discos de bokeh más brillantes que el resto."""
    if amount <= 0:
        return np.ones(image.shape[:2], dtype=np.float32)
    lum = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    t = np.clip((lum - 0.7) / 0.3, 0.0, 1.0)
    return (1.0 + 24.0 * min(amount, 1.0) * t * t * (3.0 - 2.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Desenfoque
# ---------------------------------------------------------------------------
def lens_blur(image: np.ndarray, depth: np.ndarray, params: LensBlur, levels: int = 6) -> np.ndarray:
    """Aplica el desenfoque de lente. image (alto, ancho, 3) float32 0..1; depth (alto, ancho) 0..1, 0 = cerca.

    Devuelve float32 del mismo tamaño. Si el desenfoque no se notaría (apertura muy cerrada, todo en foco),
    devuelve la imagen sin tocar. `levels`: número de capas de radio creciente (más = transiciones más finas).
    """
    params.validate()
    image = np.asarray(image, dtype=np.float32)
    depth = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    if image.shape[:2] != depth.shape:
        depth = _resize(depth, (image.shape[1], image.shape[0]))
    coc = coc_map(depth, params)
    if float(coc.max()) < MIN_RADIUS_PX:
        return image

    h, w = depth.shape
    scale = min(1.0, WORK_LONG_SIDE / max(h, w))
    work_size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    work_img = _resize(image, work_size) if scale < 1.0 else image
    work_coc = (_resize(coc, work_size) * scale) if scale < 1.0 else coc
    boost = _highlight_weight(work_img, params.highlights)
    edge = edge_weights(work_img.shape[:2], params.cat_eye) if params.cat_eye > 0 else None
    levels = int(min(max(levels, 2), MAX_LEVELS))
    r_max = float(work_coc.max())
    step = r_max / levels

    f = coc * scale / step                         # posición de cada píxel entre las capas, a la resolución de salida
    out = image * np.clip(1.0 - f, 0.0, 1.0)[..., None]          # capa 0: la imagen nítida
    for k in range(1, levels + 1):
        wk = np.clip(1.0 - np.abs(f - k), 0.0, 1.0)              # peso «triangular» de esta capa
        if not wk.any():
            continue
        radius = k * step
        # Solo cuentan los píxeles al menos así de desenfocados: lo nítido no tiñe el desenfoque de lo que tiene detrás.
        weight = np.clip(work_coc / radius, 0.0, 1.0) * boost
        stack = np.concatenate([work_img.transpose(2, 0, 1) * weight[None], weight[None]], axis=0)
        blurred = _scatter(stack, radius, params, edge)
        den = blurred[3]
        layer = blurred[:3] / np.maximum(den, 1e-4)[None]
        layer = np.where(den[None] > 1e-3, layer, work_img.transpose(2, 0, 1))      # sin datos cerca: la imagen
        layer = layer.transpose(1, 2, 0)
        if scale < 1.0:
            layer = _resize(layer, (w, h))
        out += layer * wk[..., None]
    if params.near > 0:
        out = _near_field(out, work_img, depth, coc, work_coc, boost, params, step, levels, scale, (w, h), edge)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _near_field(out: np.ndarray, work_img: np.ndarray, depth: np.ndarray, coc: np.ndarray, work_coc: np.ndarray,
                boost: np.ndarray, params: LensBlur, step: float, levels: int, scale: float,
                size: tuple[int, int], edge=None) -> np.ndarray:
    """Los objetos desenfocados que están DELANTE del foco se extienden sobre lo que tienen detrás, aunque eso
    esté nítido: su borde se ve suave y translúcido, como en una foto real (no un recorte de borde duro).

    Cada píxel cercano reparte su color entre las capas según su propio radio (como en el desenfoque normal) y
    cada capa lo esparce con su núcleo; se acumulan color y cobertura y se pone todo por encima del resultado.
    """
    w, h = size
    near = (depth < params.focus) & (coc >= MIN_RADIUS_PX)
    if not near.any():
        return out
    work_near = _resize(near.astype(np.float32), (work_img.shape[1], work_img.shape[0])) if scale < 1.0 else \
        near.astype(np.float32)
    f_work = work_coc / step
    premult = np.zeros((3, work_img.shape[0], work_img.shape[1]), dtype=np.float32)
    cover = np.zeros(work_img.shape[:2], dtype=np.float32)
    for k in range(1, levels + 1):
        share = np.clip(1.0 - np.abs(f_work - k), 0.0, 1.0) * work_near          # parte de cada píxel cercano en la capa k
        if not share.any():
            continue
        weight = share * boost
        stack = np.concatenate([work_img.transpose(2, 0, 1) * weight[None], share[None]], axis=0)
        blurred = _scatter(stack, k * step, params, edge)
        # Color con las altas luces reforzadas, normalizado por el peso de las luces; la cobertura es la real.
        weight_sum = _scatter(weight[None], k * step, params, edge)[0]
        color = blurred[:3] / np.maximum(weight_sum, 1e-4)[None]
        premult += color * blurred[3][None]
        cover += blurred[3]
    if float(cover.max()) <= 1e-4:
        return out
    cover_up = _resize(cover, (w, h)) if scale < 1.0 else cover
    premult_up = _resize(premult.transpose(1, 2, 0), (w, h)) if scale < 1.0 else premult.transpose(1, 2, 0)
    alpha = np.clip(cover_up, 0.0, 1.0)
    # Si varias capas se solapan (cobertura > 1) el color se normaliza para no pasarse de brillo.
    norm = np.where(cover_up > 1.0, 1.0 / np.maximum(cover_up, 1e-4), 1.0)

    # Debajo de un objeto cercano translúcido tiene que verse lo que hay detrás, no el propio objeto: se rellena
    # con el color de los píxeles NO cercanos de alrededor (donde no hay ninguno, queda el objeto, que ahí es opaco).
    keep = 1.0 - work_near
    fill_radius = float(work_coc[work_near > 0].max())
    stack = np.concatenate([work_img.transpose(2, 0, 1) * keep[None], keep[None]], axis=0)
    filled = _convolve(stack, aperture_kernel(fill_radius, 0, 0.0, 0.5))
    den = filled[3]
    behind = np.where(den[None] > 1e-2, filled[:3] / np.maximum(den, 1e-4)[None], work_img.transpose(2, 0, 1))
    behind = behind.transpose(1, 2, 0)
    behind_up = _resize(behind, (w, h)) if scale < 1.0 else behind
    near_f = near.astype(np.float32)[..., None]
    base = out * (1.0 - near_f) + behind_up * near_f
    return base * (1.0 - alpha)[..., None] + premult_up * norm[..., None]
