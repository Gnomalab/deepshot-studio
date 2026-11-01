"""Exportación a resolución completa: el efecto, la imagen final (con el Look) o solo el mapa de profundidad.

Formatos del efecto y del mapa (los tres que ofrece la técnica de referencia):
  - PNG de 8 bits   (Pillow).
  - PNG de 16 bits  (escritor propio: Pillow no guarda PNG de 16 bits en color, solo en gris).
  - EXR de 32 bits  (OpenEXR, licencia BSD-3-Clause; coma flotante, sin pérdida).
La imagen final, además, sale en JPG, TIFF de 8 o 16 bits y WebP, a un tamaño elegido (porcentaje, ancho, alto o lado
largo), con su resolución en ppp y los metadatos EXIF de la foto si se quiere.

Sin interfaz: recibe números y rutas, y puede probarse sola. La ventana la llama desde un hilo.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication

import logging
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from deepshot.depth import load_photo
from deepshot.depth_control import DepthControl
from deepshot.depth_view import PALETTES, colorize_float
from deepshot.maps import resize as resize_map
from deepshot.project import Project, render

log = logging.getLogger(__name__)

FORMATS = ("png8", "png16", "exr32", "jpg", "tiff8", "tiff16", "webp")
FINAL_FORMATS = ("jpg", "tiff8", "tiff16", "png8", "png16", "webp")        # los de la imagen final (sin EXR)
EXTENSIONS = {"png8": ".png", "png16": ".png", "exr32": ".exr", "jpg": ".jpg", "tiff8": ".tif", "tiff16": ".tif",
              "webp": ".webp"}
CONTENTS = ("effect", "final", "depth")
SIZE_MODES = ("original", "percent", "width", "height", "long_side")
MAX_SIDE = 30000                                                              # lado máximo de la imagen exportada
Progress = Callable[[str, float], None]


@dataclass
class ExportOptions:
    content: str = "effect"        # "effect" (efecto, sin Look), "final" (efecto con el Look) o "depth" (solo el mapa)
    format: str = "png16"          # ver FORMATS
    folder: Path = Path(".")
    name: str = "imagen"           # nombre del archivo sin extensión
    near_bright: bool = True       # mapa de profundidad: cerca = claro (como se ve en la app)
    depth_zones: bool = False      # mapa de profundidad: con las zonas movidas del control de profundidad
    depth_palette: str = ""        # mapa de profundidad: "" = gris; o una paleta de `depth_view.PALETTES` (en color)
    save_settings: bool = True     # guardar junto al efecto el JSON con los ajustes
    quality: int = 92              # JPG y WebP: 1..100
    size_mode: str = "original"    # imagen final: "original" | "percent" | "width" | "height" | "long_side"
    size_value: float = 100.0      # porcentaje (percent) o píxeles (width, height, long_side)
    dpi: int = 300                 # imagen final: resolución guardada en el archivo (no cambia los píxeles)
    keep_metadata: bool = True     # imagen final: copiar los metadatos EXIF de la foto

    def validate(self) -> None:
        if self.content not in CONTENTS:
            raise ValueError(QCoreApplication.translate("Export", "Contenido de exportación desconocido: {0}").format(self.content))
        if self.format not in FORMATS:
            raise ValueError(QCoreApplication.translate("Export", "Formato de exportación desconocido: {0}").format(self.format))
        if self.content == "final" and self.format not in FINAL_FORMATS:
            raise ValueError(QCoreApplication.translate("Export", "Ese formato no sirve para la imagen final: {0}").format(self.format))
        if self.size_mode not in SIZE_MODES:
            raise ValueError(QCoreApplication.translate("Export", "Modo de tamaño desconocido: {0}").format(self.size_mode))
        if self.size_mode != "original" and self.size_value <= 0:
            raise ValueError(QCoreApplication.translate("Export", "El tamaño tiene que ser mayor que cero."))
        if not self.name.strip():
            raise ValueError(QCoreApplication.translate("Export", "El nombre del archivo está vacío."))
        if self.depth_palette and self.depth_palette not in PALETTES:
            raise ValueError(QCoreApplication.translate("Export", "Paleta del mapa de profundidad desconocida: {0}").format(self.depth_palette))


def _no_progress(_message: str, _fraction: float) -> None:
    pass


# ---------------------------------------------------------------------------
# Escritura de archivos
# ---------------------------------------------------------------------------
def free_path(folder: Path, name: str, extension: str) -> Path:
    """Ruta que no pisa un archivo existente: si «nombre.png» existe, usa «nombre_2.png», etc."""
    safe = "".join("_" if c in '\\/:*?"<>|' else c for c in name).strip(" .") or "image"
    path = folder / f"{safe}{extension}"
    n = 2
    while path.exists():
        path = folder / f"{safe}_{n}{extension}"
        n += 1
    return path


def write_png16(path: Path, data: np.ndarray) -> None:
    """PNG de 16 bits por canal, en gris (alto, ancho) o en color RGB (alto, ancho, 3), de uint16.

    Se escribe por bloques de filas para no duplicar en memoria imágenes de decenas de megapíxeles.
    """
    if data.dtype != np.uint16 or data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[2] != 3):
        raise ValueError(QCoreApplication.translate("Export", "Se esperaba un array uint16 de forma (alto, ancho) o (alto, ancho, 3)."))
    h, w = data.shape[:2]
    color_type = 0 if data.ndim == 2 else 2

    def chunk(f, tag: bytes, payload: bytes) -> None:
        f.write(struct.pack(">I", len(payload)) + tag + payload)
        f.write(struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        chunk(f, b"IHDR", struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0))
        packer = zlib.compressobj(4)
        step = 64
        for y0 in range(0, h, step):
            rows = np.ascontiguousarray(data[y0:y0 + step].astype(">u2"))
            n = rows.shape[0]
            raw = np.zeros((n, 1 + rows[0].nbytes), dtype=np.uint8)   # byte 0 de cada fila: filtro «ninguno»
            raw[:, 1:] = rows.reshape(n, -1).view(np.uint8)
            piece = packer.compress(raw.tobytes())
            if piece:
                chunk(f, b"IDAT", piece)
        chunk(f, b"IDAT", packer.flush())
        chunk(f, b"IEND", b"")


def _is_gray(image: np.ndarray) -> bool:
    return image.ndim == 2 or bool(np.abs(image[..., 0] - image[..., 1]).max() < 1e-6
                                   and np.abs(image[..., 1] - image[..., 2]).max() < 1e-6)


def save_image(image: np.ndarray, path: Path, fmt: str) -> None:
    """Guarda una imagen float32 en 0..1, (alto, ancho) en gris o (alto, ancho, 3) en color.

    Una imagen en blanco y negro se guarda en gris (más pequeña y es lo que esperan los programas de
    retoque); en EXR siempre se guarda con tres canales RGB (excepto los mapas de un solo canal).
    """
    if fmt not in FORMATS:
        raise ValueError(f"Formato desconocido: {fmt}")
    image = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0) if fmt != "exr32" else image.astype(np.float32)
    gray = _is_gray(image)
    if fmt == "png8":
        data = np.rint((image[..., 0] if (gray and image.ndim == 3) else image) * 255.0).astype(np.uint8)
        Image.fromarray(data).save(path)
    elif fmt == "png16":
        data = np.rint((image[..., 0] if (gray and image.ndim == 3) else image) * 65535.0).astype(np.uint16)
        write_png16(path, data)
    else:
        import OpenEXR

        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        channels = {"Z": np.ascontiguousarray(image)} if image.ndim == 2 else {"RGB": np.ascontiguousarray(image)}
        with OpenEXR.File(header, channels) as out:
            out.write(str(path))


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------
def target_size(width: int, height: int, mode: str, value: float) -> tuple[int, int]:
    """Tamaño final (ancho, alto) según el modo, conservando la proporción. Mínimo 1 px y máximo `MAX_SIDE`."""
    if mode == "percent":
        scale = value / 100.0
    elif mode == "width":
        scale = value / width
    elif mode == "height":
        scale = value / height
    elif mode == "long_side":
        scale = value / max(width, height)
    else:
        return width, height
    scale = min(scale, MAX_SIDE / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Cambia el tamaño de una imagen float32 (alto, ancho, 3) con Lanczos, canal a canal y sin perder precisión."""
    h, w = image.shape[:2]
    if (w, h) == size:
        return image
    channels = [np.asarray(Image.fromarray(np.ascontiguousarray(image[..., c]), mode="F").resize(size, Image.LANCZOS))
                for c in range(image.shape[2])]
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0).astype(np.float32)


def write_tiff16(path: Path, data: np.ndarray, dpi: int = 300) -> None:
    """TIFF de 16 bits por canal en RGB, sin compresión (Pillow no escribe TIFF RGB de 16 bits).

    data: uint16 (alto, ancho, 3). Se escribe por tiras de filas para no duplicar la imagen en memoria."""
    if data.dtype != np.uint16 or data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(QCoreApplication.translate("Export", "Se esperaba un array uint16 de forma (alto, ancho, 3)."))
    h, w = data.shape[:2]
    row_bytes = w * 6
    rows_per_strip = max(1, (1 << 20) // row_bytes)
    strips = [(y, min(rows_per_strip, h - y)) for y in range(0, h, rows_per_strip)]
    offsets, counts, pos = [], [], 8                 # los datos de la imagen empiezan justo después de la cabecera
    for _y, n in strips:
        offsets.append(pos)
        counts.append(n * row_bytes)
        pos += n * row_bytes
    if pos % 2:
        pos += 1
    extra = pos                                      # tras los píxeles: los valores que no caben en una entrada
    bits_off = extra
    xres_off = bits_off + 6
    yres_off = xres_off + 8
    offs_off = yres_off + 8
    cnts_off = offs_off + 4 * len(strips)
    ifd_off = cnts_off + 4 * len(strips)

    def entry(tag: int, kind: int, count: int, value: int) -> bytes:
        return struct.pack("<HHII", tag, kind, count, value)

    SHORT, LONG, RATIONAL = 3, 4, 5
    many = len(strips) > 1
    entries = [
        entry(256, LONG, 1, w), entry(257, LONG, 1, h), entry(258, SHORT, 3, bits_off), entry(262, SHORT, 1, 2),
        entry(273, LONG, len(strips), offs_off if many else offsets[0]), entry(277, SHORT, 1, 3),
        entry(278, LONG, 1, rows_per_strip), entry(279, LONG, len(strips), cnts_off if many else counts[0]),
        entry(282, RATIONAL, 1, xres_off), entry(283, RATIONAL, 1, yres_off), entry(296, SHORT, 1, 2),
    ]
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", ifd_off))
        for y, n in strips:
            f.write(np.ascontiguousarray(data[y:y + n].astype("<u2")).tobytes())
        if f.tell() % 2:
            f.write(b"\x00")
        f.write(struct.pack("<HHH", 16, 16, 16))
        f.write(struct.pack("<II", dpi, 1) + struct.pack("<II", dpi, 1))
        f.write(struct.pack("<%dI" % len(strips), *offsets) + struct.pack("<%dI" % len(strips), *counts))
        f.write(struct.pack("<H", len(entries)) + b"".join(entries) + struct.pack("<I", 0))


def _exif_bytes(photo_path: Path) -> bytes | None:
    """Metadatos EXIF de la foto original, con la orientación ya aplicada (la imagen exportada sale derecha)."""
    try:
        with Image.open(photo_path) as img:
            exif = img.getexif()
            if not len(exif):
                return None
            exif[0x0112] = 1
            return exif.tobytes()
    except Exception:  # una foto sin EXIF legible no impide exportar
        log.warning("No se pudieron leer los metadatos de %s", photo_path)
        return None


def save_final(image: np.ndarray, path: Path, fmt: str, options: ExportOptions, exif: bytes | None = None) -> None:
    """Guarda la imagen final (float32 RGB 0..1) en JPG, TIFF 8/16 bits, PNG 8/16 bits o WebP."""
    if fmt not in FINAL_FORMATS:
        raise ValueError(f"Formato desconocido para la imagen final: {fmt}")
    image = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    dpi = (options.dpi, options.dpi)
    extra = {"exif": exif} if exif else {}
    if fmt == "tiff16":
        write_tiff16(path, np.rint(image * 65535.0).astype(np.uint16), options.dpi)
    elif fmt == "png16":
        write_png16(path, np.rint(image * 65535.0).astype(np.uint16))
    else:
        pil = Image.fromarray(np.rint(image * 255.0).astype(np.uint8), mode="RGB")
        if fmt == "jpg":
            pil.save(path, quality=int(options.quality), subsampling=0 if options.quality >= 90 else -1, optimize=True,
                     dpi=dpi, **extra)
        elif fmt == "webp":
            pil.save(path, quality=int(options.quality), method=4, **extra)
        elif fmt == "tiff8":
            pil.save(path, compression="tiff_lzw", dpi=dpi, **extra)
        else:
            pil.save(path, dpi=dpi, **extra)


def depth_image(depth: np.ndarray, near_bright: bool = True) -> np.ndarray:
    """Mapa de profundidad listo para guardar: 0..1 en gris. cerca = claro es lo habitual (como la vista)."""
    d = np.clip(depth.astype(np.float32), 0.0, 1.0)
    return 1.0 - d if near_bright else d


def export_depth(depth: np.ndarray, options: ExportOptions, progress: Progress = _no_progress,
                 depth_control: DepthControl | None = None) -> Path:
    """Guarda el mapa de profundidad (un solo canal) a su resolución completa.

    Con `options.depth_zones` y un `depth_control` se guarda el mapa con las zonas movidas (lo que leen la rampa,
    las bandas, el desenfoque y las áreas); si no, el mapa original, tal como salió del modelo."""
    options.validate()
    options.folder.mkdir(parents=True, exist_ok=True)
    path = free_path(options.folder, options.name, EXTENSIONS[options.format])
    progress(QCoreApplication.translate("Export", "Guardando el mapa de profundidad…"), 0.5)
    if options.depth_zones and depth_control is not None:
        depth = depth_control.apply(depth)
    if options.depth_palette:        # en color: cerca → primer color de la paleta (no aplica «cerca = claro»)
        image = colorize_float(depth, options.depth_palette)
    else:
        image = depth_image(depth, options.near_bright)
    save_image(image, path, options.format)
    log.info("Mapa de profundidad exportado: %s", path)
    progress(QCoreApplication.translate("Export", "Listo"), 1.0)
    return path


def _render_full(project: Project, depth: np.ndarray, photo_path: Path, painted_mask: np.ndarray | None,
                 progress: Progress, with_look: bool, crop: tuple | None = None,
                 masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Carga la foto y procesa la imagen a la resolución del mapa de profundidad (con o sin la pila de Look)."""
    progress(QCoreApplication.translate("Export", "Cargando la fotografía…"), 0.05)
    photo = load_photo(Path(photo_path), crop)               # con el reencuadre previo, solo esa zona (como el mapa)
    mask = None
    if painted_mask is not None and project.composite.mask == "painted":
        h, w = depth.shape
        img = Image.fromarray(painted_mask, mode="L")
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        mask = np.asarray(img, dtype=np.float32) / 255.0
    progress(QCoreApplication.translate("Export", "Procesando la imagen a resolución completa…"), 0.2)
    h, w = depth.shape
    aux = {k: resize_map(v, (w, h)) for k, v in (masks or {}).items()}      # mapas auxiliares al tamaño del mapa
    return render(project, depth, photo, painted_mask=mask, with_look=with_look, masks=aux)


def export_effect(project: Project, depth: np.ndarray, photo_path: Path, options: ExportOptions,
                  painted_mask: np.ndarray | None = None, progress: Progress = _no_progress,
                  crop: tuple | None = None, masks: dict[str, np.ndarray] | None = None) -> list[Path]:
    """Procesa el efecto a resolución completa y lo guarda (y los ajustes en JSON, si se pide).

    painted_mask: máscara pintada (uint8, 2D) en memoria; se reescala al tamaño del mapa.
    masks: mapas auxiliares (id -> float32 0..1) para las capas Pre / Post FX limitadas a un mapa; se reescalan.
    Devuelve las rutas escritas.
    """
    options.validate()
    options.folder.mkdir(parents=True, exist_ok=True)
    final = options.content == "final"
    image = _render_full(project, depth, photo_path, painted_mask, progress, with_look=final, crop=crop, masks=masks)
    exif = _exif_bytes(Path(photo_path)) if final and options.keep_metadata else None
    if final:                                                                # tamaño elegido, con Lanczos
        h, w = image.shape[:2]
        size = target_size(w, h, options.size_mode, options.size_value)
        if size != (w, h):
            progress(QCoreApplication.translate("Export", "Cambiando el tamaño…"), 0.7)
            image = resize_image(image, size)
    path = free_path(options.folder, options.name, EXTENSIONS[options.format])
    progress(QCoreApplication.translate("Export", "Guardando…"), 0.8)
    if final:
        save_final(image, path, options.format, options, exif)
    else:
        save_image(image, path, options.format)
    written = [path]
    if options.save_settings:
        settings_path = path.with_suffix(".deepshot.json")
        project.save(settings_path)
        written.append(settings_path)
    log.info("Efecto exportado (%d×%d, %s): %s", image.shape[1], image.shape[0], options.format, path)
    progress(QCoreApplication.translate("Export", "Listo"), 1.0)
    return written
