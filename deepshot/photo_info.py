"""Datos de una fotografía para el panel de información: archivo, cámara, exposición y color (sin interfaz).

Todo sale de los metadatos EXIF que lee Pillow; no hay dependencias nuevas. El GPS va aparte (`gps`) porque es un dato
sensible: la interfaz lo muestra oculto salvo que el usuario lo pida.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from PIL import Image

EXIF_IFD, GPS_IFD = 0x8769, 0x8825
ORIENTATIONS = {5, 6, 7, 8}                         # las que giran 90° (ancho y alto cambian)


@dataclass
class PhotoInfo:
    name: str = ""
    path: str = ""
    file_bytes: int = 0
    format: str = ""
    width: int = 0                                   # tal como la ve la app (con la rotación EXIF aplicada)
    height: int = 0
    megapixels: float = 0.0
    camera: str = ""
    lens: str = ""
    focal_mm: str = ""
    aperture: str = ""
    shutter: str = ""
    iso: str = ""
    taken: str = ""
    software: str = ""
    color_profile: str = ""
    orientation: int = 1
    gps: str = ""                                    # «lat, lon» o "" si no hay
    extra: dict[str, str] = field(default_factory=dict)


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _shutter(value) -> str:
    seconds = _num(value)
    if not seconds or seconds <= 0:
        return ""
    if seconds >= 1:
        return f"{seconds:g} s"
    return f"1/{round(1 / seconds)} s"


def _text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    return str(value).strip("\x00 ").strip() if value is not None else ""


def _gps(ifd) -> str:
    try:
        def deg(parts, ref):
            d, m, s = (float(Fraction(x)) if not isinstance(x, (int, float)) else float(x) for x in parts)
            value = d + m / 60 + s / 3600
            return -value if _text(ref).upper() in ("S", "W") else value
        lat, lon = deg(ifd[2], ifd[1]), deg(ifd[4], ifd[3])
        return f"{lat:.5f}, {lon:.5f}"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ""


def read(path: Path) -> PhotoInfo:
    """Lee los datos de la foto. Nunca falla: lo que no se puede leer queda vacío."""
    path = Path(path)
    info = PhotoInfo(name=path.name, path=str(path))
    try:
        info.file_bytes = path.stat().st_size
        with Image.open(path) as img:
            info.format = img.format or path.suffix.lstrip(".").upper()
            w, h = img.size
            exif = img.getexif()
            info.orientation = int(exif.get(0x0112, 1) or 1)
            if info.orientation in ORIENTATIONS:
                w, h = h, w
            info.width, info.height = w, h
            info.megapixels = round(w * h / 1e6, 1)
            ifd = exif.get_ifd(EXIF_IFD) if exif else {}
            make, model = _text(exif.get(0x010F)), _text(exif.get(0x0110))
            info.camera = model if model.startswith(make) else f"{make} {model}".strip()
            info.lens = _text(ifd.get(0xA434))
            focal = _num(ifd.get(0x920A))
            info.focal_mm = f"{focal:g} mm" if focal else ""
            f_number = _num(ifd.get(0x829D))
            info.aperture = f"f/{f_number:g}" if f_number else ""
            info.shutter = _shutter(ifd.get(0x829A))
            iso = ifd.get(0x8827)
            info.iso = f"ISO {iso}" if iso else ""
            when = _text(ifd.get(0x9003) or exif.get(0x0132))
            try:
                info.taken = dt.datetime.strptime(when, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
            except ValueError:
                info.taken = when
            info.software = _text(exif.get(0x0131))
            profile = img.info.get("icc_profile")
            if profile:
                try:
                    from PIL import ImageCms
                    import io
                    info.color_profile = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip()
                except Exception:                                  # perfil raro: basta con saber que lo hay
                    info.color_profile = "ICC"
            if exif:
                info.gps = _gps(exif.get_ifd(GPS_IFD))
    except (OSError, ValueError):
        pass
    return info


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"
