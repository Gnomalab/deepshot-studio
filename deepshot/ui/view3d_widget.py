"""Visor 3D del relieve del mapa de profundidad (órbita con el ratón) y edición de áreas en el espacio.

El dibujo del relieve está en `deepshot/view3d.py` y la geometría de las áreas en `deepshot/area3d.py`.

Ratón: arrastrar el fondo = girar · Mayús+arrastrar o botón central = mover · rueda = acercar o alejar ·
doble clic o Inicio = restablecer. Teclas 1 / 3 / 7 = vista de frente, lateral y desde arriba (como en Blender).
Áreas: clic en un contorno la selecciona; en la seleccionada, las flechas mueven por eje, los cuadrados escalan
cada semieje (Mayús: todos), los anillos giran por eje (Ctrl: de 15° en 15°) y el punto central o el contorno
la mueven en el plano de la pantalla. Esc cancela el gesto en curso.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from deepshot.ui import theme
from deepshot import area3d, view3d
from deepshot.areas import Area, Scene

AXIS_COLORS = {"X": QColor("#d1493f"), "Y": QColor("#6ba84f"), "Z": QColor("#4772b3")}
HIT_POINT = 9.0          # píxeles de tolerancia al tocar un cuadrado o el punto central
HIT_LINE = 6.5           # ... y una línea (flecha, anillo o contorno)
DEAD_ZONE = 3.0          # píxeles que hay que mover antes de que un gesto cambie el área


def _qimage_to_rgb(image: QImage, size: tuple[int, int]) -> np.ndarray:
    """QImage → (alto, ancho, 3) uint8 escalada a `size` = (ancho, alto)."""
    scaled = image.scaled(size[0], size[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation) \
        .convertToFormat(QImage.Format_RGB888)
    stride = scaled.bytesPerLine()
    buf = np.frombuffer(scaled.constBits(), dtype=np.uint8, count=stride * scaled.height())
    return buf.reshape(scaled.height(), stride)[:, : scaled.width() * 3].reshape(scaled.height(), scaled.width(), 3).copy()


def _dist_to_polyline(px: float, py: float, sx: np.ndarray, sy: np.ndarray) -> float:
    """Distancia mínima de un punto de pantalla a una poligonal proyectada (ignora los tramos no visibles)."""
    ok = np.isfinite(sx) & np.isfinite(sy)
    if ok.sum() < 2:
        return float("inf")
    ax, ay, bx, by = sx[:-1], sy[:-1], sx[1:], sy[1:]
    valid = ok[:-1] & ok[1:]
    if not valid.any():
        return float("inf")
    ax, ay, bx, by = ax[valid], ay[valid], bx[valid], by[valid]
    dx, dy = bx - ax, by - ay
    length2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
    return float(np.min(np.hypot(px - (ax + t * dx), py - (ay + t * dy))))


class View3D(QWidget):
    """Relieve del mapa de profundidad como superficie 3D que se puede orbitar, con las áreas encima."""

    cameraChanged = Signal()
    # Gesto sobre un área: {"phase": press|move|release|cancel, "kind", "index", "center", "scale", "rotation"}
    areaGesture = Signal(dict)
    # Gesto sobre los tiradores de una banda: {"phase": press|move|release|cancel, "kind": move|pan|tilt|width, "index",
    # "position", "width", "pan", "tilt"} (los valores nuevos ya calculados a partir de la banda al pulsar)
    bandGesture = Signal(dict)
    # Cuentagotas del foco: {"phase": press|move|release|cancel, "x", "y"} con x, y en píxeles de la foto
    pickGesture = Signal(dict)
    # Dibujar / Colocar sobre el relieve: {"kind": "draw"|"place", "phase": press|move|release|cancel, "x", "y", "mods",
    # "index"} con x, y en píxeles de la foto (el punto de la superficie bajo el ratón). Es el mismo gesto que el del
    # lienzo 2D (`EditSession.gesture`): lo único que cambia es de dónde sale el punto.
    toolGesture = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)
        self.camera = view3d.OrbitCamera()
        self._depth: np.ndarray | None = None
        self._scene = Scene()
        self._texture: QImage | None = None
        self._grid: np.ndarray | None = None          # puntos (filas, columnas, 3)
        self._colors: np.ndarray | None = None        # color de cada punto (uint8)
        self._extra: tuple[np.ndarray, np.ndarray] | None = None   # puntos que tapan los acantilados
        self._grid_key: tuple | None = None
        self._color_key: tuple | None = None
        self._frame: QImage | None = None
        self._frame_key: tuple | None = None
        self._last = QPointF()
        self._drag = ""                               # "orbit" | "pan" | "area" | ""
        self.shading = 0.4                            # fuerza del sombreado del relieve (0 = ninguno)
        self._areas: list[Area] = []
        self._selected = -1
        self._hover = ""                              # tirador bajo el ratón (para resaltarlo)
        self._axis_hover = ""                         # bola del gizmo de orientación bajo el ratón
        self._gz: dict | None = None                  # gesto de área en curso
        self._modal: dict | None = None               # G / S / R con el teclado en curso (ver `_start_modal`)
        # Bandas: (esquinas del plano, color, `area3d.BandFrame`, índice en el proyecto)
        self._bands: list[tuple[np.ndarray, tuple[float, float, float], area3d.BandFrame, int]] = []
        self._selected_band = -1                      # la banda elegida en la lista: la que lleva tiradores
        self._bg: dict | None = None                  # gesto de banda en curso
        self._focus: np.ndarray | None = None         # nitidez 0..255 de la capa verde «Ver zona en foco»
        self._focus_serial = 0
        self._tool = ""                               # "pick_focus" | "draw" | "place" con esa herramienta activa, "" sin ella
        self._tool_last: tuple[float, float] | None = None   # último punto de la superficie del gesto de Dibujar / Colocar
        # Mientras se mueve la cámara se dibuja a media resolución y sin los puntos de relleno (fluido);
        # 160 ms después del último movimiento se redibuja con toda la calidad.
        self._fast = False
        # Calidad 3D: nivel de view3d.QUALITY_POINTS o "auto" (mide el coste del primer dibujo y elige).
        self._quality = view3d.AUTO
        self._auto_points: int | None = None          # elegido por el modo automático para la foto actual
        self._refine = QTimer(self)
        self._refine.setSingleShot(True)
        self._refine.setInterval(160)
        self._refine.timeout.connect(self._refine_now)

    # --- Datos -----------------------------------------------------------------
    def set_data(self, depth: np.ndarray | None, scene: Scene | None, texture: QImage | None,
                 keep_camera: bool = True) -> None:
        """depth: mapa 0..1 (0 = cerca) · scene: relieve y perspectiva · texture: imagen que colorea la superficie."""
        first = self._depth is None and depth is not None
        changed_photo = depth is not None and self._depth is not None and depth.shape != self._depth.shape
        self._depth, self._scene, self._texture = depth, scene or Scene(), texture
        if first or changed_photo or not keep_camera:
            self.reset_view()
        self.update()

    def set_areas(self, areas: list[Area], selected: int = -1) -> None:
        """Áreas a dibujar (las mismas del proyecto) y cuál está seleccionada."""
        self._areas = list(areas)
        self._selected = selected if 0 <= selected < len(self._areas) else -1
        self.update()

    def set_bands(self, planes: list[tuple], selected: int = -1) -> None:
        """Planos de las bandas a dibujar: [(esquinas (4, 3), color rgb 0..1, BandFrame, índice)] y cuál está elegida
        (la que lleva tiradores)."""
        self._bands = list(planes)
        self._selected_band = selected
        self.update()

    def set_focus_overlay(self, sharpness: np.ndarray | None) -> None:
        """Capa verde de la zona en foco sobre el relieve (nitidez uint8, 255 = en foco). None la quita."""
        if sharpness is None and self._focus is None:
            return
        self._focus = sharpness
        self._focus_serial += 1
        self.update()

    def reset_view(self) -> None:
        self.camera.reset(depth_center=self._scene.relief * 0.5)
        self.cameraChanged.emit()
        self.update()

    def set_tool(self, tool: str) -> None:
        """Herramienta activa en el visor 3D: «pick_focus» (cuentagotas del foco), «draw» (dibujar un área sobre el
        relieve), «place» (colocar la seleccionada) o «» (órbita y áreas)."""
        self._tool = tool if tool in self.TOOLS else ""
        self.setCursor(Qt.CrossCursor if self._tool else Qt.ArrowCursor)

    TOOLS = ("pick_focus", "draw", "place")

    def _emit_tool(self, phase: str, pos: QPointF | None, mods=Qt.NoModifier) -> bool:
        """Gesto de Dibujar / Colocar con el punto de la superficie bajo `pos`. Fuera de la superficie se repite el
        último punto bueno (soltar fuera no deshace el gesto). Devuelve False si no hay ningún punto."""
        point = self.pick_photo_point(pos) if pos is not None else None
        if point is not None:
            self._tool_last = point
        elif self._tool_last is None or phase == "press":
            return False
        x, y = self._tool_last
        self.toolGesture.emit({"phase": phase, "kind": self._tool, "index": -1, "x": x, "y": y,
                               "mods": int(mods.value) if hasattr(mods, "value") else int(mods)})
        return True

    def pick_photo_point(self, pos: QPointF, radius: float = 10.0) -> tuple[float, float] | None:
        """Punto de la foto (x, y en píxeles) de la superficie que se ve bajo `pos`, o None si no hay superficie.

        Se proyectan los puntos de la rejilla a la pantalla y, de los que caen cerca del puntero, vale el más cercano
        a la cámara (el que no tapa nada): así se elige lo que se ve, no lo que queda detrás."""
        if self._depth is None or not self._prepare():
            return None
        grid = self._grid
        gh, gw = grid.shape[:2]
        sx, sy, zc = self.camera.project(grid, self.width(), self.height())
        d2 = (sx - pos.x()) ** 2 + (sy - pos.y()) ** 2
        d2 = np.where(np.isnan(d2), np.inf, d2)
        near = d2 <= radius ** 2
        if not near.any():
            return None
        index = int(np.argmin(np.where(near, zc, np.inf)))
        row, col = divmod(index, gw)
        h, w = self._depth.shape
        return (col + 0.5) / gw * w, (row + 0.5) / gh * h

    def has_data(self) -> bool:
        return self._depth is not None

    # --- Calidad 3D ------------------------------------------------------------------------
    def quality(self) -> str:
        return self._quality

    def set_quality(self, quality: str) -> None:
        """«auto» o uno de los niveles de `view3d.QUALITY_POINTS`. Cambiarla rehace la rejilla."""
        if quality != view3d.AUTO and quality not in view3d.QUALITY_POINTS:
            quality = view3d.AUTO
        if quality == self._quality:
            return
        self._quality = quality
        self._auto_points = None
        self.update()

    def release_caches(self) -> None:
        """Suelta la rejilla, los colores y el último dibujo; se rehacen en el siguiente dibujo."""
        self._grid = self._colors = self._extra = None
        self._grid_key = self._color_key = None
        self._frame = self._frame_key = None

    def max_points(self) -> int:
        """Puntos de la rejilla con la calidad actual (en automático, el nivel inicial hasta medir)."""
        if self._quality == view3d.AUTO:
            return self._auto_points or view3d.QUALITY_POINTS[view3d.AUTO_FIRST]
        return view3d.QUALITY_POINTS[self._quality]

    # --- Mallas y color (con caché) ---------------------------------------------------
    def _prepare(self) -> bool:
        depth = self._depth
        if depth is None:
            return False
        h, w = depth.shape
        gw, gh = view3d.grid_size(w, h, self.max_points())
        key = (id(depth), depth.shape, round(self._scene.relief, 4), round(self._scene.fov_deg, 3), gw, gh)
        if key != self._grid_key:
            small = np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize((gw, gh), Image.BILINEAR))
            self._grid = Scene(self._scene.relief, self._scene.fov_deg).points(np.clip(small, 0.0, 1.0))
            self._grid_key = key
            self._color_key = None                     # el sombreado depende de la forma
        ckey = (self._grid_key, self._texture.cacheKey() if self._texture is not None else None, self.shading,
                self._focus_serial)
        if ckey != self._color_key:
            if self._texture is not None and not self._texture.isNull():
                colors = _qimage_to_rgb(self._texture, (gw, gh))
            else:
                gray = np.clip(self._grid[..., 2] / max(self._scene.relief, 1e-6), 0, 1)
                colors = np.repeat((255 * (1.0 - gray))[..., None], 3, axis=-1).astype(np.uint8)
            self._colors = view3d.shade(colors, self._grid[..., 2], w / h, self.shading)
            if self._focus is not None:           # tinte verde de lo que queda nítido, igual que en la vista 2D
                alpha = np.asarray(Image.fromarray(self._focus).resize((gw, gh), Image.BILINEAR),
                                   dtype=np.float32)[..., None] / 255.0 * 0.55
                green = np.array([40, 220, 90], dtype=np.float32)
                self._colors = (self._colors * (1.0 - alpha) + green * alpha).astype(np.uint8)
            self._extra = view3d.cliff_points(self._grid, self._colors, max_total=int(self.max_points() * 1.3))
            self._color_key = ckey
        return True

    def _render_frame(self) -> QImage | None:
        if not self._prepare():
            return None
        key = (self._color_key, self.camera.key(), self.width(), self.height(), self._fast)
        if key != self._frame_key:
            scale = 2 if self._fast else 1
            grid, colors = self._grid, self._colors
            if self._fast:      # al mover la cámara se dibuja solo una parte de los puntos: fluido con cualquier calidad
                step = max(1, int(np.sqrt(grid.shape[0] * grid.shape[1] / view3d.FAST_POINTS)))
                grid, colors = grid[::step, ::step], colors[::step, ::step]
            started = time.perf_counter()
            bg = theme.color("viewer")
            arr = view3d.render_cloud(grid, colors, self.camera, max(self.width() // scale, 1),
                                      max(self.height() // scale, 1), extra=None if self._fast else self._extra,
                                      background=(bg.red(), bg.green(), bg.blue(), 255))
            elapsed = time.perf_counter() - started
            self._frame = QImage(arr.data, arr.shape[1], arr.shape[0], arr.shape[1] * 4, QImage.Format_RGBA8888).copy()
            self._frame_key = key
            if not self._fast:
                self._auto_adjust(elapsed)
        return self._frame

    def _auto_adjust(self, seconds: float) -> None:
        """Modo automático: con el coste medido del primer dibujo completo elige los puntos de la rejilla."""
        if self._quality != view3d.AUTO or self._auto_points is not None or self._depth is None:
            return
        total = self._grid.shape[0] * self._grid.shape[1] + (len(self._extra[0]) if self._extra else 0)
        h, w = self._depth.shape
        chosen = view3d.auto_points(seconds / max(total, 1), w * h)
        self._auto_points = chosen
        if chosen != view3d.QUALITY_POINTS[view3d.AUTO_FIRST]:
            self.update()                       # la rejilla cambia de tamaño: se rehace en el siguiente dibujo

    # --- Proyección ---------------------------------------------------------------------
    def _screen(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sx, sy, _ = self.camera.project(points, self.width(), self.height())
        return sx, sy

    def _ray(self, pos: QPointF) -> area3d.Ray:
        return area3d.ray_through(self.camera, pos.x(), pos.y(), self.width(), self.height())

    # --- Dibujo ---------------------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), theme.color("viewer"))
        frame = self._render_frame()
        if frame is None:
            p.setPen(theme.color("text_dim"))
            p.drawText(self.rect(), Qt.AlignCenter, self.tr("Calcula o carga el mapa de profundidad para ver el relieve."))
            return
        p.drawImage(self.rect(), frame)               # a media resolución se estira al tamaño del visor
        p.setRenderHint(QPainter.Antialiasing, True)
        self._draw_plane(p)
        self._draw_bands(p)
        self._draw_areas(p)
        self._draw_gizmo(p)
        self._draw_axes(p)
        p.setPen(theme.color("text_dim"))
        font = QFont(self.font())
        font.setPointSizeF(max(self.font().pointSizeF() - 1, 7))
        p.setFont(font)
        if self._modal is not None:
            p.setPen(theme.color("text"))
            p.drawText(QRectF(10, self.height() - 26, self.width() - 20, 18), Qt.AlignLeft | Qt.AlignVCenter,
                       self._modal_hint())
            return
        p.drawText(QRectF(10, self.height() - 26, self.width() - 20, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   self.tr("Arrastrar: girar · Mayús+arrastrar: mover · Rueda: zoom · Doble clic: restablecer · "
                           "Clic en un área: seleccionar · G / S / R: mover, escalar, girar · Clic en un eje: alinear la vista"))

    def _draw_plane(self, p: QPainter) -> None:
        """Contorno del plano de la foto (z = 0): sirve de referencia para ver cuánto sale el relieve."""
        if self._depth is None:
            return
        h, w = self._depth.shape
        a = w / h
        corners = np.array([[-a, 1, 0], [a, 1, 0], [a, -1, 0], [-a, -1, 0]], dtype=np.float64)
        sx, sy = self._screen(corners)
        if not (np.isfinite(sx).all() and np.isfinite(sy).all()):
            return
        p.setPen(QPen(QColor(255, 255, 255, 70), 1.0, Qt.DashLine))
        for i in range(4):
            j = (i + 1) % 4
            p.drawLine(QPointF(sx[i], sy[i]), QPointF(sx[j], sy[j]))

    def _draw_bands(self, p: QPainter) -> None:
        """Plano de cada banda (inclinado si tiene pan o tilt): relleno suave y borde en el color de la banda."""
        for corners, (r, g, b), _frame, _index in self._bands:
            sx, sy = self._screen(corners)
            if not (np.isfinite(sx).all() and np.isfinite(sy).all()):
                continue
            color = QColor.fromRgbF(r, g, b)
            fill = QColor(color)
            fill.setAlpha(34)
            edge = QColor(color)
            edge.setAlpha(190)
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in zip(sx, sy)])
            p.setPen(QPen(QColor(0, 0, 0, 120), 3.0))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(poly)
            p.setPen(QPen(edge, 1.6, Qt.DashLine))
            p.setBrush(fill)
            p.drawPolygon(poly)
        p.setBrush(Qt.NoBrush)
        self._draw_band_handles(p)

    def _selected_band_frame(self) -> area3d.BandFrame | None:
        for _corners, _color, frame, index in self._bands:
            if index == self._selected_band:
                return frame
        return None

    def _draw_band_handles(self, p: QPainter) -> None:
        """Tiradores de la banda elegida: el grueso de la franja (dos planos tenues), la flecha en profundidad, los
        círculos de pan (derecha) y tilt (arriba) y el cuadrado del grueso (izquierda)."""
        frame = self._selected_band_frame()
        if frame is None:
            return
        p.setPen(QPen(QColor(255, 255, 255, 90), 1.0, Qt.DotLine))
        for sign in (-1, 1):
            sx, sy = self._screen(area3d.band_slab_corners(frame, sign))
            if np.isfinite(sx).all() and np.isfinite(sy).all():
                p.drawPolygon(self._polyline(sx, sy))
        points = area3d.band_handles(frame)
        length = area3d.gizmo_length(self.camera)
        csx, csy = self._screen(points["move"][None, :])
        tsx, tsy = self._screen((points["move"] + area3d.AXES["z"] * length)[None, :])
        if np.isfinite([csx[0], csy[0], tsx[0], tsy[0]]).all():          # flecha en profundidad
            hot = self._hover == "band_move"
            color = QColor(AXIS_COLORS["Z"])
            origin, tip = QPointF(float(csx[0]), float(csy[0])), QPointF(float(tsx[0]), float(tsy[0]))
            p.setPen(QPen(color, 3.2 if hot else 2.2))
            p.drawLine(origin, tip)
            d = tip - origin
            n = (d.x() ** 2 + d.y() ** 2) ** 0.5
            if n > 1e-6:
                ux, uy = d.x() / n, d.y() / n
                p.setBrush(color)
                p.drawPolygon(QPolygonF([tip + QPointF(ux * 9, uy * 9), tip + QPointF(-uy * 4.5, ux * 4.5),
                                         tip + QPointF(uy * 4.5, -ux * 4.5)]))
                p.setBrush(Qt.NoBrush)
        for name, axis in (("pan", "Y"), ("tilt", "X"), ("width", "Z")):
            sx, sy = self._screen(points[name][None, :])
            if not (np.isfinite(sx[0]) and np.isfinite(sy[0])):
                continue
            half = 6.0 if self._hover == f"band_{name}" else 4.5
            p.setPen(QPen(QColor("#ffffff"), 1.2))
            p.setBrush(QColor(AXIS_COLORS[axis]).lighter(135))
            if name == "width":
                p.drawRect(QRectF(sx[0] - half, sy[0] - half, 2 * half, 2 * half))
            else:
                p.drawEllipse(QPointF(float(sx[0]), float(sy[0])), half + 1, half + 1)
        p.setBrush(Qt.NoBrush)

    def band_hit(self, pos: QPointF) -> str:
        """Tirador de la banda elegida bajo el puntero («move», «pan», «tilt», «width») o «»."""
        frame = self._selected_band_frame()
        if frame is None:
            return ""
        px, py = pos.x(), pos.y()
        points = area3d.band_handles(frame)
        for name in ("pan", "tilt", "width"):                              # los círculos y el cuadrado, antes que la flecha
            sx, sy = self._screen(points[name][None, :])
            if np.hypot(px - sx[0], py - sy[0]) <= HIT_POINT:
                return name
        length = area3d.gizmo_length(self.camera)
        sx, sy = self._screen(np.stack([points["move"], points["move"] + area3d.AXES["z"] * length]))
        if _dist_to_polyline(px, py, sx, sy) <= HIT_LINE:
            return "move"
        return ""

    def _start_band_gesture(self, kind: str, pos: QPointF) -> None:
        self._bg = {"kind": kind, "index": self._selected_band, "frame0": self._selected_band_frame(),
                    "ray0": self._ray(pos), "moved": False, "press": pos}
        self._drag = "band"
        self.bandGesture.emit({"phase": "press", "kind": kind, "index": self._selected_band})

    def _band_update(self, pos: QPointF) -> None:
        bg = self._bg
        dx, dy = pos.x() - bg["press"].x(), pos.y() - bg["press"].y()
        bg["moved"] = bg["moved"] or dx * dx + dy * dy > DEAD_ZONE ** 2
        if not bg["moved"]:
            return
        new = area3d.BAND_GESTURES[bg["kind"]](bg["frame0"], bg["ray0"], self._ray(pos))
        self.bandGesture.emit({"phase": "move", "kind": bg["kind"], "index": bg["index"], "position": new.position,
                               "width": new.width, "pan": new.pan, "tilt": new.tilt})

    def _end_band_gesture(self, cancel: bool = False) -> None:
        bg, self._bg = self._bg, None
        self._drag = ""
        if bg is not None:
            self.bandGesture.emit({"phase": "cancel" if cancel else "release", "kind": bg["kind"], "index": bg["index"]})
        self.update()

    @staticmethod
    def _polyline(sx: np.ndarray, sy: np.ndarray) -> QPolygonF:
        return QPolygonF([QPointF(float(x), float(y)) for x, y in zip(sx, sy) if np.isfinite(x) and np.isfinite(y)])

    def _draw_areas(self, p: QPainter) -> None:
        """Contorno de cada área (elipses o aristas de la caja) con su color; la seleccionada, más gruesa."""
        for i, area in enumerate(self._areas):
            if not area.enabled:                  # desactivada = no está: tampoco se dibuja su contorno
                continue
            r, g, b = area.display_color(i)
            color = QColor.fromRgbF(r, g, b)
            selected = i == self._selected
            pen = QPen(color, 2.4 if selected else 1.4)
            p.setPen(pen)
            for line in area3d.wireframe(area):
                sx, sy = self._screen(line)
                p.drawPolyline(self._polyline(sx, sy))

    def _gizmo_frame(self, area: Area) -> tuple[np.ndarray, float]:
        """Centro del gizmo del área seleccionada y largo de sus flechas (tamaño fijo en pantalla)."""
        return np.asarray(area.center, dtype=np.float64), area3d.gizmo_length(self.camera)

    def _draw_gizmo(self, p: QPainter) -> None:
        if not (0 <= self._selected < len(self._areas)) or not self._areas[self._selected].enabled:
            return
        area = self._areas[self._selected]
        c, length = self._gizmo_frame(area)
        csx, csy = self._screen(c[None, :])
        if not (np.isfinite(csx[0]) and np.isfinite(csy[0])):
            return
        origin = QPointF(float(csx[0]), float(csy[0]))
        for name in ("x", "y", "z"):                  # anillos de giro (uno por eje del mundo)
            color = QColor(AXIS_COLORS[name.upper()])
            hot = self._hover == f"rotate_{name}"
            color.setAlpha(255 if hot else 150)
            p.setPen(QPen(color, 3.0 if hot else 1.6))
            sx, sy = self._screen(area3d.ring_points(c, name, length * 0.8))
            p.drawPolyline(self._polyline(sx, sy))
        for name in ("x", "y", "z"):                  # flechas de movimiento
            color = QColor(AXIS_COLORS[name.upper()])
            hot = self._hover == f"move_{name}"
            sx, sy = self._screen((c + area3d.AXES[name] * length)[None, :])
            if not (np.isfinite(sx[0]) and np.isfinite(sy[0])):
                continue
            tip = QPointF(float(sx[0]), float(sy[0]))
            p.setPen(QPen(color, 3.2 if hot else 2.2))
            p.drawLine(origin, tip)
            d = tip - origin
            n = (d.x() ** 2 + d.y() ** 2) ** 0.5
            if n > 1e-6:                              # punta de flecha
                ux, uy = d.x() / n, d.y() / n
                head = QPolygonF([tip + QPointF(ux * 9, uy * 9), tip + QPointF(-uy * 4.5, ux * 4.5),
                                  tip + QPointF(uy * 4.5, -ux * 4.5)])
                p.setBrush(color)
                p.drawPolygon(head)
                p.setBrush(Qt.NoBrush)
        for name, pt in area3d.scale_handle_points(area).items():      # cuadrados de escala
            sx, sy = self._screen(pt[None, :])
            if not (np.isfinite(sx[0]) and np.isfinite(sy[0])):
                continue
            half = 6.0 if self._hover == f"scale_{name}" else 4.5
            p.setPen(QPen(QColor("#ffffff"), 1.2))
            p.setBrush(QColor(AXIS_COLORS[name.upper()]).lighter(135))
            p.drawRect(QRectF(sx[0] - half, sy[0] - half, 2 * half, 2 * half))
        p.setBrush(QColor("#ffffff") if self._hover == "move_free" else QColor(240, 240, 240, 200))
        p.setPen(QPen(QColor("#202020"), 1.2))
        p.drawEllipse(origin, 5.5, 5.5)               # punto central: arrastre libre
        p.setBrush(Qt.NoBrush)

    # --- Gizmo de orientación (arriba a la derecha, como el de Blender) ------------------------
    GIZMO_RADIUS = 34.0           # distancia del centro a la punta de cada eje, en píxeles
    GIZMO_BALL = 9.0              # radio de las bolas de los ejes

    # Vista que se pone al pulsar cada bola: (yaw, pitch); None = se conserva el valor actual.
    # La cámara mira hacia el lado de la bola: +X desde la derecha, +Y desde arriba, -Z de frente (vista 1).
    AXIS_VIEWS = {"+X": (90.0, 0.0), "-X": (-90.0, 0.0), "+Y": (None, view3d.MAX_PITCH),
                  "-Y": (None, view3d.MIN_PITCH), "+Z": (180.0, 0.0), "-Z": (0.0, 0.0)}

    def _gizmo_center(self) -> QPointF:
        return QPointF(self.width() - self.GIZMO_RADIUS - 28, self.GIZMO_RADIUS + 28)

    def _gizmo_balls(self) -> list[tuple[str, QPointF, float]]:
        """Las seis bolas del gizmo: (nombre «+X»…, posición en pantalla, cuánto apuntan a la cámara)."""
        right, up, forward = self.camera.basis()
        c = self._gizmo_center()
        out = []
        for axis, v in (("X", np.array([1.0, 0, 0])), ("Y", np.array([0, 1.0, 0])), ("Z", np.array([0, 0, 1.0]))):
            for sign, name in ((1, "+" + axis), (-1, "-" + axis)):
                d = v * sign
                out.append((name, QPointF(c.x() + float(d @ right) * self.GIZMO_RADIUS,
                                          c.y() - float(d @ up) * self.GIZMO_RADIUS), float(d @ -forward)))
        return out

    def gizmo_hit(self, pos: QPointF) -> str:
        """Bola del gizmo bajo el puntero («+X», «-Z»…) o «». Si hay dos solapadas, la que mira a la cámara."""
        best, best_toward = "", -2.0
        for name, pt, toward in self._gizmo_balls():
            if (pt.x() - pos.x()) ** 2 + (pt.y() - pos.y()) ** 2 <= (self.GIZMO_BALL + 2) ** 2 and toward > best_toward:
                best, best_toward = name, toward
        return best

    def align_to_axis(self, name: str) -> None:
        """Pone la cámara mirando desde el lado de ese eje («+X», «-Y»…); el zoom y el punto de mira se conservan."""
        yaw, pitch = self.AXIS_VIEWS[name]
        self.camera.set_view(self.camera.yaw if yaw is None else yaw, pitch)
        self._moved()

    def _draw_axes(self, p: QPainter) -> None:
        """Gizmo de orientación: ejes X (rojo), Y (verde) y Z (azul) con una bola en cada punta; pulsar una bola
        alinea la vista con ese eje. Las bolas negativas son más pequeñas y sin letra."""
        c = self._gizmo_center()
        hovered = self._axis_hover
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 38 if hovered else 16))
        p.drawEllipse(c, self.GIZMO_RADIUS + 16, self.GIZMO_RADIUS + 16)           # disco de fondo
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(max(self.font().pointSizeF() - 1, 7))
        p.setFont(font)
        for name, pt, _toward in sorted(self._gizmo_balls(), key=lambda b: b[2]):  # las que miran a la cámara, encima
            color = AXIS_COLORS[name[1]]
            positive = name[0] == "+"
            if positive:
                p.setPen(QPen(color, 2.0))
                p.drawLine(c, pt)
            radius = self.GIZMO_BALL if positive else self.GIZMO_BALL * 0.62
            fill = QColor(color) if positive else QColor(color.red(), color.green(), color.blue(), 90)
            if name == hovered:
                fill = fill.lighter(140)
                p.setPen(QPen(QColor("#ffffff"), 1.4))
            else:
                p.setPen(QPen(color.darker(130), 1.0) if positive else Qt.NoPen)
            p.setBrush(fill)
            p.drawEllipse(pt, radius, radius)
            if positive:
                p.setPen(QColor("#ffffff"))
                p.drawText(QRectF(pt.x() - 8, pt.y() - 8, 16, 16), Qt.AlignCenter, name[1])
        p.setBrush(Qt.NoBrush)

    # --- Qué hay bajo el ratón --------------------------------------------------------
    def hit_test(self, pos: QPointF) -> tuple[str, int]:
        """(tirador, índice de área) bajo un punto de la pantalla, o ("", -1).

        Tiradores del área seleccionada, por orden de prioridad: cuadrados de escala, punto central, flechas,
        anillos y su contorno. Después, el contorno de cualquier otra área (la selecciona)."""
        px, py = pos.x(), pos.y()
        if 0 <= self._selected < len(self._areas) and self._areas[self._selected].enabled:
            area = self._areas[self._selected]
            c, length = self._gizmo_frame(area)
            # Cuadrados de escala y punto central: gana el más cercano al puntero, pero si un cuadrado cae justo
            # sobre el centro (el eje apunta a la cámara) se queda el centro; para escalar ese eje, se gira la vista.
            csx, csy = self._screen(c[None, :])
            d_center = float(np.hypot(px - csx[0], py - csy[0]))
            best_cube, best_d = "", HIT_POINT
            for name, pt in area3d.scale_handle_points(area).items():
                sx, sy = self._screen(pt[None, :])
                d = float(np.hypot(px - sx[0], py - sy[0]))
                if d <= best_d:
                    best_cube, best_d = name, d
            if d_center <= HIT_POINT + 1 and (not best_cube or d_center <= best_d + 3.0):
                return "move_free", self._selected
            if best_cube:
                return f"scale_{best_cube}", self._selected
            for name in ("x", "y", "z"):
                sx, sy = self._screen(np.stack([c, c + area3d.AXES[name] * length]))
                if _dist_to_polyline(px, py, sx, sy) <= HIT_LINE:
                    return f"move_{name}", self._selected
            for name in ("x", "y", "z"):
                sx, sy = self._screen(area3d.ring_points(c, name, length * 0.8))
                if _dist_to_polyline(px, py, sx, sy) <= HIT_LINE:
                    return f"rotate_{name}", self._selected
            for line in area3d.wireframe(area):
                sx, sy = self._screen(line)
                if _dist_to_polyline(px, py, sx, sy) <= HIT_LINE:
                    return "move_free", self._selected
        best, best_i = HIT_LINE, -1
        for i, area in enumerate(self._areas):
            if i == self._selected or not area.enabled:
                continue
            for line in area3d.wireframe(area):
                sx, sy = self._screen(line)
                d = _dist_to_polyline(px, py, sx, sy)
                if d < best:
                    best, best_i = d, i
        return ("select", best_i) if best_i >= 0 else ("", -1)

    # --- Gestos de área -------------------------------------------------------------------
    def _start_area_gesture(self, kind: str, index: int, pos: QPointF) -> None:
        self._gz = {"kind": kind, "index": index, "area0": copy.deepcopy(self._areas[index]),
                    "ray0": self._ray(pos), "press": pos, "moved": False}
        self._drag = "area"
        self.areaGesture.emit({"phase": "press", "kind": kind, "index": index})

    def _area_values(self, pos: QPointF, mods) -> Area | None:
        gz = self._gz
        area0, kind = gz["area0"], gz["kind"]
        ray1 = self._ray(pos)
        if kind == "move_free":
            return area3d.move_in_view_plane(area0, self.camera, gz["ray0"], ray1)
        what, _, axis = kind.partition("_")
        if what == "move":
            return area3d.move_along_axis(area0, axis, gz["ray0"], ray1)
        if what == "scale":
            return area3d.scale_along_axis(area0, axis, ray1, uniform=bool(mods & Qt.ShiftModifier))
        if what == "rotate":
            return area3d.rotate_about_axis(area0, axis, gz["ray0"], ray1, snap=bool(mods & Qt.ControlModifier))
        return None

    # --- G / S / R con el teclado (como en 2D y en Blender) ----------------------------------------
    MODAL_KEYS = {Qt.Key_G: "G", Qt.Key_S: "S", Qt.Key_R: "R"}

    def _modal_hint(self) -> str:
        m = self._modal
        names = {"G": self.tr("G · Mover"), "S": self.tr("S · Escalar"), "R": self.tr("R · Girar")}
        axis = f" · {m['axis'].upper()}" if m["axis"] else ""
        return self.tr("{0}{1}  —  X / Y / Z: fijar el eje · clic o Intro: aceptar · Esc: cancelar").format(
            names[m["mode"]], axis)

    def _start_modal(self, mode: str) -> None:
        """Empieza G / S / R sobre el área seleccionada: el ratón (sin pulsar) la mueve, escala o gira."""
        if self._drag or not (0 <= self._selected < len(self._areas)):
            return
        pos = self.mapFromGlobal(QCursor.pos())
        pos = QPointF(pos)
        area = copy.deepcopy(self._areas[self._selected])
        sx, sy = self._screen(np.asarray(area.center, dtype=np.float64)[None, :])
        self._modal = {"mode": mode, "axis": "", "index": self._selected, "area0": area, "ray0": self._ray(pos),
                       "pos0": pos, "center": (float(sx[0]), float(sy[0])), "moved": False}
        self._drag = "modal"
        self.setCursor(Qt.SizeAllCursor)
        self.areaGesture.emit({"phase": "press", "kind": "modal", "index": self._selected})
        self.update()

    def _modal_values(self, pos: QPointF, mods) -> Area | None:
        m = self._modal
        area0, axis = m["area0"], m["axis"]
        if m["mode"] == "G":
            ray1 = self._ray(pos)
            return (area3d.move_along_axis(area0, axis, m["ray0"], ray1) if axis
                    else area3d.move_in_view_plane(area0, self.camera, m["ray0"], ray1))
        if m["mode"] == "S":
            cx, cy = m["center"]
            d0 = max(float(np.hypot(m["pos0"].x() - cx, m["pos0"].y() - cy)), 8.0)
            d1 = float(np.hypot(pos.x() - cx, pos.y() - cy))
            return area3d.scale_by_factor(area0, d1 / d0, axis)
        return area3d.rotate_about_axis(area0, axis or "z", m["ray0"], self._ray(pos), snap=bool(mods & Qt.ControlModifier))

    def _modal_update(self, pos: QPointF, mods) -> None:
        new = self._modal_values(pos, mods)
        if new is None:
            return
        self._modal["moved"] = True
        self.areaGesture.emit({"phase": "move", "kind": "modal", "index": self._modal["index"],
                               "center": tuple(new.center), "scale": tuple(new.scale), "rotation": tuple(new.rotation)})

    def _end_modal(self, cancel: bool = False) -> None:
        m, self._modal = self._modal, None
        self._drag = ""
        self.unsetCursor()
        if m is not None:
            self.areaGesture.emit({"phase": "cancel" if cancel else "release", "kind": "modal", "index": m["index"]})
        self.update()

    def _end_area_gesture(self, cancel: bool = False) -> None:
        gz, self._gz = self._gz, None
        self._drag = ""
        if gz is not None:
            self.areaGesture.emit({"phase": "cancel" if cancel else "release", "kind": gz["kind"],
                                   "index": gz["index"]})
        self.update()

    # --- Ratón y teclado -----------------------------------------------------------------
    def _moved(self) -> None:
        """La cámara cambió: dibujar rápido ahora y con toda la calidad cuando se deje de mover."""
        self._fast = True
        self._refine.start()
        self.cameraChanged.emit()
        self.update()

    def _refine_now(self) -> None:
        self._fast = False
        self.update()

    def _emit_pick(self, phase: str, pos: QPointF | None = None) -> None:
        point = self.pick_photo_point(pos) if pos is not None else None
        g = {"phase": phase}
        if point is not None:
            g.update(x=point[0], y=point[1])
        if phase in ("release", "cancel") or point is not None:    # fuera de la superficie no hay valor nuevo
            self.pickGesture.emit(g)

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self._last = event.position()
        if self._modal is not None:                   # en G / S / R el clic izquierdo acepta y el derecho cancela
            self._end_modal(cancel=event.button() != Qt.LeftButton)
            return
        if event.button() == Qt.LeftButton and self._depth is not None:
            axis = self.gizmo_hit(event.position())
            if axis:                                  # pulsar una bola del gizmo alinea la vista; no empieza ningún arrastre
                self.align_to_axis(axis)
                return
        if self._tool == "pick_focus" and event.button() == Qt.LeftButton:
            self._drag = "pick"
            self._emit_pick("press", event.position())
            return
        if self._tool in ("draw", "place") and event.button() == Qt.LeftButton:
            self._tool_last = None
            if self._emit_tool("press", event.position(), event.modifiers()):
                self._drag = "tool"
            return
        if event.button() == Qt.LeftButton and not event.modifiers() & Qt.ShiftModifier:
            handle = self.band_hit(event.position())
            if handle:                                # un tirador de la banda elegida
                self._start_band_gesture(handle, event.position())
                return
        pan = event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton
                                                    and event.modifiers() & Qt.ShiftModifier
                                                    and not (self._areas and self.hit_test(event.position())[0]))
        if event.button() == Qt.LeftButton and not pan and self._areas:
            kind, index = self.hit_test(event.position())
            if kind:
                self._start_area_gesture(kind, index, event.position())
                return
        if pan:
            self._drag = "pan"
        elif event.button() == Qt.LeftButton:
            self._drag = "orbit"
        else:
            self._drag = ""
        if self._drag:
            self.setCursor(Qt.ClosedHandCursor if self._drag == "pan" else Qt.SizeAllCursor)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._drag == "modal" and self._modal is not None:
            self._modal_update(pos, event.modifiers())
            return
        if self._drag == "pick":
            self._emit_pick("move", pos)
            return
        if self._drag == "tool":
            self._emit_tool("move", pos, event.modifiers())
            return
        if self._drag == "band" and self._bg is not None:
            self._band_update(pos)
            return
        if self._drag == "area" and self._gz is not None:
            gz = self._gz
            dx, dy = pos.x() - gz["press"].x(), pos.y() - gz["press"].y()
            gz["moved"] = gz["moved"] or dx * dx + dy * dy > DEAD_ZONE ** 2     # un clic solo selecciona
            if gz["moved"] and gz["kind"] != "select":
                new = self._area_values(pos, event.modifiers())
                if new is not None:
                    self.areaGesture.emit({"phase": "move", "kind": gz["kind"], "index": gz["index"],
                                           "center": tuple(new.center), "scale": tuple(new.scale),
                                           "rotation": tuple(new.rotation)})
            return
        if not self._drag:
            axis = self.gizmo_hit(pos) if self._depth is not None else ""
            hover, _ = self.hit_test(pos) if self._areas and not axis else ("", -1)
            if not hover and not axis and self._bands and (handle := self.band_hit(pos)):
                hover = f"band_{handle}"
            if axis != self._axis_hover or hover != self._hover:
                self._axis_hover, self._hover = axis, hover
                self.setCursor(Qt.PointingHandCursor if hover or axis
                               else Qt.CrossCursor if self._tool else Qt.ArrowCursor)
                self.update()
            return
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if self._drag == "orbit":
            self.camera.orbit(-dx * 0.4, dy * 0.4)
        else:
            self.camera.pan(dx, dy, self.height())
        self._moved()

    def mouseReleaseEvent(self, event) -> None:
        if self._modal is not None:
            return
        if self._drag == "tool":
            self._emit_tool("release", event.position(), event.modifiers())
        if self._drag == "pick":
            self._emit_pick("release", event.position())
        if self._drag == "area":
            self._end_area_gesture()
        if self._drag == "band":
            self._end_band_gesture()
        self._drag = ""
        self.setCursor(Qt.CrossCursor) if self._tool else self.unsetCursor()

    def mouseDoubleClickEvent(self, event) -> None:
        if self._tool:
            return
        if event.button() == Qt.LeftButton and not (self._areas and self.hit_test(event.position())[0]):
            self.reset_view()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.camera.zoom(1.15 ** steps)
            self._moved()
        event.accept()

    def event(self, event) -> bool:
        # Durante G / S / R las teclas X / Y / Z son del eje: que el atajo global de «Z» (zona en foco) no se las lleve.
        if event.type() == QEvent.ShortcutOverride and self._modal is not None:
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if self._modal is not None:
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self._end_modal()
            elif key == Qt.Key_Escape:
                self._end_modal(cancel=True)
            elif key in (Qt.Key_X, Qt.Key_Y, Qt.Key_Z):
                axis = {Qt.Key_X: "x", Qt.Key_Y: "y", Qt.Key_Z: "z"}[key]
                self._modal["axis"] = "" if self._modal["axis"] == axis else axis      # la misma tecla suelta el eje
                self._modal_update(QPointF(self.mapFromGlobal(QCursor.pos())), event.modifiers())
                self.update()
            return
        if event.key() in self.MODAL_KEYS and not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier) \
                and not self._tool:
            self._start_modal(self.MODAL_KEYS[event.key()])
            return
        if event.key() == Qt.Key_Escape and self._drag == "tool":
            self._drag = ""
            self._emit_tool("cancel", None)
            return
        if event.key() == Qt.Key_Escape and self._drag == "pick":
            self._drag = ""
            self._emit_pick("cancel")
            return
        if event.key() == Qt.Key_Escape and self._drag == "area":
            self._end_area_gesture(cancel=True)
            return
        views = {Qt.Key_1: (0.0, 0.0), Qt.Key_2: (90.0, 0.0), Qt.Key_7: (0.0, 89.0)}
        if event.key() in views:
            self.camera.set_view(*views[event.key()])
        elif event.key() == Qt.Key_Home:
            self.reset_view()
            return
        else:
            super().keyPressEvent(event)
            return
        self._moved()

    def leaveEvent(self, _event) -> None:
        if (self._hover or self._axis_hover) and not self._drag:
            self._hover = self._axis_hover = ""
            self.update()
