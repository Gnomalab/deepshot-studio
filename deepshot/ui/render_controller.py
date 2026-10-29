"""Vista previa en vivo del resultado, calculada en segundo plano.

Cada cambio de ajustes pide un render:
  1. Vista previa rápida a resolución reducida (lado largo PREVIEW_SIZE).
  2. Si no hay más cambios en FULL_DELAY_MS, render detallado (hasta DETAIL_SIZE px:
     más que cualquier pantalla; la exportación de la fase 5 usará la resolución real).
Solo cuenta la última petición: si llegan varias seguidas (al arrastrar un control),
las intermedias se descartan. La interfaz nunca se bloquea.

Hay dos hilos: uno para las vistas previas y otro para el render completo. Así, si
vuelves a tocar un control mientras se calcula el render completo (2-3 s en fotos
grandes), la vista previa no tiene que esperar: el render completo se descarta.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

from deepshot.project import Project, render, to_uint8

log = logging.getLogger(__name__)

PREVIEW_SIZE = 1024     # lado largo de la vista previa, en píxeles
DETAIL_SIZE = 3072      # lado largo del render "completo" de pantalla (la exportación usa el tamaño real)
FULL_DELAY_MS = 700     # espera sin cambios antes del render completo


def to_qimage(image: np.ndarray) -> QImage:
    """float32 (alto, ancho, 3) 0..1 -> QImage RGB (copia independiente del array)."""
    data = np.ascontiguousarray(to_uint8(image))
    h, w = data.shape[:2]
    return QImage(data.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class _Level:
    """Datos de entrada a una resolución: profundidad, foto y caché de la nube de puntos."""

    def __init__(self, depth: np.ndarray, photo: Image.Image) -> None:
        self.depth = depth
        h, w = depth.shape
        self.photo = np.asarray(photo.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        self._points_key = None
        self._points = None
        self._mask_src = None         # máscara (uint8) de la que sale `_mask_scaled`
        self._mask_scaled = None

    def mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        """Máscara pintada reescalada al tamaño de este nivel (con caché por objeto de máscara)."""
        if mask is None:
            return None
        if mask is not self._mask_src:
            h, w = self.depth.shape
            img = Image.fromarray(mask, mode="L")
            if img.size != (w, h):
                img = img.resize((w, h), Image.BILINEAR)
            self._mask_scaled = np.asarray(img, dtype=np.float32) / 255.0
            self._mask_src = mask
        return self._mask_scaled

    def points(self, project: Project) -> np.ndarray:
        # Los puntos 3D se calculan con la profundidad ya movida por el control de profundidad.
        key = (project.scene.relief, project.scene.fov_deg, project.depth_control.key())
        if key != self._points_key:
            self._points = project.scene.points(project.depth_control.apply(self.depth))
            self._points_key = key
        return self._points


class _Worker:
    """Hilo con un único hueco de trabajo: una petición nueva sustituye a la pendiente."""

    def __init__(self, run) -> None:
        self._run = run
        self._pending = None
        self._stopped = False
        self._cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        """Termina el hilo en cuanto acabe el trabajo en curso."""
        with self._cond:
            self._stopped = True
            self._pending = None
            self._cond.notify()

    def submit(self, job) -> None:
        with self._cond:
            self._pending = job
            self._cond.notify()

    def clear(self) -> None:
        with self._cond:
            self._pending = None

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                job, self._pending = self._pending, None
            self._run(job)


class RenderController(QObject):
    """Recibe peticiones de render y emite `rendered(imagen, es_completo, ms)`."""

    rendered = Signal(QImage, bool, float)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._preview: _Level | None = None
        self._full: _Level | None = None
        self._full_depth: np.ndarray | None = None
        self._full_photo: Image.Image | None = None
        self._generation = 0
        self._lock = threading.Lock()
        self._last_project: Project | None = None
        self._last_mask: np.ndarray | None = None
        self._alive = True
        self._preview_worker = _Worker(self._run)
        self._full_worker = _Worker(self._run)

        self._full_timer = QTimer(self)
        self._full_timer.setSingleShot(True)
        self._full_timer.setInterval(FULL_DELAY_MS)
        self._full_timer.timeout.connect(self._request_full)

    # --- Fuente -------------------------------------------------------------------
    def set_source(self, depth: np.ndarray | None, photo: Image.Image | None) -> None:
        """Cambia la foto y su mapa. Prepara la versión reducida para la vista previa."""
        with self._lock:
            self._generation += 1      # invalida lo que esté en marcha
        self._preview_worker.clear()
        self._full_worker.clear()
        self._full_timer.stop()
        if depth is None or photo is None:
            self._preview = self._full = None
            return
        h, w = depth.shape
        scale = min(1.0, PREVIEW_SIZE / max(h, w))
        small = (max(1, round(w * scale)), max(1, round(h * scale)))
        small_depth = np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize(small, Image.BILINEAR))
        self._preview = _Level(np.clip(small_depth, 0, 1).astype(np.float32), photo)
        # Nivel detallado: limitado a DETAIL_SIZE (se prepara en el hilo, al primer uso).
        detail = min(1.0, DETAIL_SIZE / max(h, w))
        if detail < 1.0:
            size = (max(1, round(w * detail)), max(1, round(h * detail)))
            depth = np.clip(np.asarray(Image.fromarray(depth.astype(np.float32), mode="F")
                                       .resize(size, Image.BILINEAR)), 0, 1).astype(np.float32)
        self._full_depth, self._full_photo = depth, photo
        self._full = None   # se prepara al primer render completo (tarda algo más)

    def release_caches(self) -> None:
        """Suelta lo que se puede volver a calcular: el nivel detallado y las cachés de puntos y máscara.

        La foto y su mapa se conservan; el nivel detallado se prepara otra vez al siguiente render completo."""
        with self._lock:
            self._generation += 1                  # lo que esté en marcha se descarta
        self._preview_worker.clear()
        self._full_worker.clear()
        self._full_timer.stop()
        if self._preview is not None and self._full_photo is not None:
            self._preview = _Level(self._preview.depth, self._full_photo)     # nivel nuevo, sin cachés
        self._full = None

    def shutdown(self) -> None:
        """Detiene los hilos. Llamar antes de destruir el controlador (al cerrar la ventana)."""
        self._alive = False
        self._full_timer.stop()
        self._preview_worker.stop()
        self._full_worker.stop()

    def has_source(self) -> bool:
        return self._preview is not None

    def preview_depth(self) -> np.ndarray | None:
        """Mapa de la vista previa (sin el control de profundidad; ver `Editor.warped`)."""
        return self._preview.depth if self._preview else None

    # --- Peticiones -----------------------------------------------------------------
    def request(self, project: Project, mask: np.ndarray | None = None) -> None:
        """Pide una vista previa con estos ajustes (copia: se pueden seguir editando).

        mask: máscara pintada (uint8, 2D) ya copiada por quien llama; se reescala al nivel de cada render.
        """
        if self._preview is None:
            return
        self._last_project = Project.from_dict(project.to_dict())
        self._last_mask = mask
        self._submit(self._last_project, full=False)
        self._full_timer.start()

    def _request_full(self) -> None:
        if self._last_project is not None and self._preview is not None:
            self._submit(self._last_project, full=True)

    def _submit(self, project: Project, full: bool) -> None:
        with self._lock:
            if not full:
                self._generation += 1   # un cambio nuevo invalida el render completo en curso
            generation = self._generation
        (self._full_worker if full else self._preview_worker).submit((generation, project, full, self._last_mask))

    # --- Trabajo (en los hilos) -------------------------------------------------------
    def _run(self, job) -> None:
        generation, project, full, mask = job
        if generation != self._generation:
            return                      # ya hay un cambio más reciente
        try:
            level = self._preview
            if full:
                if self._full is None and self._full_depth is not None:
                    self._full = _Level(self._full_depth, self._full_photo)
                level = self._full
            if level is None:
                return
            t = time.perf_counter()
            painted = level.mask(mask) if project.composite.mask == "painted" else None
            image = render(project, level.depth, level.photo, points=level.points(project), painted_mask=painted)
            ms = (time.perf_counter() - t) * 1000
            if generation == self._generation and self._alive:   # descartar si ya no vale
                self._emit(self.rendered, to_qimage(image), full, ms)
        except Exception as exc:  # un ajuste inválido no debe tumbar el hilo
            log.exception("Fallo en el render")
            self._emit(self.failed, str(exc))

    def _emit(self, signal, *args) -> None:
        """Emite desde el hilo; si el objeto ya se destruyó (app cerrándose), no hace nada."""
        try:
            signal.emit(*args)
        except RuntimeError:
            self._alive = False
