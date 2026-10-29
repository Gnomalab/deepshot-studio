"""Mensaje comprensible para el usuario a partir de un error al calcular la profundidad (sin hilos ni editor).

Lo usan la ventana principal (`depth_worker`) y el Mapa rápido, y se lo llevará tal cual la aplicación básica."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication


def friendly_error(exc: Exception) -> str:
    """Mensaje comprensible para el usuario; el detalle técnico queda en el registro."""
    from deepshot.models import ModelError

    if isinstance(exc, ModelError):
        return str(exc)  # ya está redactado para el usuario
    text = f"{type(exc).__name__}: {exc}"
    if "out of memory" in text.lower():
        return QCoreApplication.translate("DepthWorker", "No hay memoria de vídeo suficiente. Cierra otros programas que usen la GPU o baja la resolución de proceso.")
    if any(k in text for k in ("ConnectionError", "ConnectError", "Timeout", "LocalEntryNotFound", "HTTP")):
        return QCoreApplication.translate("DepthWorker", "No se pudo descargar el modelo. La primera vez hace falta conexión a internet; después funciona sin ella.")
    return QCoreApplication.translate("DepthWorker", "No se pudo calcular la profundidad. Consulta el panel Registro para ver el detalle.")
