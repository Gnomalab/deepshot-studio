"""Aplicación básica: solo el Mapa de profundidad rápido, sin editor, proyectos ni rampas.

Comparte con la completa el estimador, el gestor de modelos, el tema, el idioma, el visor de revisión rápida y la
ventana del Mapa rápido (`SimpleDepthDialog`). No importa la ventana principal ni el editor: el anfitrión
(`QuickHost`) se construye aquí con sus propios selectores.

Arrancar: `python -m deepshot.quick_app` (o el comando `deepshot-quick`).
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtWidgets import QComboBox

from deepshot.bootstrap import create_app
from deepshot.depth import DEFAULT_MODEL, DEFAULT_QUALITY, QUALITY_PRESETS, RESOLUTION_CHOICES, DepthEstimator
from deepshot.models import ModelError, ModelManager, can_compute
from deepshot.ui.model_manager import ModelManagerDialog
from deepshot.ui.simple_depth_dialog import QuickHost, SimpleDepthDialog


def fill_model_combo(combo: QComboBox, manager: ModelManager, active: str) -> str:
    """Rellena el selector con los modelos que salen en él y marca el activo. Devuelve el activo (el de por defecto si el
    guardado ya no existe)."""
    try:
        manager.get(active)
    except ModelError:
        active = DEFAULT_MODEL
    combo.clear()
    for entry in manager.all():
        if manager.is_enabled(entry.id) or entry.id == active:
            if can_compute(entry):
                combo.addItem(entry.display_label, entry.id)
    combo.setCurrentIndex(max(0, combo.findData(active)))
    return active


def build_host() -> QuickHost:
    """Estimador y selectores propios (modelo, calidad y resolución), con la elección guardada en los ajustes."""
    settings = QSettings()
    manager = ModelManager()
    estimator = DepthEstimator(manager=manager)

    model_combo = QComboBox()
    fill_model_combo(model_combo, manager, str(settings.value("depth/active_model", DEFAULT_MODEL)))

    def open_manager() -> None:
        """El gestor de modelos de la aplicación completa, sin entrar en ella: lo que se elige se guarda en los ajustes."""
        dialog = ModelManagerDialog(manager, estimator, model_combo.currentData() or DEFAULT_MODEL, lambda: False)

        def activate(model_id: str) -> None:
            settings.setValue("depth/active_model", model_id)
            fill_model_combo(model_combo, manager, model_id)

        dialog.active_changed.connect(activate)
        dialog.models_changed.connect(lambda: fill_model_combo(model_combo, manager, model_combo.currentData() or DEFAULT_MODEL))
        dialog.exec()
        fill_model_combo(model_combo, manager, model_combo.currentData() or DEFAULT_MODEL)

    quality_combo = QComboBox()
    for preset in QUALITY_PRESETS.values():
        quality_combo.addItem(QCoreApplication.translate("Depth", preset.label), preset.id)
        quality_combo.setItemData(quality_combo.count() - 1, QCoreApplication.translate("Depth", preset.hint), Qt.ToolTipRole)
    quality_combo.setCurrentIndex(max(0, quality_combo.findData(settings.value("depth/quality", DEFAULT_QUALITY))))

    resolution_combo = QComboBox()
    resolution_combo.addItem(QCoreApplication.translate("QuickApp", "Resolución: la de la calidad"), 0)
    for px in RESOLUTION_CHOICES:
        resolution_combo.addItem(QCoreApplication.translate("QuickApp", "Resolución: {0} px").format(px), px)
    resolution_combo.setCurrentIndex(max(0, resolution_combo.findData(settings.value("depth/resolution", 0, type=int))))
    resolution_combo.setToolTip(QCoreApplication.translate(
        "QuickApp", "Lado mayor con el que el modelo calcula el mapa. Más alto = más detalle, más tiempo y más memoria "
                    "de vídeo; más bajo = más rápido. «La de la calidad» usa la del preset."))

    return QuickHost(estimator, model_combo, quality_combo, resolution_combo, open_model_manager=open_manager)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    create_app()
    dialog = SimpleDepthDialog(build_host())
    dialog.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
