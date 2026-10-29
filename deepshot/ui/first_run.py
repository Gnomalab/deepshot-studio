"""Aviso del primer arranque: todavía no hay ningún modelo de profundidad instalado.

Explica qué hace falta, cuánto ocupa, de dónde se descarga y que se puede seguir trabajando sin él (importando un mapa de
profundidad hecho fuera). No descarga nada por su cuenta: lleva al gestor de modelos, donde se confirma y se ve el progreso.
"""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from deepshot.models import DEFAULT_MODEL, ModelManager, can_compute


def needs_first_run(manager: ModelManager) -> bool:
    """True si no hay ningún modelo con el que la app sepa calcular la profundidad ya instalado."""
    return not any(manager.is_installed(e.id) for e in manager.all(include_hidden=True) if can_compute(e))


class FirstRunDialog(QDialog):
    """Pregunta qué hacer al no haber modelos. `action` queda en «manager», «folder» o «later»."""

    def __init__(self, manager: ModelManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Falta un modelo de profundidad"))
        self.setMinimumWidth(520)
        self.action = "later"
        entry = manager.get(DEFAULT_MODEL)

        title = QLabel(self.tr("Para calcular mapas de profundidad hace falta un modelo"))
        title.setObjectName("AppTitle")
        title.setWordWrap(True)
        body = QLabel(self.tr(
            "La aplicación no incluye los modelos para que pese poco. El recomendado es «{0}» ({1:.1f} GB, licencia {2}): "
            "se descarga una sola vez desde Hugging Face y queda guardado en tu equipo; después todo funciona sin "
            "conexión. Es lo único que la aplicación descarga de internet.\n\n"
            "Si no tienes conexión, puedes instalar un modelo desde una carpeta. Y mientras tanto puedes seguir "
            "trabajando: importa un mapa de profundidad hecho con otra herramienta (Archivo › Cargar mapa de profundidad).")
            .format(entry.display_label, entry.approx_gb or 0.0, entry.license))
        body.setWordWrap(True)
        body.setProperty("dim", True)

        self.manager_button = QPushButton(self.tr("Abrir el gestor de modelos"))
        self.manager_button.setObjectName("Primary")
        self.manager_button.setToolTip(self.tr("Elige y descarga un modelo; antes de descargar se pide confirmación y se "
                                               "ve el progreso"))
        self.manager_button.clicked.connect(lambda: self._choose("manager"))
        self.folder_button = QPushButton(self.tr("Instalar desde carpeta…"))
        self.folder_button.setToolTip(self.tr("Si ya tienes un modelo en una carpeta (o no hay conexión), instálalo desde ahí"))
        self.folder_button.clicked.connect(lambda: self._choose("folder"))
        self.later_button = QPushButton(self.tr("Ahora no"))
        self.later_button.setToolTip(self.tr("Sigue sin modelo: podrás importar mapas de profundidad hechos fuera"))
        self.later_button.clicked.connect(lambda: self._choose("later"))
        self.dont_ask = QCheckBox(self.tr("No volver a preguntar"))
        self.dont_ask.setToolTip(self.tr("No se vuelve a mostrar este aviso al arrancar. El gestor de modelos sigue en "
                                         "Profundidad › Gestor de modelos (Ctrl+M)."))

        buttons = QHBoxLayout()
        buttons.addWidget(self.dont_ask)
        buttons.addStretch()
        for b in (self.later_button, self.folder_button, self.manager_button):
            buttons.addWidget(b)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addLayout(buttons)

    def _choose(self, action: str) -> None:
        self.action = action
        self.accept()
