"""Pantalla de inicio: proyecto nuevo, abrir uno o retomar uno reciente.

Sale al arrancar la aplicación (se puede desactivar con la casilla; se recupera en Preferencias o en
Archivo › Pantalla de inicio…). Esc o cerrar la ventana = empezar en blanco.

Resultado (`action`): "new" · "open" (el selector de archivos de siempre) · "recent" (con `chosen_path`) ·
"quick" (el modo sencillo: solo el mapa de profundidad).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deepshot import APP_NAME
from deepshot.ui.icons import set_icon


class StartDialog(QDialog):
    def __init__(self, recents: list[str], parent: QWidget | None = None, show_checkbox: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.setModal(True)
        self.setMinimumWidth(520)
        self.action = "new"                 # lo que se hace si se cierra sin elegir: empezar en blanco
        self.chosen_path: str | None = None
        self.removed: list[str] = []        # recientes que el usuario quitó de la lista aquí

        title = QLabel(self.tr("¿Qué quieres hacer?"))
        title.setObjectName("AppTitle")
        title.setToolTip(self.tr("Empieza un proyecto nuevo o retoma uno de los últimos que abriste"))

        self.list = QListWidget()
        self.list.setToolTip(self.tr("Proyectos recientes, el más nuevo arriba. Doble clic para abrir."))
        self.list.setMinimumHeight(180)
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.itemDoubleClicked.connect(lambda _item: self._open_selected())
        self.empty_label = QLabel(self.tr("Aún no hay proyectos recientes."))
        self.empty_label.setProperty("dim", True)
        self.empty_label.setToolTip(self.tr("Cuando guardes o abras un proyecto, aparecerá aquí"))

        self.new_button = QPushButton(self.tr("Proyecto nuevo"))
        self.new_button.setObjectName("Primary")
        set_icon(self.new_button, "plus")
        self.new_button.setToolTip(self.tr("Empieza en blanco: importa fotografías y calcula su profundidad"))
        self.new_button.clicked.connect(self._new)
        self.quick_button = QPushButton(self.tr("Mapa de profundidad rápido…"))
        self.quick_button.setObjectName("Primary")
        set_icon(self.quick_button, "layers")                # el mismo icono que el «0 Rápido» de la cabecera
        self.quick_button.setToolTip(self.tr("Modo sencillo: elige fotos y obtén solo el mapa de profundidad "
                                             "(o de normales), sin abrir el editor"))
        self.quick_button.clicked.connect(self._quick)
        self.open_button = QPushButton(self.tr("Abrir proyecto…"))
        set_icon(self.open_button, "folder")
        self.open_button.setToolTip(self.tr("Busca un archivo .deepshot en tu equipo"))
        self.open_button.clicked.connect(self._open_file)
        self.recent_button = QPushButton(self.tr("Abrir el seleccionado"))
        set_icon(self.recent_button, "import")
        self.recent_button.setToolTip(self.tr("Abre el proyecto reciente seleccionado en la lista"))
        self.recent_button.clicked.connect(self._open_selected)
        self.remove_button = QPushButton(self.tr("Quitar de la lista"))
        set_icon(self.remove_button, "minus")
        self.remove_button.setToolTip(self.tr("Quita el proyecto de la lista de recientes (no borra el archivo)"))
        self.remove_button.clicked.connect(self._remove_selected)

        self.dont_show = QCheckBox(self.tr("No volver a mostrar al iniciar"))
        self.dont_show.setToolTip(self.tr("La próxima vez la aplicación arrancará en blanco. Puedes volver a "
                                          "activarlo en Preferencias."))
        self.dont_show.setVisible(show_checkbox)

        row = QHBoxLayout()
        for button in (self.recent_button, self.remove_button):
            row.addWidget(button)
        row.addStretch()
        bottom = QHBoxLayout()
        bottom.addWidget(self.dont_show)
        bottom.addStretch()
        bottom.addWidget(self.open_button)
        # Las dos formas de empezar, arriba y con el mismo peso: un proyecto completo o el modo rápido por lotes.
        for button in (self.new_button, self.quick_button):
            button.setMinimumHeight(44)
            button.setIconSize(QSize(24, 24))
        for button in (self.open_button, self.recent_button, self.remove_button):
            button.setIconSize(QSize(16, 16))
        # Debajo de cada una, una línea que explica para qué sirve (más amable para quien llega por primera vez).
        new_hint = QLabel(self.tr("Importa fotos, calcula su profundidad y da forma al efecto"))
        quick_hint = QLabel(self.tr("Crea mapas de profundidad o de normales de muchas fotos de golpe, sin abrir el editor"))
        for label in (new_hint, quick_hint):
            label.setWordWrap(True)
            label.setProperty("dim", True)
        start_row = QHBoxLayout()
        start_row.setSpacing(12)
        for button, label in ((self.new_button, new_hint), (self.quick_button, quick_hint)):
            column = QVBoxLayout()
            column.setSpacing(4)
            column.addWidget(button)
            column.addWidget(label)
            start_row.addLayout(column, 1)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addLayout(start_row)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.empty_label)
        layout.addLayout(row)
        layout.addLayout(bottom)
        self.set_recents(recents)

    # --- Lista -----------------------------------------------------------------------------------
    def set_recents(self, recents: list[str]) -> None:
        self.list.clear()
        for file in recents:
            path = Path(file)
            exists = path.is_file()
            when = ""
            if exists:
                try:
                    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
                except OSError:
                    when = ""
            note = when if exists else self.tr("(no se encuentra el archivo)")
            item = QListWidgetItem(f"{path.stem}\n{path.parent}  ·  {note}")
            item.setData(Qt.UserRole, file)
            item.setToolTip(file)
            if not exists:
                item.setForeground(QBrush(QColor("#7a7a7a")))
            self.list.addItem(item)
        self.empty_label.setVisible(self.list.count() == 0)
        self.list.setVisible(self.list.count() > 0)
        if self.list.count():
            self.list.setCurrentRow(0)
        self._update_buttons()

    def selected_path(self) -> str | None:
        items = self.list.selectedItems()
        return str(items[0].data(Qt.UserRole)) if items else None

    def _update_buttons(self) -> None:
        path = self.selected_path()
        self.recent_button.setEnabled(path is not None and Path(path).is_file())
        self.remove_button.setEnabled(path is not None)

    # --- Acciones --------------------------------------------------------------------------------
    def _new(self) -> None:
        self.action = "new"
        self.accept()

    def _quick(self) -> None:
        self.action = "quick"
        self.accept()

    def _open_file(self) -> None:
        self.action = "open"
        self.accept()

    def _open_selected(self) -> None:
        path = self.selected_path()
        if path is None or not Path(path).is_file():
            return
        self.action, self.chosen_path = "recent", path
        self.accept()

    def _remove_selected(self) -> None:
        path = self.selected_path()
        if path is None:
            return
        self.removed.append(path)
        self.set_recents([str(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())
                          if str(self.list.item(i).data(Qt.UserRole)) != path])
