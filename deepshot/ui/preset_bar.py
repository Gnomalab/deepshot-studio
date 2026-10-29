"""Selector de presets (de fábrica y propios) con botón para guardar y borrar los propios.

Se usa para las rampas de color, los estilos completos y el bokeh (`deepshot/presets.py`).
Emite `chosen(datos)` con el contenido del preset elegido; quien lo recibe lo aplica.

Dos ayudas para saber qué está pasando:
  - Tras elegir un preset, el desplegable sigue mostrando su nombre mientras los ajustes no se toquen
    (`sync()` lo comprueba); en cuanto se cambia algo vuelve al texto guía, y si se deshace hasta ese estado,
    el nombre vuelve a verse.
  - Al pasar el ratón por la lista (o moverse con las flechas) emite `previewed(datos)` para que se vea el
    resultado en la vista previa sin aplicarlo; si se cierra la lista sin elegir, `previewEnded()`.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QWidget,
)

from deepshot import presets
from deepshot.ui.icons import icon, set_icon


class PresetBar(QWidget):
    chosen = Signal(dict)
    previewed = Signal(dict)        # el ratón está sobre un preset de la lista: enseñarlo sin aplicarlo
    previewEnded = Signal()         # se cerró la lista sin elegir: volver a lo que había

    def __init__(self, kind: str, placeholder: str, get_current: Callable[[], dict],
                 parent: QWidget | None = None) -> None:
        """kind: "ramp" o "style". get_current: devuelve los datos a guardar (`presets.from_project`)."""
        super().__init__(parent)
        self.kind = kind
        self._get_current = get_current
        self._placeholder = placeholder
        self._active: tuple | None = None          # (origen, clave) del último preset aplicado
        self._active_snapshot: str | None = None   # ajustes justo después de aplicarlo (para saber si se tocaron)
        self._previewing = False
        self._committed = False                    # se eligió algo antes de cerrar la lista

        self.combo = QComboBox()
        # No se sobrescriben showPopup / hidePopup (una subclase en Python con esos métodos virtuales hace que Qt
        # falle al destruir el widget): el cierre de la lista se detecta mirando si su ventana sigue visible.
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._check_popup)
        self.combo.setToolTip(self.tr("Elige un preset de fábrica o uno propio: se aplica al instante "
                                      "y se puede deshacer"))
        self.combo.activated.connect(self._on_activated)
        self.combo.highlighted.connect(self._on_highlighted)
        self.menu_button = QPushButton()
        set_icon(self.menu_button, "save")
        self.menu_button.setIconSize(QSize(16, 16))
        self.menu_button.setFixedWidth(30)
        self.menu_button.setStyleSheet("padding: 0;")
        self.menu_button.setToolTip(self.tr("Guardar el ajuste actual como preset propio, importar o exportar "
                                            "presets de otras personas, o borrar uno"))
        self.menu_button.clicked.connect(self._show_menu)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.combo, 1)
        row.addWidget(self.menu_button)
        self.refresh()

    # --- Lista ----------------------------------------------------------------------
    def refresh(self) -> None:
        """Rellena el desplegable: placeholder, presets de fábrica y, aparte, los propios."""
        self.combo.clear()
        self.combo.addItem(self._placeholder, None)
        for key, name in presets.builtin_names(self.kind):
            self.combo.addItem(QCoreApplication.translate("Presets", name), ("builtin", key))
        user = presets.list_user(self.kind)
        if user:
            self.combo.insertSeparator(self.combo.count())
            for name in user:
                self.combo.addItem(name, ("user", name))
                self.combo.setItemData(self.combo.count() - 1, self.tr("Preset propio"), Qt.ToolTipRole)
        if self._active is not None and self._index_of(self._active) < 0:
            self._active = self._active_snapshot = None            # el preset activo ya no está en la lista
        self._show_active()

    def _index_of(self, data: tuple) -> int:
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == data:
                return i
        return -1

    def _load(self, data: tuple) -> dict:
        source, key = data
        return presets.builtin(self.kind, key) if source == "builtin" else presets.load_user(self.kind, key)

    def _snapshot(self) -> str | None:
        """Los ajustes actuales de este tipo como texto comparable (None si aún no hay proyecto)."""
        try:
            return json.dumps(self._get_current(), sort_keys=True)
        except (AttributeError, TypeError, ValueError):
            return None

    def _show_active(self) -> None:
        """El desplegable muestra el preset activo si los ajustes siguen siendo los que dejó; si no, el texto guía."""
        if self._poll.isActive():                                  # la lista está abierta mirando presets: no tocarla
            return
        index = 0
        if self._active is not None and self._active_snapshot is not None and self._snapshot() == self._active_snapshot:
            index = max(self._index_of(self._active), 0)
        if self.combo.currentIndex() != index:
            self.combo.blockSignals(True)        # cambiar el texto mostrado por código no es pasar el ratón por un preset
            self.combo.setCurrentIndex(index)
            self.combo.blockSignals(False)

    def sync(self) -> None:
        """Que el nombre del preset activo se vea solo mientras los ajustes sigan siendo los suyos."""
        self._show_active()

    def _remember(self, data: tuple | None) -> None:
        self._active = data
        self._active_snapshot = self._snapshot() if data is not None else None
        self._show_active()

    # --- Elegir y previsualizar ---------------------------------------------------------------------
    def _on_activated(self, index: int) -> None:
        data = self.combo.itemData(index)
        self._committed = True
        self._previewing = False
        if not data:
            self._show_active()
            return
        try:
            preset = self._load(data)
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, self.tr("Preset"), str(exc))
            self.refresh()
            return
        self.chosen.emit(preset)             # quien lo recibe lo aplica aquí mismo
        self._remember(data)                 # y el desplegable se queda mostrando su nombre

    def _on_highlighted(self, index: int) -> None:
        """El ratón (o las flechas) está sobre un elemento de la lista: se previsualiza."""
        data = self.combo.itemData(index)
        if not data:
            if self._previewing:
                self._previewing = False
                self.previewEnded.emit()
            return
        try:
            preset = self._load(data)
        except (KeyError, ValueError):
            return
        self._previewing = True
        if not self._poll.isActive():
            self._poll.start()               # vigila cuándo se cierra la lista
        self.previewed.emit(preset)

    def _check_popup(self) -> None:
        if self.combo.view().isVisible():
            return
        self._poll.stop()
        self._after_popup()

    def _after_popup(self) -> None:
        """Lista cerrada: si se miraba un preset y no se eligió ninguno, se vuelve a lo que había."""
        if self._previewing and not self._committed:
            self._previewing = False
            self.previewEnded.emit()
        self._committed = False
        self._show_active()

    # --- Guardar y borrar -----------------------------------------------------------------
    def _show_menu(self) -> None:
        menu = QMenu(self)
        save = menu.addAction(icon("save"), self.tr("Guardar ajuste actual como preset…"))
        save.triggered.connect(lambda _checked=False: self.save_current())   # Qt pasa `checked`: no es el nombre
        user = presets.list_user(self.kind)
        menu.addSeparator()
        load = menu.addAction(icon("import"), self.tr("Importar preset desde un archivo…"))
        load.triggered.connect(lambda _checked=False: self.import_file())
        export = menu.addMenu(icon("export"), self.tr("Exportar preset propio a un archivo"))
        export.setEnabled(bool(user))
        for name in user:
            action = export.addAction(name)
            action.triggered.connect(lambda _checked=False, n=name: self.export_file(n))
        delete = menu.addMenu(icon("trash"), self.tr("Borrar preset propio"))
        delete.setEnabled(bool(user))
        for name in user:
            action = delete.addAction(name)
            action.triggered.connect(lambda _checked=False, n=name: self.delete(n))
        menu.exec(self.menu_button.mapToGlobal(self.menu_button.rect().bottomLeft()))

    def save_current(self, name: str | None = None) -> bool:
        """Pide un nombre (si no se da) y guarda. Devuelve True si se guardó."""
        if name is None:
            name, ok = QInputDialog.getText(self, self.tr("Guardar preset"), self.tr("Nombre del preset:"))
            if not ok:
                return False
        name = name.strip()
        if not presets.safe_name(name):
            QMessageBox.warning(self, self.tr("Guardar preset"), self.tr("Escribe un nombre para el preset."))
            return False
        if presets.exists_user(self.kind, name):
            answer = QMessageBox.question(self, self.tr("Guardar preset"),
                                          self.tr("Ya existe un preset llamado «{0}». ¿Sustituirlo?").format(name))
            if answer != QMessageBox.Yes:
                return False
        try:
            presets.save_user(self.kind, name, self._get_current())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, self.tr("Guardar preset"), self.tr("No se pudo guardar: {0}").format(exc))
            return False
        self.refresh()
        self._remember(("user", name))       # lo que se acaba de guardar es justo lo que hay: se marca
        return True

    # --- Importar y exportar (compartir con otras personas) -------------------------------------
    def _file_filter(self) -> str:
        return self.tr("Presets de Deepshot Studio (*.deepshotstyle *.json)")

    def import_file(self, file: str | None = None) -> str | None:
        """Añade a la lista un preset de un archivo: lo comprueba, pregunta qué hacer si el nombre ya existe y lo deja
        disponible al momento. Devuelve el nombre con que quedó (None si se canceló o no era válido)."""
        settings = QSettings()
        if file is None:
            file, _ = QFileDialog.getOpenFileName(self, self.tr("Importar preset"),
                                                  str(settings.value("last_preset_dir", "")), self._file_filter())
            if not file:
                return None
        try:
            name, _data = presets.read_importable(Path(file), self.kind)
        except ValueError as exc:
            QMessageBox.warning(self, self.tr("Importar preset"), str(exc))
            return None
        final = None
        if presets.exists_user(self.kind, name):
            box = QMessageBox(QMessageBox.Question, self.tr("Importar preset"),
                              self.tr("Ya tienes un preset llamado «{0}».").format(name), QMessageBox.NoButton, self)
            replace = box.addButton(self.tr("Sustituirlo"), QMessageBox.DestructiveRole)
            both = box.addButton(self.tr("Conservar los dos"), QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is replace:
                final = name
            elif box.clickedButton() is not both:
                return None
        try:
            saved = presets.import_user(self.kind, Path(file), final)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, self.tr("Importar preset"), self.tr("No se pudo importar: {0}").format(exc))
            return None
        settings.setValue("last_preset_dir", str(Path(file).parent))
        self.refresh()
        return saved

    def export_file(self, name: str, file: str | None = None) -> bool:
        """Guarda un preset propio en un archivo para pasárselo a otra persona."""
        settings = QSettings()
        if file is None:
            start = Path(str(settings.value("last_preset_dir", ""))) / f"{presets.safe_name(name)}.deepshotstyle"
            file, _ = QFileDialog.getSaveFileName(self, self.tr("Exportar preset"), str(start), self._file_filter())
            if not file:
                return False
        try:
            presets.export_user(self.kind, name, Path(file))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, self.tr("Exportar preset"), self.tr("No se pudo exportar: {0}").format(exc))
            return False
        settings.setValue("last_preset_dir", str(Path(file).parent))
        return True

    def delete(self, name: str) -> None:
        answer = QMessageBox.question(self, self.tr("Borrar preset"),
                                      self.tr("¿Borrar el preset «{0}»?").format(name))
        if answer == QMessageBox.Yes:
            presets.delete_user(self.kind, name)
            self.refresh()
