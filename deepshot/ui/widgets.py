"""Widgets reutilizables con el estilo del panel de ajustes (inspirado en Blender)."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QListWidget,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deepshot.ui.icons import icon, set_icon


class HoverHelp(QObject):
    """Ayuda contextual: al pasar el ratón por un control, su explicación (la del tooltip) sale en el
    pie de la ventana. Sirve para todos los controles, también los que se crean después de abrir la
    ventana. Se instala como filtro de eventos de la aplicación."""

    def __init__(self, window, parent: QObject | None = None) -> None:
        super().__init__(parent or window)
        self._window = window
        self._shown = ""

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind == QEvent.Enter and isinstance(obj, QWidget) and obj.window() is self._window:
            tip = obj.statusTip() or obj.toolTip()
            if tip:
                self._shown = " ".join(tip.replace("\n", " ").split())
                self._window.statusBar().showMessage(self._shown)
        elif kind == QEvent.Leave and self._shown and isinstance(obj, QWidget):
            if self._window.statusBar().currentMessage() == self._shown:
                self._window.statusBar().clearMessage()
            self._shown = ""
        return False


class _SectionHeader(QPushButton):
    """Cabecera de una sección: Ctrl+clic pide «solo esta» (abre esta y cierra las demás) y el clic derecho, su menú."""

    ctrlClicked = Signal()
    menuRequested = Signal(QPoint)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier and self.rect().contains(event.position().toPoint()):
            self.setDown(False)
            self.ctrlClicked.emit()                 # no se pliega ni se despliega: lo decide quien escucha
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        self.menuRequested.emit(event.globalPos())
        event.accept()


class CollapsibleSection(QFrame):
    """Sección plegable: cabecera con flecha (▾ abierta / ▸ cerrada) y contenido.

    Ctrl+clic en la cabecera emite `soloRequested` (como en Blender: abrir solo esta) y el clic derecho, `menuRequested`."""

    soloRequested = Signal()
    menuRequested = Signal(QPoint)

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("Section")
        self._title = title

        self.header = _SectionHeader()
        self.header.ctrlClicked.connect(self.soloRequested)
        self.header.menuRequested.connect(self.menuRequested)
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.toggled.connect(self._on_toggled)

        self.body = QWidget()
        self.body.setObjectName("SectionBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 2, 8, 8)
        self.body_layout.setSpacing(4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self._on_toggled(expanded)

    def add_widget(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool, silent: bool = False) -> None:
        """Abre o cierra la sección. silent=True: sin avisar (para medir o restaurar sin que se guarde como elección)."""
        if silent:
            self.header.blockSignals(True)
        self.header.setChecked(expanded)
        if silent:
            self.header.blockSignals(False)
            self._on_toggled(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        arrow = "▾" if expanded else "▸"
        self.header.setText(f"{arrow}  {self._title}")
        self.body.setVisible(expanded)


class SegmentedControl(QWidget):
    """Grupo de botones pegados con una sola opción activa (p. ej. Rampa | Negro | Blanco).

    Cada opción es un texto o una pareja (texto, nombre de icono de `ui/icons.py`).
    Emite `changed(index)` al cambiar de opción.
    """

    changed = Signal(int)

    def __init__(self, options: list, parent: QWidget | None = None, current: int = 0) -> None:
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        for i, option in enumerate(options):
            text, icon_name = option if isinstance(option, tuple) else (option, None)
            button = QPushButton(text)
            if icon_name:
                set_icon(button, icon_name)
                button.setIconSize(QSize(16, 16))
            button.setCheckable(True)
            # La propiedad "segment" la usa el QSS para redondear solo los extremos.
            position = "first" if i == 0 else "last" if i == len(options) - 1 else "middle"
            button.setProperty("segment", position)
            self._group.addButton(button, i)
            layout.addWidget(button)

        self._group.button(current).setChecked(True)
        self._group.idClicked.connect(self.changed)

    def set_tooltips(self, tips: list[str]) -> None:
        """Ayuda de cada opción (tooltip y texto del pie de la ventana)."""
        for i, tip in enumerate(tips):
            button = self._group.button(i)
            button.setToolTip(tip)
            button.setStatusTip(tip)

    def current_index(self) -> int:
        return self._group.checkedId()

    def set_current_index(self, index: int) -> None:
        self._group.button(index).setChecked(True)


class LibraryList(QListWidget):
    """Biblioteca de fotos: miniaturas con menú contextual, Supr para quitar y doble clic
    en el espacio vacío para importar. No borra archivos del disco: solo quita de la lista."""

    importRequested = Signal()
    removeRequested = Signal(list)     # lista de rutas (str) a quitar

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    def selected_paths(self) -> list[str]:
        return [item.data(Qt.UserRole) for item in self.selectedItems()]

    def mouseDoubleClickEvent(self, event) -> None:
        if self.itemAt(event.position().toPoint()) is None:
            self.importRequested.emit()          # doble clic en zona vacía: importar fotos
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selectedItems():
            self.removeRequested.emit(self.selected_paths())
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        add = menu.addAction(icon("plus"), self.tr("Importar fotografías…"))
        add.triggered.connect(self.importRequested)
        if self.itemAt(event.pos()) is not None:
            if not self.itemAt(event.pos()).isSelected():
                self.setCurrentItem(self.itemAt(event.pos()))
            menu.addSeparator()
            remove = menu.addAction(icon("trash"), self.tr("Quitar de la biblioteca"))
            remove.setStatusTip(self.tr("Solo la quita de la lista: el archivo no se borra del disco"))
            remove.triggered.connect(lambda: self.removeRequested.emit(self.selected_paths()))
        menu.exec(event.globalPos())
