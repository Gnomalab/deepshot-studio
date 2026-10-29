"""Paneles de edición del paso "Enfoque y expresión": relieve, bandas, áreas y mezcla.

Cada panel edita directamente el `Project` que recibe en `bind()` y emite:
  changed()    en cada cambio (la vista previa se recalcula al momento);
  committed()  al terminar un gesto (se guarda un paso de deshacer).
"""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QRect, QSize, Qt, QT_TRANSLATE_NOOP, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyledItemDelegate,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from deepshot.ui import theme
from deepshot import presets
from deepshot.areas import Area
from deepshot.depth_control import MAX_EXTRA, DepthControl
from deepshot.ui.color_pick import pick_color
from deepshot.ui.icons import icon, set_icon
from deepshot.ui.preset_bar import PresetBar
from deepshot.project import Project
from deepshot.tonemap import Band
from deepshot.ui.value_slider import ValueSlider
from deepshot.ui.widgets import CollapsibleSection, SegmentedControl


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("dim", True)
    return label


def _swatch(button: QPushButton, color: tuple[float, float, float]) -> None:
    c = QColor.fromRgbF(*color)
    button.setStyleSheet(f"background: {c.name()}; border: 1px solid {theme.color('line').name()};")


class _Panel(QWidget):
    changed = Signal()
    committed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project: Project | None = None
        self._loading = False   # True mientras se rellenan los controles desde el proyecto
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(4)

    def slider(self, label: str, setter, **kwargs) -> ValueSlider:
        """Crea un ValueSlider que escribe en el proyecto con `setter(valor)`."""
        s = ValueSlider(label, **kwargs)

        def on_change(value: float) -> None:
            if not self._loading and self.project is not None:
                setter(value)
                self.changed.emit()

        s.valueChanged.connect(on_change)
        s.editingFinished.connect(self.committed)
        return s

    def emit_both(self) -> None:
        if not self._loading:
            self.changed.emit()
            self.committed.emit()


# ---------------------------------------------------------------------------
# Base y relieve
# ---------------------------------------------------------------------------
class BasePanel(_Panel):
    """Base de la imagen (rampa / negro / blanco) sobre la que se pintan bandas y áreas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.base = SegmentedControl([self.tr("Rampa"), self.tr("Negro"), self.tr("Blanco")])
        self.base.setToolTip(self.tr("Sobre qué se pintan las bandas y las áreas"))
        self.base.changed.connect(self._on_base)
        self.layout_.addWidget(_hint(self.tr("Bandas y áreas se aplican sobre:")))
        self.layout_.addWidget(self.base)

    BASES = ["ramp", "black", "white"]

    def bind(self, project: Project) -> None:
        self.project = project
        self._loading = True
        self.base.set_current_index(self.BASES.index(project.base))
        self._loading = False

    def _on_base(self, index: int) -> None:
        if self.project is not None and not self._loading:
            self.project.base = self.BASES[index]
            self.emit_both()


class ReliefPanel(_Panel):
    """Escala de profundidad y cámara virtual (afectan a las áreas 3D)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.relief = self.slider(self.tr("Profundidad"), lambda v: setattr(self.project.scene, "relief", v),
                                  value=1.0, minimum=0.01, maximum=10.0, decimals=2, step=0.1, drag_range=4.0)
        self.relief.setToolTip(self.tr("Escala del relieve: cuánto se separan en 3D lo cercano y lo lejano"))
        self.fov = self.slider(self.tr("Campo de visión"), lambda v: setattr(self.project.scene, "fov_deg", v),
                               value=0.0, minimum=0.0, maximum=120.0, decimals=0, step=1, suffix="°")
        self.fov.setToolTip(self.tr("0° = sin perspectiva. 40-60° imita una cámara normal"))
        self.layout_.addWidget(self.relief)
        self.layout_.addWidget(self.fov)
        self.layout_.addWidget(_hint(self.tr("Afectan a la forma de las áreas 3D.")))

    def bind(self, project: Project) -> None:
        self.project = project
        self._loading = True
        self.relief.setValue(project.scene.relief)
        self.fov.setValue(project.scene.fov_deg)
        self._loading = False


# ---------------------------------------------------------------------------
# Lista genérica (bandas, áreas)
# ---------------------------------------------------------------------------
class _SoloDelegate(QStyledItemDelegate):
    """Pinta un ojo a la derecha de cada fila (lleno = «solo» de esa fila, tenue = sin solo) y atiende su clic."""

    SIZE = 16
    MARGIN = 6

    def _eye_rect(self, rect: QRect) -> QRect:
        return QRect(rect.right() - self.SIZE - self.MARGIN, rect.center().y() - self.SIZE // 2, self.SIZE, self.SIZE)

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        panel = self.parent()
        active = index.row() in panel.solo
        painter.save()
        painter.setOpacity(1.0 if active else 0.28)
        painter.drawPixmap(self._eye_rect(option.rect), icon("eye").pixmap(self.SIZE, self.SIZE))
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton \
                and self._eye_rect(option.rect).adjusted(-3, -3, 3, 3).contains(event.position().toPoint()):
            self.parent().toggle_row_solo(index.row())
            return True
        return super().editorEvent(event, model, option, index)

    def helpEvent(self, event, view, option, index) -> bool:
        if self._eye_rect(option.rect).contains(event.pos()):
            QToolTip.showText(event.globalPos(), view.tr("Solo: ve solo este elemento (clic para ponerlo o quitarlo)"), view)
            return True
        return super().helpEvent(event, view, option, index)


class _ListPanel(_Panel):
    """Lista con casilla de activación, botones añadir / duplicar / borrar y propiedades."""

    selectionChanged = Signal(int)

    def __init__(self, add_text: str, add_tip: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.list = QListWidget()
        self.list.setMaximumHeight(96)
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.setItemDelegate(_SoloDelegate(self))              # el ojo de «solo» a la derecha de cada fila
        self.list.setMouseTracking(True)
        add = QPushButton(add_text)
        add.setToolTip(add_tip or self.tr("Añade un elemento nuevo a la lista"))
        add.clicked.connect(self._add)
        dup = QPushButton(self.tr("Duplicar"))
        dup.setToolTip(self.tr("Copia el elemento seleccionado"))
        dup.clicked.connect(self._duplicate)
        remove = QPushButton(self.tr("Borrar"))
        remove.setToolTip(self.tr("Borra el elemento seleccionado (se puede deshacer con Ctrl+Z)"))
        remove.clicked.connect(self._remove)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(add, 1)
        row.addWidget(dup)
        row.addWidget(remove)
        # Solo, como en un programa de audio o de capas: con el modo activado, solo se ve el elemento elegido en la lista y
        # al elegir otro, el «solo» se va con él. Además, el ojo de cada fila pone o quita el «solo» de esa fila.
        self.solo: set[int] = set()                  # filas en «solo» (solo de vista: no se guarda ni se exporta)
        self.solo_mode = False                       # el «solo» sigue a la fila elegida
        self.solo_button = QPushButton(self.tr("Solo"))
        set_icon(self.solo_button, "eye")
        self.solo_button.setIconSize(QSize(16, 16))
        self.solo_button.setCheckable(True)
        self.solo_button.setToolTip(self.tr(
            "Solo: ve únicamente el elemento elegido en la lista; al elegir otro, ves solo ese. Los demás se ocultan en "
            "la vista previa y sus manejadores. Solo es de vista: no cambia lo que se exporta. Para silenciar uno, "
            "desmarca su casilla."))
        self.solo_button.toggled.connect(self._solo_toggled)
        solo_row = QHBoxLayout()
        solo_row.setSpacing(4)
        solo_row.addWidget(self.solo_button, 1)
        self.props = QWidget()
        self.props_layout = QVBoxLayout(self.props)
        self.props_layout.setContentsMargins(0, 4, 0, 0)
        self.props_layout.setSpacing(4)
        self.layout_.addLayout(row)
        self.layout_.addWidget(self.list)
        self.layout_.addLayout(solo_row)
        self.layout_.addWidget(self.props)
        # Al terminar cada gesto, el texto de la lista refleja los valores nuevos.
        self.committed.connect(self.refresh_current_label)

    # A implementar por cada lista:
    def items(self) -> list: ...
    def new_item(self): ...
    def item_label(self, i: int, item) -> str: ...
    def show_props(self, item) -> None: ...
    def item_icon(self, i: int, item) -> QIcon | None:
        return None

    def bind(self, project: Project) -> None:
        self.project = project
        self.solo.clear()                            # otra foto o un deshacer: el «solo» no se arrastra
        self.solo_mode = False
        self._loading = True
        self.solo_button.setChecked(False)
        self._loading = False
        self.refresh_list(select=0)

    def _valid_solo(self) -> set[int]:
        """Las filas en «solo» que existen (un índice de una fila ya borrada no esconde nada)."""
        count = len(self.items())
        return {r for r in self.solo if 0 <= r < count}

    def hidden_by_solo(self, row: int) -> bool:
        """True si hay elementos en «solo» y este no es uno de ellos (no se ve ni se dibuja)."""
        solo = self._valid_solo()
        return bool(solo) and row not in solo

    def _show_solo_state(self) -> None:
        """El botón Solo está marcado siempre que haya algo en «solo» (aunque sea por el ojo de una fila): así se ve que
        hay elementos escondidos y un clic los devuelve."""
        self._loading = True
        self.solo_button.setChecked(bool(self._valid_solo()))
        self._loading = False

    def _label(self, i: int, item) -> str:
        return self.item_label(i, item)

    def _solo_changed(self) -> None:
        self.list.viewport().update()                # los ojos de la lista
        self.changed.emit()                          # se vuelve a pintar la vista previa y los manejadores (sin deshacer)

    def _solo_toggled(self, on: bool) -> None:
        """El botón Solo: activa o desactiva el modo que sigue a la fila elegida."""
        if self._loading:
            return
        self.solo_mode = on                          # activado: sigue a la fila elegida · desactivado: todo vuelve
        row = self.list.currentRow()
        self.solo = {row} if on and 0 <= row < len(self.items()) else set()
        self._solo_changed()

    def toggle_row_solo(self, row: int) -> None:
        """El ojo de una fila: pone o quita el «solo» de esa fila (se pueden dejar varias)."""
        if not 0 <= row < len(self.items()):
            return
        self.solo.symmetric_difference_update({row})
        self.solo_mode = False                       # a mano: ya no sigue a la selección
        self._show_solo_state()
        self._solo_changed()

    def _sync_solo_buttons(self) -> None:
        """Con el modo «solo» activado, el solo se va a la fila elegida."""
        row = self.list.currentRow()
        if self.solo_mode and 0 <= row < len(self.items()) and self.solo != {row}:
            self.solo = {row}
            self.list.viewport().update()
        self.solo = self._valid_solo()               # nunca queda un «solo» de una fila que ya no existe
        self._show_solo_state()
        self.solo_button.setEnabled(self.current() is not None)

    def refresh_list(self, select: int | None = None) -> None:
        self._loading = True
        current = self.list.currentRow() if select is None else select
        self.list.clear()
        for i, item in enumerate(self.items()):
            entry = QListWidgetItem(self.item_label(i, item))
            entry.setFlags(entry.flags() | Qt.ItemIsUserCheckable)
            if (ic := self.item_icon(i, item)) is not None:
                entry.setIcon(ic)
            entry.setCheckState(Qt.Checked if item.enabled else Qt.Unchecked)
            self.list.addItem(entry)
        self._loading = False
        count = self.list.count()
        self.props.setVisible(count > 0)
        if count:
            self.list.setCurrentRow(min(max(current, 0), count - 1))
            self._on_row(self.list.currentRow())

    def refresh_current_label(self) -> None:
        row = self.list.currentRow()
        items = self.items()
        if 0 <= row < len(items) and self.list.item(row) is not None:
            self._loading = True
            self.list.item(row).setText(self.item_label(row, items[row]))
            if (ic := self.item_icon(row, items[row])) is not None:
                self.list.item(row).setIcon(ic)
            self._loading = False

    def current(self):
        row = self.list.currentRow()
        items = self.items()
        return items[row] if 0 <= row < len(items) else None

    def _on_row(self, row: int) -> None:
        item = self.current()
        if item is not None and not self._loading:
            self._loading = True
            self.show_props(item)
            self._loading = False
        self._sync_solo_buttons()
        self.selectionChanged.emit(row)

    def _on_item_changed(self, entry: QListWidgetItem) -> None:
        if self._loading or self.project is None:
            return
        row = self.list.row(entry)
        self.items()[row].enabled = entry.checkState() == Qt.Checked
        self._sync_solo_buttons()
        self.emit_both()

    def _add(self) -> None:
        if self.project is not None:
            self.add_item(self.new_item())

    def add_item(self, item, commit: bool = True) -> None:
        """Añade un elemento y lo selecciona. commit=False: sin paso de deshacer (lo hará quien
        siga editando, p. ej. al dibujar un área arrastrando)."""
        if self.project is None:
            return
        self.items().append(item)
        if self.solo:                                # con un «solo» activo, lo nuevo también se ve
            self.solo.add(len(self.items()) - 1)
        self.refresh_list(select=len(self.items()) - 1)
        if commit:
            self.emit_both()

    def select(self, row: int) -> None:
        if 0 <= row < self.list.count() and row != self.list.currentRow():
            self.list.setCurrentRow(row)

    def sync_current(self) -> None:
        """Refresca controles y texto de la fila tras un cambio hecho desde fuera (p. ej. arrastrando
        en el lienzo), sin reconstruir la lista."""
        item = self.current()
        if item is not None:
            self._loading = True
            self.show_props(item)
            self._loading = False
            self.refresh_current_label()

    def _duplicate(self) -> None:
        item = self.current()
        if item is not None:
            clone = copy.deepcopy(item)
            if hasattr(clone, "from_style"):
                clone.from_style = ""                  # una copia es del usuario, aunque el original viniera de un estilo
            self.items().append(clone)
            if self.solo:
                self.solo.add(len(self.items()) - 1)
            self.refresh_list(select=len(self.items()) - 1)
            self.emit_both()

    def _remove(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.items()):
            del self.items()[row]
            self.solo.clear()                        # las filas se corren: se quita el «solo»
            self.refresh_list(select=row - 1)
            self.emit_both()


# ---------------------------------------------------------------------------
# Bandas
# ---------------------------------------------------------------------------
class BandsPanel(_ListPanel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self.tr("+ Añadir banda"), self.tr("Añade una franja de profundidad que se ilumina "
                                                             "u oscurece sobre la base"), parent)
        cur = lambda: self.current()  # noqa: E731 (atajo legible para los setters)
        self.position = self.slider(self.tr("Posición"), lambda v: setattr(cur(), "position", v),
                                    value=0.5, decimals=3, step=0.005)
        self.position.setToolTip(self.tr("Profundidad del centro de la banda (0 = cerca, 1 = lejos)"))
        self.width_ = self.slider(self.tr("Anchura"), lambda v: setattr(cur(), "width", v),
                                  value=0.1, maximum=1.0, decimals=3, step=0.005, drag_range=0.5)
        self.width_.setToolTip(self.tr("Anchura de la franja con efecto pleno, en el rango de la rampa (0 a 1)"))
        self.softness = self.slider(self.tr("Suavidad"), lambda v: setattr(cur(), "softness", v),
                                    value=0.05, maximum=0.5, decimals=3, step=0.005)
        self.softness.setToolTip(self.tr("Cuánto se difumina la franja a cada lado: 0 = borde duro"))
        self.strength = self.slider(self.tr("Intensidad"), lambda v: setattr(cur(), "strength", v), value=1.0)
        self.strength.setToolTip(self.tr("Opacidad de la banda: 0 = invisible, 1 = totalmente visible"))
        # Pan y tilt: la franja deja de ser un plano paralelo a la cámara y se inclina (como un giro del plano).
        self.pan = self.slider(self.tr("Pan"), lambda v: setattr(cur(), "pan", v), value=0.0, minimum=-80, maximum=80,
                               decimals=0, step=1, suffix="°")
        self.pan.setToolTip(self.tr("Gira la banda sobre el eje vertical: la franja se aleja hacia la derecha (+) "
                                    "o hacia la izquierda (−). El centro de la imagen no cambia."))
        self.tilt = self.slider(self.tr("Tilt"), lambda v: setattr(cur(), "tilt", v), value=0.0, minimum=-80,
                                maximum=80, decimals=0, step=1, suffix="°")
        self.tilt.setToolTip(self.tr("Gira la banda sobre el eje horizontal: la franja se aleja hacia arriba (+) "
                                     "o se acerca (−). El centro de la imagen no cambia."))
        self.color = QPushButton()
        self.color.setToolTip(self.tr("Color de la banda"))
        self.color.clicked.connect(self._pick_color)
        self.color_ramp = QCheckBox(self.tr("De la rampa"))
        self.color_ramp.setToolTip(self.tr("La banda toma el color de la rampa artística en su posición, así siempre "
                                           "casa con el resto. Si está marcado, el color de al lado no se usa."))
        self.color_ramp.toggled.connect(self._on_color_ramp)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel(self.tr("Color")))
        color_row.addWidget(self.color, 1)
        color_row.addWidget(self.color_ramp)
        # Controles de dos en dos: ocupan la mitad de líneas (como los campos X Y Z de Blender).
        for pair in ((self.position, self.width_), (self.softness, self.strength), (self.pan, self.tilt)):
            row = QHBoxLayout()
            row.setSpacing(4)
            for w in pair:
                row.addWidget(w, 1)
            self.props_layout.addLayout(row)
        self.props_layout.addLayout(color_row)

    def items(self) -> list[Band]:
        return self.project.bands if self.project else []

    def new_item(self) -> Band:
        return Band(position=0.3, width=0.08, softness=0.05)

    def item_label(self, i: int, band: Band) -> str:
        return self.tr("Banda {0}  ·  {1:.2f}").format(i + 1, band.position)

    def show_props(self, band: Band) -> None:
        self.position.setValue(band.position)
        self.width_.setValue(band.width)
        self.softness.setValue(band.softness)
        self.strength.setValue(band.strength)
        self.pan.setValue(band.pan)
        self.tilt.setValue(band.tilt)
        self.color_ramp.setChecked(band.color_from_ramp)
        self.color.setEnabled(not band.color_from_ramp)
        _swatch(self.color, band.color)

    def _on_color_ramp(self, on: bool) -> None:
        band = self.current()
        if band is None or self._loading:
            return
        band.color_from_ramp = on
        self.color.setEnabled(not on)
        self.emit_both()

    def _pick_color(self) -> None:
        band = self.current()
        if band is None:
            return

        def live(color) -> None:      # vista previa en vivo mientras se elige
            band.color = color
            _swatch(self.color, color)
            self.changed.emit()

        if pick_color(self, band.color, self.tr("Color de la banda"), live) is not None:
            self.committed.emit()


# ---------------------------------------------------------------------------
# Áreas
# ---------------------------------------------------------------------------
def _color_icon(color: tuple[float, float, float]) -> QIcon:
    """Cuadradito de color para la lista de áreas."""
    pix = QPixmap(14, 14)
    pix.fill(QColor.fromRgbF(*color))
    return QIcon(pix)


class AreasPanel(_ListPanel):
    """Áreas 3D. Se pueden crear y mover con las herramientas «Dibujar» y «Colocar», o arrastrando sobre
    la imagen (ver ui/canvas.py); aquí están además los valores exactos."""

    toolChanged = Signal(str)       # "" (ninguna), "draw" o "place"
    SHAPES = ["ellipsoid", "box"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self.tr("+ Añadir área"), self.tr("Añade un área 3D de luz o enfoque en el centro; "
                                                            "después se mueve y se escala sobre la imagen"), parent)
        cur = lambda: self.current()  # noqa: E731

        # Herramientas del lienzo (se activan una a la vez; vuelven solas al terminar el gesto).
        self.draw_button = QPushButton(self.tr("Dibujar"))
        set_icon(self.draw_button, "draw_area")
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip(self.tr("Dibuja un área nueva sobre la imagen: pulsa en el centro y "
                                            "arrastra para darle tamaño; el efecto se ve al instante "
                                            "(Mayús+A). Mayús al arrastrar: caja rectangular."))
        self.place_button = QPushButton(self.tr("Colocar"))
        set_icon(self.place_button, "target")
        self.place_button.setCheckable(True)
        self.place_button.setToolTip(self.tr("Mueve el área seleccionada: pulsa y arrastra por la imagen "
                                             "y el área sigue al puntero, pegada a la superficie, hasta "
                                             "soltar (P)"))
        for b in (self.draw_button, self.place_button):
            b.setIconSize(QSize(16, 16))
        self.draw_button.toggled.connect(lambda on: self._tool_toggled("draw", on))
        self.place_button.toggled.connect(lambda on: self._tool_toggled("place", on))
        tools = QHBoxLayout()
        tools.setSpacing(4)
        tools.addWidget(self.draw_button, 1)
        tools.addWidget(self.place_button, 1)

        self.name = QLineEdit()
        self.name.setPlaceholderText(self.tr("Nombre del área"))
        self.name.setToolTip(self.tr("Nombre del área (doble clic en la lista para editarlo)"))
        self.name.editingFinished.connect(self._on_name)
        self.color_button = QPushButton()
        self.color_button.setFixedWidth(34)
        self.color_button.setToolTip(self.tr("Color con el que se dibuja esta área sobre la imagen "
                                             "(no afecta al resultado)"))
        self.color_button.clicked.connect(self._pick_area_color)
        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        name_row.addWidget(self.name, 1)
        name_row.addWidget(self.color_button)
        self.list.itemDoubleClicked.connect(lambda _item: (self.name.setFocus(), self.name.selectAll()))
        self.list.setToolTip(self.tr("Áreas. Casilla: activar o desactivar · clic: seleccionar"))

        self.shape = SegmentedControl([self.tr("Elipsoide"), self.tr("Caja")])
        self.shape.set_tooltips([self.tr("Zona ovalada con el borde que se desvanece"),
                                 self.tr("Zona rectangular (caja 3D)")])
        self.shape.changed.connect(self._on_shape)

        def vec(attr, axis):
            def setter(v):
                values = list(getattr(cur(), attr))
                values[axis] = v
                setattr(cur(), attr, tuple(values))
            return setter

        # Etiquetas cortas (X, Y, Z): van tres por fila y el título de la fila dice de qué son.
        self.center = [self.slider(a, vec("center", i), value=0.0, minimum=None,
                                   maximum=None, decimals=3, step=0.01, drag_range=2.0)
                       for i, a in enumerate("XYZ")]
        self.scale = [self.slider(a, vec("scale", i), value=0.4, minimum=0.001,
                                  maximum=None, decimals=3, step=0.01, drag_range=1.0)
                      for i, a in enumerate("XYZ")]
        self.rotation = [self.slider(a, vec("rotation", i), value=0.0, minimum=-180,
                                     maximum=180, decimals=0, step=1, suffix="°")
                         for i, a in enumerate("XYZ")]
        for sliders, tips in (
            (self.center, [self.tr("Posición horizontal del centro del área"),
                           self.tr("Posición vertical del centro del área (arriba = positivo)"),
                           self.tr("Profundidad del centro: 0 = lo más cercano")]),
            (self.scale, [self.tr("Tamaño horizontal (semieje X)"), self.tr("Tamaño vertical (semieje Y)"),
                          self.tr("Grosor en profundidad (semieje Z): qué tanto abarca hacia delante y atrás")]),
            (self.rotation, [self.tr("Giro sobre el eje horizontal"), self.tr("Giro sobre el eje vertical"),
                             self.tr("Giro sobre la imagen (el que se ve al arrastrar el tirador de giro)")]),
        ):
            for sl, tip in zip(sliders, tips):
                sl.setToolTip(tip)
        self.strength = self.slider(self.tr("Fuerza"), lambda v: setattr(cur(), "strength", v), value=1.0,
                                    minimum=-1.0, maximum=1.0, decimals=2, step=0.05)
        self.strength.setToolTip(self.tr("Positiva aclara, negativa oscurece"))
        self.softness = self.slider(self.tr("Suavidad del borde"), lambda v: setattr(cur(), "edge_softness", v),
                                    value=0.5)
        self.softness.setToolTip(self.tr("Cuánto se difumina el borde del área. Se ve como el anillo "
                                         "punteado interior; también: Alt + rueda sobre la imagen"))
        self.falloff = self.slider(self.tr("Caída"), lambda v: setattr(cur(), "falloff", v), value=0.5)
        self.falloff.setToolTip(self.tr("0 = intensidad uniforme; 1 = se apaga del centro hacia el borde"))

        self.props_layout.addLayout(name_row)
        self.props_layout.addWidget(self.shape)
        for w in (self.strength, self.softness, self.falloff):
            self.props_layout.addWidget(w)
        # Centro, escala y rotación: tres campos por fila dentro de un apartado que empieza plegado (se mueven y
        # escalan sobre la imagen casi siempre; los números son para afinar).
        self.transform = CollapsibleSection(self.tr("Transformación"), expanded=False)
        self.transform.setToolTip(self.tr("Centro, escala y rotación del área con números exactos"))
        for group, title in ((self.center, self.tr("Centro")), (self.scale, self.tr("Escala")),
                             (self.rotation, self.tr("Rotación"))):
            self.transform.add_widget(_hint(title))
            row = QHBoxLayout()
            row.setSpacing(3)
            for w in group:
                row.addWidget(w, 1)
            self.transform.body_layout.addLayout(row)
        self.props_layout.addWidget(self.transform)
        self.props_layout.addWidget(_hint(self.tr(
            "En la imagen: arrastra el área para moverla; tiradores para escalar y girar. "
            "G mover · S escalar · R girar (X / Y / Z fijan el eje). "
            "Rueda con Ctrl: tamaño · Mayús: profundidad · Alt: suavidad.")))
        # Las herramientas van arriba de la lista, antes que las propiedades.
        self.layout_.insertLayout(0, tools)

    # --- Herramientas ---------------------------------------------------------------------
    def tool(self) -> str:
        return "draw" if self.draw_button.isChecked() else "place" if self.place_button.isChecked() else ""

    def set_tool(self, tool: str, emit: bool = True) -> None:
        """Activa una herramienta ("" = ninguna) sin duplicar señales. emit=False: sin avisar
        (cambio hecho porque se activó otra herramienta)."""
        for button, name in ((self.draw_button, "draw"), (self.place_button, "place")):
            button.blockSignals(True)
            button.setChecked(tool == name)
            button.blockSignals(False)
        if emit:
            self.toolChanged.emit(tool)

    def _tool_toggled(self, name: str, on: bool) -> None:
        self.set_tool(name if on else "")

    # --- Lista --------------------------------------------------------------------------------
    def items(self) -> list[Area]:
        return self.project.areas if self.project else []

    def new_item(self) -> Area:
        return Area(name=self.tr("Área {0}").format(len(self.items()) + 1), center=(0.0, 0.0, 0.3),
                    scale=(0.4, 0.4, 0.2))

    def item_label(self, i: int, area: Area) -> str:
        shape = self.tr("elipsoide") if area.shape == "ellipsoid" else self.tr("caja")
        return f"{area.name}  ·  {shape}  ·  {area.strength:+.2f}"

    def item_icon(self, i: int, area: Area) -> QIcon:
        return _color_icon(area.display_color(i))

    def show_props(self, area: Area) -> None:
        self.shape.set_current_index(self.SHAPES.index(area.shape))
        for sliders, values in ((self.center, area.center), (self.scale, area.scale),
                                (self.rotation, area.rotation)):
            for s, v in zip(sliders, values):
                s.setValue(v)
        self.strength.setValue(area.strength)
        self.softness.setValue(area.edge_softness)
        self.falloff.setValue(area.falloff)
        if not self.name.hasFocus():
            self.name.setText(area.name)
        _swatch(self.color_button, area.display_color(self.list.currentRow()))

    def _on_name(self) -> None:
        area = self.current()
        if area is None or self._loading:
            return
        text = self.name.text().strip() or self.tr("Área {0}").format(self.list.currentRow() + 1)
        if text != area.name:
            area.name = text
            self.refresh_current_label()
            self.emit_both()
        self.name.setText(text)

    def _pick_area_color(self) -> None:
        area = self.current()
        if area is None:
            return
        row = self.list.currentRow()
        original = area.ui_color

        def live(color) -> None:
            area.ui_color = color
            _swatch(self.color_button, color)
            self.refresh_current_label()
            self.changed.emit()       # redibuja los contornos (no cambia la imagen)

        if pick_color(self, area.display_color(row), self.tr("Color del área"), live) is not None:
            self.committed.emit()
        else:
            area.ui_color = original
            _swatch(self.color_button, area.display_color(row))
            self.refresh_current_label()
            self.changed.emit()

    def _on_shape(self, index: int) -> None:
        area = self.current()
        if area is not None and not self._loading:
            area.shape = self.SHAPES[index]
            self.refresh_list()
            self.emit_both()

    def update_current_from_project(self) -> None:
        """Refresca controles y lista tras un cambio hecho fuera (compatibilidad)."""
        self.sync_current()


# ---------------------------------------------------------------------------
# Mezcla y máscaras
# ---------------------------------------------------------------------------
class MixPanel(_Panel):
    MODES = [("normal", QT_TRANSLATE_NOOP("Panels", "Normal")), ("multiply", QT_TRANSLATE_NOOP("Panels", "Multiplicar")),
             ("screen", QT_TRANSLATE_NOOP("Panels", "Trama")), ("overlay", QT_TRANSLATE_NOOP("Panels", "Superponer")),
             ("soft_light", QT_TRANSLATE_NOOP("Panels", "Luz suave")), ("add", QT_TRANSLATE_NOOP("Panels", "Sumar")),
             ("darken", QT_TRANSLATE_NOOP("Panels", "Oscurecer")), ("lighten", QT_TRANSLATE_NOOP("Panels", "Aclarar")),
             ("difference", QT_TRANSLATE_NOOP("Panels", "Diferencia"))]
    MASKS = [("none", QT_TRANSLATE_NOOP("Panels", "Sin máscara")), ("areas", QT_TRANSLATE_NOOP("Panels", "Áreas")), ("depth_range", QT_TRANSLATE_NOOP("Panels", "Rango de profundidad")),
             ("painted", QT_TRANSLATE_NOOP("Panels", "Pintada con pincel"))]

    paintToolChanged = Signal(str)      # "" (ninguna), "paint" o "erase"
    brushChanged = Signal()             # cambió el tamaño, la dureza o la opacidad del pincel
    maskOperation = Signal(str)         # "fill" | "clear" | "invert"
    maskFileChosen = Signal(str)        # ruta de un PNG a cargar como máscara
    showMaskChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        comp = lambda: self.project.composite  # noqa: E731
        self.mode = QComboBox()
        for key, label in self.MODES:
            self.mode.addItem(QCoreApplication.translate("Panels", label), key)
        self.mode.setToolTip(self.tr("Cómo se combina la foto con el resultado"))
        self.mode.activated.connect(lambda _i: self._set("mode", self.mode.currentData()))
        self.opacity = self.slider(self.tr("Foto"), lambda v: setattr(comp(), "opacity", v), value=0.0,
                                   decimals=2, step=0.05)
        self.opacity.setToolTip(self.tr("Cantidad de foto: 0 = solo el resultado"))
        self.gray = QCheckBox(self.tr("Foto en blanco y negro"))
        self.gray.setToolTip(self.tr("Convierte la foto a blanco y negro antes de mezclarla"))
        self.gray.toggled.connect(lambda v: self._set("photo_grayscale", v))

        self.mask = QComboBox()
        for key, label in self.MASKS:
            self.mask.addItem(QCoreApplication.translate("Panels", label), key)
        self.mask.setToolTip(self.tr("Dónde se aplica la foto: en todas partes, solo dentro de las áreas, "
                                     "en un rango de profundidad o donde la pintes con el pincel"))
        self.mask.activated.connect(lambda _i: self._set("mask", self.mask.currentData()))
        self.invert = QCheckBox(self.tr("Invertir máscara"))
        self.invert.setToolTip(self.tr("Intercambia dónde se aplica y dónde no la foto"))
        self.invert.toggled.connect(lambda v: self._set("invert_mask", v))
        self.near = self.slider(self.tr("Desde"), lambda v: setattr(comp(), "depth_near", v), value=0.0,
                                decimals=3, step=0.01)
        self.far = self.slider(self.tr("Hasta"), lambda v: setattr(comp(), "depth_far", v), value=0.5,
                               decimals=3, step=0.01)
        self.soft = self.slider(self.tr("Suavidad"), lambda v: setattr(comp(), "depth_softness", v),
                                value=0.05, maximum=0.5, decimals=3, step=0.01)
        self.near.setToolTip(self.tr("Profundidad desde la que se aplica la foto (0 = lo más cercano)"))
        self.far.setToolTip(self.tr("Profundidad hasta la que se aplica la foto (1 = lo más lejano)"))
        self.soft.setToolTip(self.tr("Cuánto se difumina el paso entre lo que se aplica y lo que no"))
        # Retoque de cualquier máscara: difuminarla y pegarla a los bordes de la foto.
        self.feather = self.slider(self.tr("Suavizado"), lambda v: setattr(comp(), "mask_feather", v), value=0.0)
        self.feather.setToolTip(self.tr("Difumina el borde de la máscara (de la que sea). Con valores altos la foto se "
                                        "funde poco a poco en lugar de cortar en seco."))
        self.snap = self.slider(self.tr("Pegar a bordes"), lambda v: setattr(comp(), "mask_snap", v), value=0.0)
        self.snap.setToolTip(self.tr("Ajusta el borde de la máscara a los bordes de la foto: no hace falta pintar con "
                                     "precisión, la máscara se pega al objeto. Tarda algo más en fotos grandes."))
        self.retouch_box = QWidget()
        retouch = QHBoxLayout(self.retouch_box)
        retouch.setContentsMargins(0, 0, 0, 0)
        retouch.setSpacing(4)
        retouch.addWidget(self.feather, 1)
        retouch.addWidget(self.snap, 1)
        self.range_box = QWidget()
        rl = QVBoxLayout(self.range_box)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        for w in (self.near, self.far, self.soft):
            rl.addWidget(w)
        self._build_paint_box()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(self.tr("Modo")))
        mode_row.addWidget(self.mode, 1)
        mask_row = QHBoxLayout()
        mask_row.addWidget(QLabel(self.tr("Máscara")))
        mask_row.addWidget(self.mask, 1)
        for item in (mode_row, self.opacity, self.gray, mask_row, self.invert, self.retouch_box, self.range_box,
                     self.paint_box):
            (self.layout_.addLayout if isinstance(item, QHBoxLayout) else self.layout_.addWidget)(item)

    def bind(self, project: Project) -> None:
        self.project = project
        c = project.composite
        self._loading = True
        self.mode.setCurrentIndex(max(0, self.mode.findData(c.mode)))
        self.opacity.setValue(c.opacity)
        self.gray.setChecked(c.photo_grayscale)
        self.mask.setCurrentIndex(max(0, self.mask.findData(c.mask)))
        self.invert.setChecked(c.invert_mask)
        self.feather.setValue(c.mask_feather)
        self.snap.setValue(c.mask_snap)
        self.near.setValue(c.depth_near)
        self.far.setValue(c.depth_far)
        self.soft.setValue(c.depth_softness)
        self._loading = False
        self._update_visibility()

    def _set(self, attr: str, value) -> None:
        if self.project is None or self._loading:
            return
        setattr(self.project.composite, attr, value)
        self._update_visibility()
        self.emit_both()

    def _update_visibility(self) -> None:
        mask = self.project.composite.mask if self.project else "none"
        self.invert.setVisible(mask != "none")
        self.retouch_box.setVisible(mask != "none")
        self.range_box.setVisible(mask == "depth_range")
        self.paint_box.setVisible(mask == "painted")
        if mask != "painted" and self.paint_tool():
            self.set_paint_tool("")              # sin máscara pintada no hay pincel

    # --- Pincel de la máscara pintada --------------------------------------------------------------------------
    def _build_paint_box(self) -> None:
        """Pincel, borrador, tamaño / dureza / opacidad y operaciones sobre toda la máscara."""
        self.paint_box = QWidget()
        box = QVBoxLayout(self.paint_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        self.paint_button = QPushButton(self.tr("Pincel"))
        set_icon(self.paint_button, "brush")
        self.paint_button.setToolTip(self.tr("Pinta la máscara con el ratón sobre la imagen: ahí se aplica la "
                                             "foto (B). [ y ] cambian el tamaño."))
        self.erase_button = QPushButton(self.tr("Borrador"))
        set_icon(self.erase_button, "eraser")
        self.erase_button.setToolTip(self.tr("Borra la máscara con el ratón: ahí no se aplica la foto (E)"))
        for b in (self.paint_button, self.erase_button):
            b.setCheckable(True)
            b.setIconSize(QSize(16, 16))
        self.paint_button.toggled.connect(lambda on: self._paint_toggled("paint", on))
        self.erase_button.toggled.connect(lambda on: self._paint_toggled("erase", on))
        tools = QHBoxLayout()
        tools.setSpacing(4)
        tools.addWidget(self.paint_button, 1)
        tools.addWidget(self.erase_button, 1)
        box.addLayout(tools)

        self.brush_size = ValueSlider(self.tr("Tamaño"), 6.0, 0.2, 40.0, decimals=1, step=0.5, suffix=" %")
        self.brush_size.setToolTip(self.tr("Diámetro del pincel en % del lado largo de la foto (también: [ y ])"))
        self.brush_hardness = ValueSlider(self.tr("Dureza"), 0.5, 0.0, 1.0, decimals=2, step=0.05)
        self.brush_hardness.setToolTip(self.tr("0 = borde muy difuminado · 1 = borde nítido. "
                                               "Se ve como el anillo punteado del cursor"))
        self.brush_opacity = ValueSlider(self.tr("Opacidad"), 1.0, 0.01, 1.0, decimals=2, step=0.05)
        self.brush_opacity.setToolTip(self.tr("Cuánto pinta cada pasada: con valores bajos se va acumulando"))
        for sl in (self.brush_size, self.brush_hardness, self.brush_opacity):
            sl.valueChanged.connect(lambda _v: self.brushChanged.emit())
            box.addWidget(sl)

        ops = QHBoxLayout()
        ops.setSpacing(4)
        for key, label, tip in (("fill", self.tr("Rellenar"), self.tr("Todo blanco: la foto se aplica en toda la imagen")),
                                ("clear", self.tr("Vaciar"), self.tr("Todo negro: la foto no se aplica en ninguna parte")),
                                ("invert", self.tr("Invertir"), self.tr("Cambia blanco por negro y al revés"))):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, k=key: self.maskOperation.emit(k))
            ops.addWidget(b)
        box.addLayout(ops)

        self.show_mask = QCheckBox(self.tr("Ver máscara sobre la imagen"))
        self.show_mask.setChecked(True)
        self.show_mask.setToolTip(self.tr("Tiñe de rojo lo que está pintado en la máscara"))
        self.show_mask.toggled.connect(self.showMaskChanged)
        box.addWidget(self.show_mask)
        load = QPushButton(self.tr("Cargar máscara desde PNG…"))
        load.setToolTip(self.tr("Sustituye la máscara por un PNG en gris: blanco = se aplica, negro = no"))
        load.clicked.connect(self._pick_mask)
        box.addWidget(load)

    def brush_params(self) -> dict:
        return {"size": self.brush_size.value(), "hardness": self.brush_hardness.value(),
                "opacity": self.brush_opacity.value()}

    def paint_tool(self) -> str:
        return "paint" if self.paint_button.isChecked() else "erase" if self.erase_button.isChecked() else ""

    def set_paint_tool(self, tool: str, emit: bool = True) -> None:
        """Activa el pincel o el borrador ("" = ninguno). emit=False: sin avisar (cambio hecho por otra herramienta)."""
        for button, name in ((self.paint_button, "paint"), (self.erase_button, "erase")):
            button.blockSignals(True)
            button.setChecked(tool == name)
            button.blockSignals(False)
        if emit:
            self.paintToolChanged.emit(tool)

    def _paint_toggled(self, name: str, on: bool) -> None:
        self.set_paint_tool(name if on else "")

    def activate_paint(self, tool: str) -> bool:
        """Atajo B / E: elige la máscara «pintada» si hace falta y activa la herramienta."""
        if self.project is None or not self.isEnabled():
            return False
        if self.project.composite.mask != "painted":
            self.mask.setCurrentIndex(self.mask.findData("painted"))
            self._set("mask", "painted")
        self.set_paint_tool("" if self.paint_tool() == tool else tool)
        return True

    def _pick_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Máscara desde PNG"), "",
                                              self.tr("Imágenes (*.png *.tif *.tiff)"))
        if path:
            self.maskFileChosen.emit(path)


# ---------------------------------------------------------------------------
# Desenfoque de lente (DOF)
# ---------------------------------------------------------------------------
class LensPanel(_Panel):
    """Desenfoque por profundidad con apertura y diafragma de lente (ver deepshot/dof.py)."""

    TARGETS = ["result", "photo"]
    toolChanged = Signal(str)            # "pick_focus" al activar el cuentagotas, "" al soltarlo
    showFocusChanged = Signal(bool)      # «Ver zona en foco» marcado o no
    presetChosen = Signal(dict)          # preset de bokeh elegido (lo aplica el editor, con deshacer)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Presets de bokeh (retrato, macro, cine…): solo cambian el carácter del desenfoque, no el foco de la foto.
        self.preset_bar = PresetBar("lens", self.tr("Cargar bokeh…"), lambda: presets.from_project("lens", self.project))
        self.preset_bar.combo.setToolTip(self.tr("Presets de bokeh: apertura, diafragma y ojo de gato de un tipo de "
                                                 "objetivo. No tocan el foco ni el destino de tu foto."))
        self.preset_bar.chosen.connect(self.presetChosen)
        self.layout_.addWidget(self.preset_bar)
        self.enabled = QCheckBox(self.tr("Activar desenfoque de lente"))
        self.enabled.setToolTip(self.tr("Desenfoca lo que queda lejos de la profundidad en foco, como una cámara real. "
                                        "Tocar cualquier ajuste lo activa."))
        self.enabled.toggled.connect(self._on_enabled)
        self.target = SegmentedControl([self.tr("Imagen final"), self.tr("Solo la foto")])
        self.target.setToolTip(self.tr("Imagen final: desenfoca todo lo que ves. Solo la foto: desenfoca la foto "
                                       "antes de mezclarla (necesita opacidad de mezcla mayor que 0)."))
        self.target.changed.connect(self._on_target)

        def lens(name):
            return lambda v: self._set(name, v)

        self.focus = self.slider(self.tr("Foco"), lens("focus"), value=0.35, minimum=0.0, maximum=1.0,
                                 decimals=3, step=0.01)
        self.focus.setToolTip(self.tr("Profundidad que queda nítida: 0 = lo más cercano, 1 = lo más lejano "
                                      "(se ve en la vista Profundidad: cerca = claro)"))
        self.pick_button = QPushButton(self.tr("Elegir foco en la imagen"))
        set_icon(self.pick_button, "pipette")
        self.pick_button.setCheckable(True)
        self.pick_button.setToolTip(self.tr("Cuentagotas: pulsa sobre la imagen en lo que quieres nítido y el foco "
                                            "toma su profundidad (puedes arrastrar para probar). Esc cancela."))
        self.pick_button.toggled.connect(lambda on: self.toolChanged.emit("pick_focus" if on else ""))
        self.show_focus = QCheckBox(self.tr("Ver zona en foco"))
        self.show_focus.setToolTip(self.tr("Tiñe de verde sobre la imagen lo que queda nítido (cuanto más verde, "
                                           "más nítido). Solo es una ayuda: no se exporta. Atajo: Z."))
        self.show_focus.toggled.connect(self.showFocusChanged)
        self.sharp = self.slider(self.tr("Zona nítida"), lens("sharp_range"), value=0.08, minimum=0.0,
                                 maximum=0.5, decimals=3, step=0.01)
        self.sharp.setToolTip(self.tr("Cuánta profundidad a cada lado del foco se mantiene sin desenfocar"))
        self.aperture = self.slider(self.tr("Apertura (f/)"), lens("aperture"), value=2.8, minimum=1.0,
                                    maximum=22.0, decimals=1, step=0.1, drag_range=12.0)
        self.aperture.setToolTip(self.tr("Número f del objetivo: f/1.4 muy abierto (mucho desenfoque), "
                                         "f/16 muy cerrado (casi todo nítido)"))
        self.near = self.slider(self.tr("Delante del foco"), lens("near"), value=1.0, minimum=0.0, maximum=2.0,
                                decimals=2, step=0.05)
        self.near.setToolTip(self.tr("Multiplica el desenfoque de lo que está por delante del foco (0 = nada)"))
        self.far = self.slider(self.tr("Detrás del foco"), lens("far"), value=1.0, minimum=0.0, maximum=2.0,
                               decimals=2, step=0.05)
        self.far.setToolTip(self.tr("Multiplica el desenfoque de lo que está por detrás del foco (0 = nada)"))
        self.blades = self.slider(self.tr("Palas del diafragma"), lens("blades"), value=0, minimum=0, maximum=12,
                                  decimals=0, step=1)
        self.blades.setToolTip(self.tr("0 = diafragma redondo. De 3 a 12: los puntos de luz desenfocados toman "
                                       "la forma de un polígono con ese número de lados"))
        self.rotation = self.slider(self.tr("Giro del diafragma"), lens("rotation"), value=0.0, minimum=0.0,
                                    maximum=180.0, decimals=0, step=1, suffix="°")
        self.rotation.setToolTip(self.tr("Gira el polígono del diafragma (no se nota con el diafragma redondo)"))
        self.highlights = self.slider(self.tr("Altas luces"), lens("highlights"), value=0.4, minimum=0.0,
                                      maximum=1.0, decimals=2, step=0.01)
        self.highlights.setToolTip(self.tr("Cuánto más brillan los puntos de luz en el desenfoque (discos de bokeh)"))
        self.softness = self.slider(self.tr("Borde del disco"), lens("softness"), value=0.25, minimum=0.0,
                                    maximum=1.0, decimals=2, step=0.01)
        self.softness.setToolTip(self.tr("0 = disco de borde duro y definido; 1 = borde difuso, más suave"))
        self.cat_eye = self.slider(self.tr("Ojo de gato"), lens("cat_eye"), value=0.0, minimum=0.0, maximum=1.0,
                                   decimals=2, step=0.01)
        self.cat_eye.setToolTip(self.tr("Los puntos de luz desenfocados de los bordes de la imagen se recortan en "
                                        "forma de almendra, como en muchos objetivos reales. Cuanto más alto, más "
                                        "pronunciado. Calcular con esto activo tarda algo más."))

        for widget in (self.enabled, self.target, self.focus, self.pick_button, self.show_focus, self.sharp,
                       self.aperture, self.near, self.far, self.blades, self.rotation, self.highlights,
                       self.softness, self.cat_eye):
            self.layout_.addWidget(widget)
        self.layout_.addWidget(_hint(self.tr("El foco y la apertura se ven en la vista Efecto. El desenfoque "
                                             "usa el mapa de profundidad: cuanto mejor el mapa, mejor el bokeh.")))

    def bind(self, project: Project) -> None:
        self.project = project
        lens = project.lens
        self._loading = True
        self.enabled.setChecked(lens.enabled)
        self.target.set_current_index(self.TARGETS.index(lens.target) if lens.target in self.TARGETS else 0)
        for slider, value in ((self.focus, lens.focus), (self.sharp, lens.sharp_range), (self.aperture, lens.aperture),
                              (self.near, lens.near), (self.far, lens.far), (self.blades, lens.blades),
                              (self.rotation, lens.rotation), (self.highlights, lens.highlights),
                              (self.softness, lens.softness), (self.cat_eye, lens.cat_eye)):
            slider.setValue(value)
        self._loading = False

    def set_pick(self, on: bool, emit: bool = True) -> None:
        """Activa o suelta el cuentagotas (el botón y el lienzo deben ir a la par)."""
        if self.pick_button.isChecked() == on:
            return
        self.pick_button.blockSignals(not emit)
        self.pick_button.setChecked(on)
        self.pick_button.blockSignals(False)

    def refresh(self) -> None:
        """Vuelve a poner en los controles los valores del proyecto (tras elegir el foco con el cuentagotas)."""
        if self.project is not None:
            self.bind(self.project)

    def _set(self, name: str, value: float) -> None:
        """Cambia un ajuste; si el desenfoque estaba apagado, se enciende (si no, el cambio no se vería)."""
        lens = self.project.lens
        setattr(lens, name, int(round(value)) if name == "blades" else value)
        if not lens.enabled:
            lens.enabled = True
            self._loading = True
            self.enabled.setChecked(True)
            self._loading = False

    def _on_enabled(self, on: bool) -> None:
        if self.project is not None and not self._loading:
            self.project.lens.enabled = on
            self.emit_both()

    def _on_target(self, index: int) -> None:
        if self.project is not None and not self._loading:
            self.project.lens.target = self.TARGETS[index]
            self.project.lens.enabled = True
            self._loading = True
            self.enabled.setChecked(True)
            self._loading = False
            self.emit_both()


# ---------------------------------------------------------------------------
# Control de profundidad (cerca / medio / lejos)
# ---------------------------------------------------------------------------
class DepthControlPanel(_Panel):
    """Mueve por separado las zonas cercana, media y lejana del mapa (ver deepshot/depth_control.py)."""

    centersRequested = Signal()          # «Centros desde el mapa»: lo resuelve el editor, que tiene el mapa

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.enabled = QCheckBox(self.tr("Activar control de profundidad"))
        self.enabled.setToolTip(self.tr("Mueve la profundidad de lo cercano, lo medio y lo lejano por separado. "
                                        "Cambia la rampa, las bandas, el desenfoque y la posición de las áreas. "
                                        "Tocar cualquier ajuste lo activa."))
        self.enabled.toggled.connect(self._on_enabled)
        self.layout_.addWidget(self.enabled)

        self.centers_button = QPushButton(self.tr("Centros desde el mapa"))
        self.centers_button.setToolTip(self.tr("Coloca el centro de cada zona donde hay profundidad de verdad en esta "
                                               "foto (los destinos se quedan iguales: la imagen no cambia)."))
        self.centers_button.clicked.connect(self.centersRequested)
        self.reset_button = QPushButton(self.tr("Restablecer"))
        self.reset_button.setToolTip(self.tr("Devuelve las tres zonas a su estado inicial y apaga el control"))
        self.reset_button.clicked.connect(self._reset)
        self.add_button = QPushButton(self.tr("Añadir zona"))
        self.add_button.setToolTip(self.tr("Añade otra zona de profundidad (hasta {0} más) para mover un tramo "
                                           "concreto del mapa sin tocar el resto.").format(MAX_EXTRA))
        self.add_button.clicked.connect(self._add_zone)
        row = QHBoxLayout()
        row.addWidget(self.centers_button)
        row.addWidget(self.reset_button)
        self.layout_.addLayout(row)
        self.layout_.addWidget(self.add_button)

        # Una caja por zona: activar, centro, destino, suavizado y límites (automáticos o manuales).
        self.zone_widgets: dict[str, dict] = {}
        labels = {
            "near": (self.tr("Cerca"), self.tr("Lo más cercano a la cámara (valores bajos del mapa).")),
            "mid": (self.tr("Medio"), self.tr("Lo que queda a media distancia.")),
            "far": (self.tr("Lejos"), self.tr("Lo más lejano: fondo y cielo (valores altos del mapa).")),
        }
        for i in range(MAX_EXTRA):
            labels[f"extra{i}"] = (self.tr("Zona {0}").format(4 + i),
                                   self.tr("Zona añadida por ti: se reparte el eje de profundidad con las demás."))
        for name, (title, tip) in labels.items():
            w = {}
            w["on"] = QCheckBox(title)
            w["on"].setStyleSheet("font-weight: bold;")
            w["on"].setToolTip(self.tr("Zona «{0}»: activa o desactiva este tramo de profundidad. {1}").format(title, tip))
            w["on"].toggled.connect(lambda on, n=name: self._on_zone_flag(n, "enabled", on))
            w["center"] = self.slider(self.tr("Centro"), self._zone_setter(name, "center"), value=0.5, minimum=0.0,
                                      maximum=1.0, decimals=3, step=0.01)
            w["center"].setToolTip(self.tr("Valor de profundidad que representa a la zona: 0 = lo más cercano, "
                                           "1 = lo más lejano (en la vista Profundidad, cerca = claro)"))
            w["target"] = self.slider(self.tr("Destino"), self._zone_setter(name, "target"), value=0.5, minimum=0.0,
                                      maximum=1.0, decimals=3, step=0.01)
            w["target"].setToolTip(self.tr("Adónde se lleva el centro de la zona. Igual que el centro: no cambia "
                                           "nada. Más alto: la zona se aleja. Más bajo: se acerca."))
            w["feather"] = self.slider(self.tr("Suavizado"), self._zone_setter(name, "feather"), value=0.05,
                                       minimum=0.0, maximum=0.5, decimals=3, step=0.01)
            w["feather"].setToolTip(self.tr("Anchura de la transición entre esta zona y las vecinas. "
                                            "Más alto: el cambio se reparte de forma más gradual."))
            w["auto"] = QCheckBox(self.tr("Límites automáticos"))
            w["auto"].setToolTip(self.tr("Los límites de la zona se reparten a medio camino entre los centros. "
                                         "Desmárcalo para fijar a mano desde dónde hasta dónde llega."))
            w["auto"].toggled.connect(lambda on, n=name: self._on_zone_flag(n, "auto_range", on))
            w["min"] = self.slider(self.tr("Desde"), self._zone_setter(name, "range_min"), value=0.0, minimum=0.0,
                                   maximum=1.0, decimals=3, step=0.01)
            w["min"].setToolTip(self.tr("Valor de profundidad donde empieza la zona (solo sin límites automáticos)"))
            w["max"] = self.slider(self.tr("Hasta"), self._zone_setter(name, "range_max"), value=1.0, minimum=0.0,
                                   maximum=1.0, decimals=3, step=0.01)
            w["max"].setToolTip(self.tr("Valor de profundidad donde termina la zona (solo sin límites automáticos)"))
            keys = ["on", "center", "target", "feather", "auto", "min", "max"]
            if name.startswith("extra"):          # las añadidas se pueden quitar
                w["remove"] = QPushButton(self.tr("Quitar esta zona"))
                w["remove"].setToolTip(self.tr("Borra la zona añadida (las de fábrica no se quitan)"))
                w["remove"].clicked.connect(lambda _checked=False, n=name: self._remove_zone(n))
                keys.append("remove")
            self.zone_widgets[name] = w
            for key in keys:
                self.layout_.addWidget(w[key])
        self.layout_.addWidget(_hint(self.tr("Mueve el destino de una zona para acercarla o alejarla sin tocar "
                                             "las demás. Se ve en las vistas Efecto y 3D.")))

    # --- Cambios ----------------------------------------------------------------
    def _zone_setter(self, name: str, attr: str):
        def setter(value: float) -> None:
            setattr(self.project.depth_control.zone(name), attr, value)
            self._turn_on()
        return setter

    def _on_zone_flag(self, name: str, attr: str, on: bool) -> None:
        if self.project is None or self._loading:
            return
        setattr(self.project.depth_control.zone(name), attr, on)
        if attr == "auto_range":
            self._update_range_enabled()
        self._turn_on()
        self.emit_both()

    def _turn_on(self) -> None:
        """Si el control estaba apagado, se enciende: si no, el cambio no se vería."""
        control = self.project.depth_control
        if not control.enabled:
            control.enabled = True
            self._loading = True
            self.enabled.setChecked(True)
            self._loading = False

    def _on_enabled(self, on: bool) -> None:
        if self.project is not None and not self._loading:
            self.project.depth_control.enabled = on
            self.emit_both()

    def _reset(self) -> None:
        if self.project is None:
            return
        self.project.depth_control = DepthControl()
        self.bind(self.project)
        self.emit_both()

    def _add_zone(self) -> None:
        if self.project is None:
            return
        if self.project.depth_control.add_zone() is None:
            return
        self.bind(self.project)
        self.emit_both()

    def _remove_zone(self, name: str) -> None:
        if self.project is not None and self.project.depth_control.remove_zone(name):
            self.bind(self.project)
            self.emit_both()

    def _update_range_enabled(self) -> None:
        control = self.project.depth_control
        for name, w in self.zone_widgets.items():
            if name not in control.zone_names():
                continue
            manual = not control.zone(name).auto_range
            w["min"].setEnabled(manual)
            w["max"].setEnabled(manual)

    # --- Proyecto ---------------------------------------------------------------
    def bind(self, project: Project) -> None:
        self.project = project
        control = project.depth_control
        self._loading = True
        self.enabled.setChecked(control.enabled)
        present = control.zone_names()
        for name, w in self.zone_widgets.items():
            for widget in w.values():             # las zonas añadidas que no existen se esconden
                widget.setVisible(name in present)
            if name not in present:
                continue
            zone = control.zone(name)
            w["on"].setChecked(zone.enabled)
            w["center"].setValue(zone.center)
            w["target"].setValue(zone.target)
            w["feather"].setValue(zone.feather)
            w["auto"].setChecked(zone.auto_range)
            w["min"].setValue(zone.range_min)
            w["max"].setValue(zone.range_max)
        self._update_range_enabled()
        self.add_button.setEnabled(len(control.extra) < MAX_EXTRA)
        self._loading = False

    def refresh(self) -> None:
        """Vuelve a poner en los controles los valores del proyecto (tras «Centros desde el mapa»)."""
        if self.project is not None:
            self.bind(self.project)
