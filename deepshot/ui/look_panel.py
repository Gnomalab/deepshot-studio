"""Panel del paso «Look»: la pila de efectos (niveles, contraste, saturación, grano, nitidez, viñeta, módulos…), ver
deepshot/look.py.

Como en Photoshop: la pila se lee de arriba abajo y se aplica de abajo arriba (la de arriba va la última), los efectos nuevos
entran arriba, se reordenan arrastrándolos (o con Subir / Bajar), se renombran con doble clic o F2, cada uno tiene su casilla
(apagado), su ojo de «solo» y su opacidad y modo de fusión. Los ajustes de cada tipo se dibujan a partir de su lista de
`ParamSpec`, así que valen igual los efectos de serie y los de los módulos descargables.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import QCoreApplication, Qt, QTimer, Signal
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from deepshot.look import KINDS, LookLayer
from deepshot.ui.editor_panels import MixPanel, _ListPanel
from deepshot.ui.icons import set_icon
from deepshot.ui.widgets import SegmentedControl


def _translate(text: str) -> str:
    """Los textos de los efectos de serie se traducen; los de un módulo van tal cual (no hay traducción y sale el mismo)."""
    return QCoreApplication.translate("Look", text) if text else text


def _one_line(text: str) -> QLabel:
    """Texto de ayuda de una sola línea y sin ajuste de línea: un QLabel que se ajusta puede calcular su altura con un ancho
    antiguo (el panel aún estrecho o oculto) y dejar un hueco enorme."""
    label = QLabel(text)
    label.setWordWrap(False)
    label.setProperty("dim", True)
    return label


class LookPanel(_ListPanel):
    LIMITS = ["all", "depth"]
    modulesRequested = Signal()                  # «Más módulos…»: abrir el gestor de modelos y módulos

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self.tr("+ Añadir efecto"), self.tr("Añade un efecto a la pila (niveles, contraste, "
                                                             "saturación, grano…). Entra arriba."), parent)
        self.list.setMaximumHeight(150)
        self.list.setToolTip(self.tr("Efectos de arriba abajo; se aplican de abajo arriba, como en Photoshop. Arrastra "
                                     "para reordenar, doble clic o F2 para renombrar. La casilla apaga el efecto y el "
                                     "ojo lo deja solo."))
        # Reordenar arrastrando: la lista mueve la fila y el proyecto sigue su nuevo orden.
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.model().rowsMoved.connect(self._on_rows_moved)
        self._reordering = False

        # Subir y bajar, además de arrastrar: el orden de los efectos cambia el resultado.
        self.up_button = QPushButton(self.tr("Subir"))
        set_icon(self.up_button, "arrow_up")
        self.up_button.setToolTip(self.tr("Sube el efecto: se aplica después de los de debajo (el orden cambia el "
                                          "resultado)"))
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button = QPushButton(self.tr("Bajar"))
        set_icon(self.down_button, "arrow_down")
        self.down_button.setToolTip(self.tr("Baja el efecto: se aplica antes (el orden cambia el resultado)"))
        self.down_button.clicked.connect(lambda: self._move(1))
        move_row = QHBoxLayout()
        move_row.setSpacing(4)
        move_row.addWidget(self.up_button, 1)
        move_row.addWidget(self.down_button, 1)
        self.layout_.insertLayout(2, move_row)
        self.layout_.insertWidget(0, _one_line(self.tr("Se aplican de abajo arriba, como en Photoshop.")))

        cur = lambda: self.current()  # noqa: E731 (atajo legible para los setters)

        # Ajustes propios de cada tipo de efecto: una caja por tipo, creada la primera vez que se necesita.
        self._boxes: dict[str, tuple[tuple, QWidget, dict]] = {}
        self.param_sliders: dict[str, object] = {}                   # los del efecto elegido (nombre -> deslizador)
        self.missing_label = _one_line(self.tr("Módulo no instalado: no hace nada. Instálalo en el Gestor (Ctrl+M)."))
        self.missing_label.setVisible(False)
        self.props_layout.addWidget(self.missing_label)

        # Comunes a todos los efectos: opacidad, modo de fusión y límite por profundidad.
        self.common = QWidget()
        common = QVBoxLayout(self.common)
        common.setContentsMargins(0, 0, 0, 0)
        common.setSpacing(4)
        self.opacity = self.slider(self.tr("Opacidad"), lambda v: setattr(cur(), "opacity", v), value=1.0,
                                   decimals=2, step=0.05)
        self.opacity.setToolTip(self.tr("Cuánto se nota el efecto: 0 = nada, 1 = por completo"))
        self.blend = QComboBox()
        for key, label in MixPanel.MODES:
            self.blend.addItem(QCoreApplication.translate("Panels", label), key)
        self.blend.setToolTip(self.tr("Cómo se combina el ajuste con la imagen que llega al efecto"))
        self.blend.activated.connect(lambda _i: self._set_layer("blend", self.blend.currentData()))
        self.limit = SegmentedControl([self.tr("Todo"), self.tr("Profundidad")])
        self.limit.setToolTip(self.tr("Todo: el efecto afecta a toda la imagen. Profundidad: solo a un rango del mapa, "
                                      "p. ej. solo el fondo."))
        self.limit.changed.connect(self._on_limit)
        self.near = self.slider(self.tr("Desde"), lambda v: setattr(cur(), "depth_near", v), value=0.0, decimals=3,
                                step=0.01)
        self.near.setToolTip(self.tr("Profundidad desde la que actúa el efecto (0 = lo más cercano)"))
        self.far = self.slider(self.tr("Hasta"), lambda v: setattr(cur(), "depth_far", v), value=1.0, decimals=3,
                               step=0.01)
        self.far.setToolTip(self.tr("Profundidad hasta la que actúa el efecto (1 = lo más lejano)"))
        self.soft = self.slider(self.tr("Suavidad"), lambda v: setattr(cur(), "depth_softness", v), value=0.1,
                                maximum=0.5, decimals=3, step=0.01)
        self.soft.setToolTip(self.tr("Cuánto se difumina el paso entre lo que se ajusta y lo que no"))
        self.range_box = QWidget()
        rl = QVBoxLayout(self.range_box)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        for w in (self.near, self.far, self.soft):
            rl.addWidget(w)
        for w in (self.opacity, self.blend, self.limit, self.range_box):
            common.addWidget(w)
        self.props_layout.addWidget(self.common)

    # --- Lista -------------------------------------------------------------
    def items(self) -> list[LookLayer]:
        return self.project.look if self.project else []

    def new_item(self) -> LookLayer:
        return LookLayer.new("levels")

    @staticmethod
    def kind_title(layer: LookLayer) -> str:
        """El nombre del tipo de efecto (el de serie traducido; el id para un módulo que no está instalado)."""
        if layer.missing:
            return layer.kind.removeprefix("module.")
        return _translate(KINDS[layer.kind].title)

    def item_label(self, i: int, layer: LookLayer) -> str:
        name = layer.name.strip() or self.kind_title(layer)
        return "{0}  ({1})".format(name, self.tr("módulo no instalado")) if layer.missing else name

    def item_icon(self, i: int, layer: LookLayer):
        return None

    def _row_tip(self, layer: LookLayer) -> str:
        where = self.tr("solo en un rango de profundidad") if layer.limit == "depth" else self.tr("toda la imagen")
        return "{0} · {1}".format(self.kind_title(layer), where)

    def refresh_list(self, select: int | None = None) -> None:
        super().refresh_list(select)
        self._loading = True                         # poner la bandera de «editable» también avisa de un cambio
        for row, layer in enumerate(self.items()):
            entry = self.list.item(row)
            entry.setFlags(entry.flags() | Qt.ItemIsEditable)            # doble clic o F2: renombrar
            entry.setToolTip(self._row_tip(layer))
        self._loading = False

    def refresh_current_label(self) -> None:
        super().refresh_current_label()
        row = self.list.currentRow()
        if 0 <= row < len(self.items()) and self.list.item(row) is not None:
            self._loading = True
            self.list.item(row).setToolTip(self._row_tip(self.items()[row]))
            self._loading = False

    def _on_item_changed(self, entry) -> None:
        """La casilla (apagar / encender) y el renombrado llegan por la misma señal de la lista."""
        if self._loading or self.project is None:
            return
        row = self.list.row(entry)
        layer = self.items()[row]
        changed = False
        enabled = entry.checkState() == Qt.Checked
        if enabled != layer.enabled:
            layer.enabled, changed = enabled, True
        text = entry.text().strip()
        if text != self.item_label(row, layer):                          # lo escribió el usuario
            layer.name = "" if (not text or text == self.kind_title(layer)) else text
            changed = True
            self._loading = True
            entry.setText(self.item_label(row, layer))                   # el nombre vacío vuelve al del efecto
            self._loading = False
        if changed:
            self._sync_solo_buttons()
            self.emit_both()

    def _shift_solo(self, row: int) -> None:
        """Se mete una fila nueva en `row`: los «solo» de las de debajo bajan un puesto y la nueva se ve también."""
        if self.solo:
            self.solo = {r + 1 if r >= row else r for r in self.solo} | {row}

    def add_item(self, item, commit: bool = True) -> None:
        """Un efecto nuevo entra arriba de la pila (como una capa nueva en Photoshop)."""
        if self.project is None:
            return
        self.items().insert(0, item)
        self._shift_solo(0)
        self.refresh_list(select=0)
        if commit:
            self.emit_both()

    def _duplicate(self) -> None:
        row = self.list.currentRow()
        if self.project is None or not 0 <= row < len(self.items()):
            return
        clone = copy.deepcopy(self.items()[row])
        if clone.name:
            clone.name += " " + self.tr("copia")
        self.items().insert(row, clone)                                  # justo encima del original
        self._shift_solo(row)
        self.refresh_list(select=row)
        self.emit_both()

    def _add(self) -> None:
        """Menú con los tipos de efecto: primero los de serie, después los de los módulos instalados."""
        if self.project is None:
            return
        menu = QMenu(self)
        built_in = [k for k in KINDS.values() if not k.module_id]
        extra = [k for k in KINDS.values() if k.module_id]
        for kind in built_in:
            menu.addAction(_translate(kind.title)).setData(kind.key)
        if extra:
            menu.addSeparator()
            for kind in extra:
                menu.addAction(kind.title).setData(kind.key)
        menu.addSeparator()
        more = menu.addAction(self.tr("Más módulos…"))
        more.setToolTip(self.tr("Abre el Gestor de modelos y módulos para instalar más efectos"))
        chosen = menu.exec(self.mapToGlobal(self.list.geometry().topLeft()))
        if chosen is more:
            self.modulesRequested.emit()
        elif chosen is not None:
            self.add_item(LookLayer.new(chosen.data()))

    def _move(self, step: int) -> None:
        row = self.list.currentRow()
        items = self.items()
        target = row + step
        if 0 <= row < len(items) and 0 <= target < len(items):
            items.insert(target, items.pop(row))
            self.solo.clear()                                            # las filas se corren: se quita el «solo»
            self.refresh_list(select=target)
            self.emit_both()

    def _on_rows_moved(self, _parent, start: int, end: int, _dest_parent, dest: int) -> None:
        """Se arrastró una fila: el proyecto sigue el orden nuevo (la lista ya lo tiene)."""
        items = self.items()
        if self._reordering or self._loading or start != end or not 0 <= start < len(items):
            return
        target = dest - 1 if dest > start else dest
        target = min(max(target, 0), len(items) - 1)
        if target == start:
            return
        self._reordering = True
        items.insert(target, items.pop(start))
        self.solo.clear()
        # La lista se rehace fuera del arrastre (aún no ha terminado de soltarse).
        QTimer.singleShot(0, lambda: self._after_drag(target))

    def _after_drag(self, target: int) -> None:
        self.refresh_list(select=target)
        self._reordering = False
        self.emit_both()

    def _sync_solo_buttons(self) -> None:
        super()._sync_solo_buttons()
        row = self.list.currentRow()
        self.up_button.setEnabled(row > 0)
        self.down_button.setEnabled(0 <= row < len(self.items()) - 1)

    # --- Ajustes del efecto elegido ------------------------------------------
    def _kind_box(self, key: str) -> tuple[QWidget, dict]:
        """La caja con los deslizadores de un tipo de efecto (se rehace si el módulo cambió sus ajustes al reinstalarlo)."""
        kind = KINDS[key]
        signature = tuple(kind.params)
        cached = self._boxes.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        if cached is not None:
            cached[1].setParent(None)
            cached[1].deleteLater()
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        sliders = {}
        for spec in kind.params:
            def setter(value, name=spec.name):
                self.current().params[name] = value

            label = _translate(spec.label)
            slider = self.slider(label, setter, value=spec.default, minimum=spec.minimum, maximum=spec.maximum,
                                 decimals=spec.decimals, step=spec.step, drag_range=spec.drag_range)
            slider.setToolTip(_translate(spec.tooltip) or label)
            sliders[spec.name] = slider
            layout.addWidget(slider)
        self.props_layout.insertWidget(0, box)
        self._boxes[key] = (signature, box, sliders)
        return box, sliders

    def show_props(self, layer: LookLayer) -> None:
        for _sig, box, _s in self._boxes.values():
            box.setVisible(False)
        self.param_sliders = {}
        self.missing_label.setVisible(layer.missing)
        self.common.setVisible(not layer.missing)
        if layer.missing:
            return
        box, sliders = self._kind_box(layer.kind)
        box.setVisible(True)
        self.param_sliders = sliders
        for name, slider in sliders.items():
            slider.setValue(layer.params.get(name, KINDS[layer.kind].defaults[name]))
        self.opacity.setValue(layer.opacity)
        self.blend.setCurrentIndex(max(0, self.blend.findData(layer.blend)))
        self.limit.set_current_index(self.LIMITS.index(layer.limit) if layer.limit in self.LIMITS else 0)
        self.near.setValue(layer.depth_near)
        self.far.setValue(layer.depth_far)
        self.soft.setValue(layer.depth_softness)
        self.range_box.setVisible(layer.limit == "depth")

    def _set_layer(self, name: str, value) -> None:
        layer = self.current()
        if layer is not None and not self._loading:
            setattr(layer, name, value)
            self.emit_both()

    def _on_limit(self, index: int) -> None:
        layer = self.current()
        if layer is not None and not self._loading:
            layer.limit = self.LIMITS[index]
            self.range_box.setVisible(layer.limit == "depth")
            self.refresh_current_label()
            self.emit_both()
