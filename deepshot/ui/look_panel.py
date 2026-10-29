"""Panel del paso «Look»: la pila de capas de ajuste (niveles, contraste, saturación, módulos…), ver deepshot/look.py.

Es una lista como la de bandas y áreas, con botones para añadir (menú de tipos), duplicar, borrar y subir / bajar: el
orden importa, cada capa recibe la imagen que dejó la anterior. Las capas se aplican de la primera de la lista a la última.
Los ajustes de cada tipo se dibujan a partir de su lista de `ParamSpec`, así que valen igual las capas de serie y las de
los módulos descargables.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QMenu, QPushButton, QStyledItemDelegate, QVBoxLayout, QWidget

from deepshot.look import KINDS, LookLayer
from deepshot.ui.editor_panels import MixPanel, _hint, _ListPanel
from deepshot.ui.icons import set_icon
from deepshot.ui.widgets import SegmentedControl


def _translate(text: str) -> str:
    """Los textos de las capas de serie se traducen; los de un módulo van tal cual (no hay traducción y sale el mismo)."""
    return QCoreApplication.translate("Look", text) if text else text


class LookPanel(_ListPanel):
    LIMITS = ["all", "depth"]
    modulesRequested = Signal()                  # «Más módulos…»: abrir el gestor de modelos y módulos

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(self.tr("+ Añadir capa"), self.tr("Añade una capa de ajuste a la pila (niveles, contraste, "
                                                           "saturación…)"), parent)
        self.list.setMaximumHeight(120)
        self.list.setItemDelegate(QStyledItemDelegate(self.list))    # sin el ojo de «solo»: aquí la casilla es el ojo
        self.solo_button.setVisible(False)
        self.list.setToolTip(self.tr("Capas de ajuste, de la primera (se aplica antes) a la última. La casilla las "
                                     "activa o desactiva."))

        # Subir y bajar: el orden de las capas cambia el resultado.
        self.up_button = QPushButton(self.tr("Subir"))
        set_icon(self.up_button, "arrow_up")
        self.up_button.setToolTip(self.tr("Sube la capa: se aplica antes (el orden cambia el resultado)"))
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button = QPushButton(self.tr("Bajar"))
        set_icon(self.down_button, "arrow_down")
        self.down_button.setToolTip(self.tr("Baja la capa: se aplica después (el orden cambia el resultado)"))
        self.down_button.clicked.connect(lambda: self._move(1))
        move_row = QHBoxLayout()
        move_row.setSpacing(4)
        move_row.addWidget(self.up_button, 1)
        move_row.addWidget(self.down_button, 1)
        self.layout_.insertLayout(2, move_row)
        self.layout_.insertWidget(0, _hint(self.tr("Las capas se aplican en orden, de la primera a la última. "
                                                   "Cambiar el orden cambia el resultado.")))

        cur = lambda: self.current()  # noqa: E731 (atajo legible para los setters)

        # Ajustes propios de cada tipo de capa: una caja por tipo, creada la primera vez que se necesita.
        self._boxes: dict[str, tuple[tuple, QWidget, dict]] = {}
        self.param_sliders: dict[str, object] = {}                   # los de la capa elegida (nombre -> deslizador)
        self.missing_label = _hint(self.tr("El módulo de esta capa no está instalado, así que no hace nada. Instálalo "
                                           "desde el Gestor de modelos y módulos (Ctrl+M): tus ajustes se conservan."))
        self.missing_label.setVisible(False)
        self.props_layout.addWidget(self.missing_label)

        # Comunes a todas las capas: opacidad, modo de fusión y límite por profundidad.
        self.common = QWidget()
        common = QVBoxLayout(self.common)
        common.setContentsMargins(0, 0, 0, 0)
        common.setSpacing(4)
        self.opacity = self.slider(self.tr("Opacidad"), lambda v: setattr(cur(), "opacity", v), value=1.0,
                                   decimals=2, step=0.05)
        self.opacity.setToolTip(self.tr("Cuánto se nota la capa: 0 = nada, 1 = por completo"))
        self.blend = QComboBox()
        for key, label in MixPanel.MODES:
            self.blend.addItem(QCoreApplication.translate("Panels", label), key)
        self.blend.setToolTip(self.tr("Cómo se combina el ajuste con la imagen que llega a la capa"))
        self.blend.activated.connect(lambda _i: self._set_layer("blend", self.blend.currentData()))
        self.limit = SegmentedControl([self.tr("Todo"), self.tr("Profundidad")])
        self.limit.setToolTip(self.tr("Todo: la capa afecta a toda la imagen. Profundidad: solo a un rango del mapa, "
                                      "p. ej. solo el fondo."))
        self.limit.changed.connect(self._on_limit)
        self.near = self.slider(self.tr("Desde"), lambda v: setattr(cur(), "depth_near", v), value=0.0, decimals=3,
                                step=0.01)
        self.near.setToolTip(self.tr("Profundidad desde la que actúa la capa (0 = lo más cercano)"))
        self.far = self.slider(self.tr("Hasta"), lambda v: setattr(cur(), "depth_far", v), value=1.0, decimals=3,
                               step=0.01)
        self.far.setToolTip(self.tr("Profundidad hasta la que actúa la capa (1 = lo más lejano)"))
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

    def item_label(self, i: int, layer: LookLayer) -> str:
        if layer.missing:
            return "{0}  ·  {1}  ({2})".format(i + 1, layer.kind.removeprefix("module."), self.tr("módulo no instalado"))
        text = "{0}  ·  {1}".format(i + 1, _translate(KINDS[layer.kind].title))
        if layer.limit == "depth":
            text += "  ·  " + self.tr("prof.")
        return text

    def _add(self) -> None:
        """Menú con los tipos de capa: primero los de serie, después los de los módulos instalados."""
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
        more.setToolTip(self.tr("Abre el Gestor de modelos y módulos para instalar más capas"))
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
            items[row], items[target] = items[target], items[row]
            self.refresh_list(select=target)
            self.emit_both()

    def _sync_solo_buttons(self) -> None:
        super()._sync_solo_buttons()
        row = self.list.currentRow()
        self.up_button.setEnabled(row > 0)
        self.down_button.setEnabled(0 <= row < len(self.items()) - 1)

    # --- Ajustes de la capa elegida -----------------------------------------
    def _kind_box(self, key: str) -> tuple[QWidget, dict]:
        """La caja con los deslizadores de un tipo de capa (se rehace si el módulo cambió sus ajustes al reinstalarlo)."""
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
