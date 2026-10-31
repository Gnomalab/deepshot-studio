"""Panel del paso «Look»: la pila de efectos (niveles, contraste, saturación, grano, nitidez, viñeta, módulos…), ver
deepshot/look.py.

Como en Photoshop: la pila se lee de arriba abajo y se aplica de abajo arriba (la de arriba va la última), los efectos nuevos
entran arriba, se reordenan arrastrándolos (o con Subir / Bajar), se renombran con doble clic o F2, cada uno tiene su ojo
(bypass: se aplica o se salta) y su opacidad y modo de fusión. Los ajustes de cada tipo se dibujan a partir de su lista de
`ParamSpec`, así que valen igual los efectos de serie y los de los módulos descargables.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import QCoreApplication, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from deepshot import maps, presets
from deepshot.look import KINDS, LookLayer
from deepshot.ui.editor_panels import DETAIL_ROLE, MixPanel, _ListPanel
from deepshot.ui.icons import set_icon
from deepshot.ui.preset_bar import PresetBar
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
    """Una pila de efectos. `stack` = "look" (Post FX: sobre el resultado) o "pre" (Pre FX: sobre la foto, antes de
    mezclarla con el efecto): el mismo código vale para las dos, solo cambia la lista del proyecto que edita."""

    LIMITS = ["all", "depth", "mask"]
    modulesRequested = Signal()                  # «Más módulos…»: abrir el gestor de modelos y módulos
    mapRequested = Signal(str)                   # «Calcular»: hay que calcular este mapa auxiliar (id de deepshot/maps.py)

    def __init__(self, stack: str = "look", parent: QWidget | None = None) -> None:
        super().__init__(self.tr("+ Añadir efecto"), self.tr("Añade un efecto a la pila (niveles, contraste, "
                                                             "saturación, grano…). Entra arriba."), parent)
        self.stack = "pre" if stack == "pre" else "look"
        self.map_state: dict[str, str] = {}      # estado de cada mapa de la foto: "ready" | "busy" | "missing" | "nomodel"
        self.map_text = ""                       # mensaje del cálculo en curso (progreso) o del último fallo
        self.list.setMaximumHeight(150)
        self.list.setToolTip(self.tr("Efectos de arriba abajo; se aplican de abajo arriba, como en Photoshop. Arrastra "
                                     "para reordenar, doble clic o F2 para renombrar. El ojo aplica o salta el efecto."))
        if self.stack == "pre":
            hint = self.tr("Retocan la foto antes de mezclarla con el efecto.")
        else:
            hint = self.tr("Se aplican de abajo arriba, como en Photoshop.")
        self.layout_.insertWidget(0, _one_line(hint))
        # Looks guardados: la pila entera. Los de fábrica usan solo efectos de serie, así valen en cualquier instalación.
        self.preset_bar = PresetBar("look", self.tr("Cargar look…"),
                                    lambda: presets.from_project("look", self.project, stack=self.stack))
        self.preset_bar.combo.setToolTip(self.tr("Looks: una pila de efectos entera (sustituye a la actual; se puede "
                                                 "deshacer). Los de fábrica usan solo efectos de serie. Si guardas uno con "
                                                 "efectos de un módulo, hará falta ese módulo para verlo."))
        self.layout_.insertWidget(1, self.preset_bar)

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
        self.limit = SegmentedControl([self.tr("Todo"), self.tr("Profundidad"), self.tr("Mapa")])
        self.limit.setToolTip(self.tr("Todo: el efecto afecta a toda la imagen. Profundidad: solo a un rango del mapa "
                                      "de profundidad, p. ej. solo el fondo. Mapa: solo a un mapa auxiliar, p. ej. el "
                                      "sujeto."))
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
        # Límite por mapa auxiliar (deepshot/maps.py): qué mapa, si se invierte y el botón que lo calcula.
        self.map_combo = QComboBox()
        for key in maps.KINDS:
            self.map_combo.addItem(maps.title(key), key)
        self.map_combo.setToolTip(self.tr("El mapa auxiliar al que se limita el efecto (se calcula una vez por foto)"))
        self.map_combo.activated.connect(lambda _i: self._on_map_chosen())
        self.map_invert = QCheckBox(self.tr("Invertir"))
        self.map_invert.setToolTip(self.tr("Invierte el mapa: el efecto actúa en todo menos en él (p. ej. solo el fondo)"))
        self.map_invert.toggled.connect(lambda on: self._set_layer("mask_invert", bool(on)))
        self.map_button = QPushButton()
        self.map_button.clicked.connect(self._on_map_button)
        self.map_status = _one_line("")
        self.map_box = QWidget()
        ml = QVBoxLayout(self.map_box)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self.map_combo, 1)
        row.addWidget(self.map_invert)
        ml.addLayout(row)
        ml.addWidget(self.map_button)
        ml.addWidget(self.map_status)
        for w in (self.opacity, self.blend, self.limit, self.range_box, self.map_box):
            common.addWidget(w)
        self.props_layout.addWidget(self.common)

    # --- Lista -------------------------------------------------------------
    def items(self) -> list[LookLayer]:
        return getattr(self.project, self.stack) if self.project else []

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

    def item_detail(self, i: int, layer: LookLayer) -> str:
        """Con un nombre propio, el tipo de efecto queda a la derecha, tenue."""
        return self.kind_title(layer) if layer.name.strip() else ""

    def _row_tip(self, layer: LookLayer) -> str:
        if layer.limit == "depth":
            where = self.tr("solo en un rango de profundidad")
        elif layer.limit == "mask":
            where = self.tr("solo en un mapa: {0}").format(maps.title(layer.mask_id))
        else:
            where = self.tr("toda la imagen")
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
        text = entry.text().strip()
        if text != self.item_label(row, layer):                          # lo escribió el usuario
            layer.name = "" if (not text or text == self.kind_title(layer)) else text
            changed = True
            self._loading = True
            entry.setText(self.item_label(row, layer))                   # el nombre vacío vuelve al del efecto
            entry.setData(DETAIL_ROLE, self.item_detail(row, layer))
            self._loading = False
        if changed:
            self.emit_both()

    def _shift_solo(self, row: int) -> None:
        """Se mete una fila nueva en `row`: con un «solo» activo, el solo pasa a la nueva."""
        if self.solo:
            self.solo = {row}

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
        self.map_combo.setCurrentIndex(max(0, self.map_combo.findData(layer.mask_id)))
        self.map_invert.setChecked(layer.mask_invert)
        self.map_box.setVisible(layer.limit == "mask")
        self._sync_map_ui()

    def _set_layer(self, name: str, value) -> None:
        layer = self.current()
        if layer is not None and not self._loading:
            setattr(layer, name, value)
            self.emit_both()

    def _on_limit(self, index: int) -> None:
        layer = self.current()
        if layer is not None and not self._loading:
            layer.limit = self.LIMITS[index]
            if layer.limit == "mask" and layer.mask_id not in maps.KINDS:
                layer.mask_id = self.map_combo.currentData() or next(iter(maps.KINDS))     # el primero (p. ej. «Sujeto»)
            self.range_box.setVisible(layer.limit == "depth")
            self.map_box.setVisible(layer.limit == "mask")
            self._sync_map_ui()
            self.refresh_current_label()
            self.emit_both()

    # --- Mapas auxiliares ------------------------------------------------------
    def set_map_state(self, state: dict[str, str], text: str = "") -> None:
        """El editor avisa del estado de los mapas de la foto actual (calculado, en curso, sin calcular, sin modelo)."""
        self.map_state = dict(state)
        self.map_text = text
        self._sync_map_ui()

    def _chosen_map(self) -> str:
        return self.map_combo.currentData() or ""

    def _on_map_chosen(self) -> None:
        layer = self.current()
        if layer is not None and not self._loading:
            layer.mask_id = self._chosen_map()
            self.refresh_current_label()
            self.emit_both()
            self._sync_map_ui()

    def _on_map_button(self) -> None:
        if self._chosen_map():
            self.mapRequested.emit(self._chosen_map())

    def _sync_map_ui(self) -> None:
        """Botón y texto del mapa elegido según su estado. Nunca se calcula solo: se pide con el botón."""
        key = self._chosen_map()
        state = self.map_state.get(key, "missing")
        name = maps.title(key) if key else ""
        if state == "ready":
            self.map_button.setText(self.tr("Recalcular {0}").format(name))
            self.map_status.setText(self.tr("Mapa listo."))
        elif state == "busy":
            self.map_button.setText(self.tr("Calculando…"))
            self.map_status.setText(self.map_text or self.tr("Calculando…"))
        elif state == "nomodel":
            self.map_button.setText(self.tr("Calcular {0}").format(name))
            self.map_status.setText(self.map_text or self.tr("Falta un modelo: instálalo en el Gestor (Ctrl+M)."))
        else:
            self.map_button.setText(self.tr("Calcular {0}").format(name))
            self.map_status.setText(self.map_text or self.tr("Sin calcular: el efecto no actúa hasta entonces."))
        self.map_button.setEnabled(state != "busy" and bool(key) and self.project is not None)
        self.map_button.setToolTip(self.tr("Calcula el mapa de esta foto en segundo plano (la primera vez tarda unos "
                                           "segundos; queda guardado)."))
        self.map_status.setToolTip(self.map_status.text())


class LookStack(QWidget):
    """El paso Look entero: un conmutador «Pre FX | Post FX» y, debajo, la pila elegida (dos `LookPanel`).

    Pre FX retoca la foto antes de mezclarla con el efecto; Post FX, el resultado. Cada pila es un `LookPanel` con su
    lista, sus presets y su «solo»; esta clase solo cambia de una a otra."""

    def __init__(self, pre: LookPanel, post: LookPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pre, self.post = pre, post
        self.switch = SegmentedControl([self.tr("Pre FX"), self.tr("Post FX")], current=1)
        self.switch.set_tooltips([
            self.tr("Pre FX: efectos sobre la foto ANTES de mezclarla con el efecto de profundidad (solo se ven si la "
                    "mezcla con la foto está activa)."),
            self.tr("Post FX: efectos sobre el resultado final, después de la mezcla."),
        ])
        self.pages = QStackedWidget()
        self.pages.addWidget(pre)
        self.pages.addWidget(post)
        self.pages.setCurrentIndex(1)
        self.switch.changed.connect(self.pages.setCurrentIndex)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.switch)
        layout.addWidget(self.pages)

    def current_panel(self) -> LookPanel:
        return self.pages.currentWidget()

    def show_stack(self, stack: str) -> None:
        """Pasa a la pila "pre" o "look"."""
        index = 0 if stack == "pre" else 1
        self.switch.set_current_index(index)
        self.pages.setCurrentIndex(index)
