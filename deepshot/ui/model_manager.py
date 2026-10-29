"""Diálogo "Gestor de modelos y módulos": descargar, instalar desde carpeta, elegir el activo y borrar.

Los módulos (capas de Look…, deepshot/modules.py) comparten la lista con los modelos de profundidad: llevan su tipo y
la carpeta donde se instalan, y se instalan o desinstalan con los mismos botones."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QGridLayout,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepshot.ui import theme
from deepshot.depth import DepthEstimator
from deepshot.paths import resource
from deepshot.ui.model_cards import CardData, ModelCard
from deepshot.ui.task_timer import TaskTimer, with_time
from deepshot.ui.widgets import SegmentedControl
from deepshot import modules
from deepshot.models import (ALL_FAMILIES, FAMILIES, ModelEntry, ModelError, ModelManager, SearchResult, fits_vram,
                            can_compute, is_permissive)

log = logging.getLogger(__name__)


class _Bridge(QObject):
    """Lleva los avisos del hilo de trabajo al hilo de la interfaz."""

    progress = Signal(str, float)  # fracción -1 = indeterminada
    done = Signal(str)             # mensaje de éxito
    failed = Signal(str)           # mensaje de error para el usuario


class ModelManagerDialog(QDialog):
    """Lista de modelos con sus acciones. Emite `active_changed` y `models_changed`."""

    active_changed = Signal(str)
    models_changed = Signal()

    COL_NAME, COL_LINK, COL_FAMILY, COL_SOURCE, COL_LICENSE, COL_STATE, COL_PRELOAD, COL_KIND, COL_DEST = range(9)
    KIND_FILTERS = ["all", "depth", "look"]           # «Mostrar»: todo, solo modelos de profundidad, solo módulos de Look
    CARD_WIDTH = 230                      # ancho aproximado de una tarjeta, para repartirlas en columnas

    def __init__(self, manager: ModelManager, estimator: DepthEstimator, active_id: str,
                 is_depth_busy: Callable[[], bool], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Gestor de modelos y módulos"))
        self.setMinimumSize(820, 420)
        self.manager = manager
        self.estimator = estimator
        self.active_id = active_id
        self.is_depth_busy = is_depth_busy
        self._busy = False
        self._current_id: str | None = None          # el modelo seleccionado (en la lista o en las tarjetas)
        self._updates: set[str] = set()              # modelos con una versión más nueva (tras «Comprobar actualizaciones»)
        self._cards: dict[str, ModelCard] = {}

        self._bridge = _Bridge(self)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.done.connect(self._on_done)
        self._bridge.failed.connect(self._on_failed)

        # --- Tabla de modelos ---
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([self.tr("Modelo"), self.tr("Ficha"), self.tr("Familia"), self.tr("Origen"),
                                   self.tr("Licencia"), self.tr("Estado"), self.tr("Al arrancar"), self.tr("Tipo"),
                                   self.tr("Se instala en")])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setToolTip(self.tr("La casilla de cada modelo decide si sale en el selector de modelos del panel de "
                                     "profundidad. Desmarcar un modelo no lo borra ni lo descarga: solo lo oculta."))
        self._refreshing = False
        self.tree.itemChanged.connect(self._on_item_checked)
        # Columnas: se pueden arrastrar para reordenarlas, cambiar de ancho y ocultar (clic derecho en la cabecera);
        # el estado se recuerda. La de la ficha es de ancho fijo (solo lleva el icono).
        header = self.tree.header()
        header.setSectionsMovable(True)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(40)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._header_menu)
        for col, width in ((self.COL_NAME, 230), (self.COL_FAMILY, 200), (self.COL_SOURCE, 70), (self.COL_LICENSE, 100),
                           (self.COL_STATE, 170), (self.COL_PRELOAD, 90), (self.COL_KIND, 150), (self.COL_DEST, 260)):
            header.setSectionResizeMode(col, QHeaderView.Interactive)
            header.resizeSection(col, width)
        header.setSectionResizeMode(self.COL_LINK, QHeaderView.Fixed)
        header.resizeSection(self.COL_LINK, 46)
        saved_header = QSettings().value("models/header")
        self._header_restored = bool(saved_header) and header.restoreState(saved_header)
        if not self._header_restored:                     # «Tipo» junto al nombre y «Se instala en» antes del estado
            header.moveSection(header.visualIndex(self.COL_KIND), 2)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._link_icon = QIcon(str(resource("icons", "external-link.svg")))
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemDoubleClicked.connect(lambda *_: self.activate_selected())

        # Dos vistas del mismo contenido: lista compacta y tarjetas con más información.
        self.cards_host = QWidget()
        self.cards_grid = QGridLayout(self.cards_host)
        self.cards_grid.setContentsMargins(4, 4, 4, 4)
        self.cards_grid.setSpacing(8)
        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setWidget(self.cards_host)
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.tree)
        self.view_stack.addWidget(self.cards_scroll)
        self.view_switch = SegmentedControl([self.tr("Lista"), self.tr("Tarjetas")])
        self.view_switch.set_tooltips([self.tr("Los modelos en una lista compacta"),
                                      self.tr("Los modelos en tarjetas, con más información y sus botones")])
        self.view_switch.changed.connect(self._on_view_changed)
        self.btn_check_updates = QPushButton(self.tr("Comprobar actualizaciones"))
        self.btn_check_updates.setToolTip(self.tr("Mira en Hugging Face si hay versiones más nuevas de los modelos "
                                                  "instalados. Solo se conecta cuando lo pulsas."))
        self.btn_check_updates.clicked.connect(self.check_updates)
        # Filtro por licencia: por defecto solo se ven los de licencia que permite uso comercial.
        self.btn_licenses = QPushButton()
        self.btn_licenses.clicked.connect(self.toggle_more_licenses)
        # Filtro por tipo: modelos de profundidad, módulos de Look o todo.
        self.kind_filter = SegmentedControl([self.tr("Todo"), self.tr("Modelos"), self.tr("Módulos de Look")])
        self.kind_filter.set_tooltips([self.tr("Modelos de profundidad y módulos"),
                                       self.tr("Solo los modelos de profundidad (Marigold, Depth Anything…)"),
                                       self.tr("Solo los módulos de Look: capas de ajuste extra para el paso Look")])
        saved_kind = str(QSettings().value("models/kind_filter", "all"))
        self.kind_filter.set_current_index(self.KIND_FILTERS.index(saved_kind) if saved_kind in self.KIND_FILTERS else 0)
        self.kind_filter.changed.connect(self._on_kind_filter)

        # --- Acciones ---
        self.btn_activate = QPushButton(self.tr("Usar este modelo"))
        self.btn_activate.setObjectName("Primary")
        self.btn_activate.clicked.connect(self.activate_selected)
        self.btn_download = QPushButton(self.tr("Instalar"))
        self.btn_download.setToolTip(self.tr("Descarga el modelo a la carpeta de modelos"))
        self.btn_download.clicked.connect(self.download_selected)
        self.btn_update = QPushButton(self.tr("Actualizar"))
        self.btn_update.setToolTip(self.tr("Descarga la versión más nueva del modelo seleccionado y borra la anterior "
                                           "al terminar bien. Primero pulsa «Comprobar actualizaciones»."))
        self.btn_update.clicked.connect(self.update_selected)
        self.btn_delete = QPushButton(self.tr("Desinstalar"))
        self.btn_delete.setToolTip(self.tr("Borra los archivos del modelo de la carpeta de modelos"))
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_add_hf = QPushButton(self.tr("Añadir desde Hugging Face…"))
        self.btn_add_hf.setToolTip(self.tr("Lista de modelos compatibles publicados en Hugging Face"))
        self.btn_add_hf.clicked.connect(self.open_search)
        self.btn_add_folder = QPushButton(self.tr("Instalar desde carpeta…"))
        self.btn_add_folder.clicked.connect(self.install_from_folder)
        # La carpeta de modelos: dónde están, abrirla, cambiarla y volver a leerla (para los modelos soltados a mano).
        self.btn_open_folder = QPushButton(self.tr("Abrir carpeta de modelos"))
        self.btn_open_folder.setToolTip(self.tr("Abre la carpeta donde se guardan los modelos. Para añadir uno a mano, "
                                                "copia su carpeta aquí (con config.json o model_index.json y archivos "
                                                ".safetensors) y pulsa «Actualizar lista»."))
        self.btn_open_folder.clicked.connect(self.open_models_folder)
        self.btn_change_folder = QPushButton(self.tr("Cambiar carpeta…"))
        self.btn_change_folder.setToolTip(self.tr("Elige otra carpeta para los modelos. Los ya descargados no se mueven: "
                                                  "cópialos tú a la nueva carpeta y pulsa «Actualizar lista»."))
        self.btn_change_folder.clicked.connect(self.change_models_folder)
        self.btn_refresh = QPushButton(self.tr("Actualizar lista"))
        self.btn_refresh.setToolTip(self.tr("Vuelve a leer la carpeta de modelos: aparecen los modelos que hayas "
                                            "copiado a mano"))
        self.btn_refresh.clicked.connect(self.refresh_from_disk)

        self.btn_adopt = QPushButton(self.tr("Mover modelos de la carpeta anterior"))
        self.btn_adopt.setToolTip(self.tr("Hay modelos en la carpeta antigua de datos de la aplicación. Los mueve a la "
                                          "carpeta de modelos actual; no pisa nada que ya esté."))
        self.btn_adopt.clicked.connect(self.adopt_legacy)

        actions = QVBoxLayout()
        for b in (self.btn_download, self.btn_activate, self.btn_update, self.btn_delete):   # el mismo orden que en las tarjetas
            actions.addWidget(b)
        actions.addSpacing(16)
        actions.addWidget(self.btn_add_hf)
        actions.addWidget(self.btn_add_folder)
        actions.addSpacing(16)
        for b in (self.btn_refresh, self.btn_open_folder, self.btn_change_folder):
            actions.addWidget(b)
        actions.addWidget(self.btn_adopt)
        actions.addStretch()

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel(self.tr("Ver:")))
        view_row.addWidget(self.view_switch)
        view_row.addSpacing(12)
        view_row.addWidget(QLabel(self.tr("Mostrar:")))
        view_row.addWidget(self.kind_filter)
        view_row.addStretch()
        view_row.addWidget(self.btn_licenses)
        view_row.addWidget(self.btn_check_updates)
        left = QVBoxLayout()
        left.addLayout(view_row)
        left.addWidget(self.view_stack, 1)
        top = QHBoxLayout()
        top.addLayout(left, 1)
        top.addLayout(actions)

        # --- Pie: progreso, aviso de seguridad y cerrar ---
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        self.status = QLabel(self.tr("Solo se aceptan pesos .safetensors: los formatos "
                                     ".bin, .pt y .ckpt pueden ejecutar código al cargarse."))
        self.status.setWordWrap(True)
        self.status.setProperty("dim", True)
        self._timer = TaskTimer(self)                 # tiempo transcurrido y estimado de descargas y copias
        self._message = ""
        self._timer.changed.connect(lambda _t: self.status.setText(with_time(self._message, self._timer)))
        close = QPushButton(self.tr("Cerrar"))
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(close)

        self.folder_label = QLabel()
        self.folder_label.setProperty("dim", True)
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._update_folder_label()

        layout = QVBoxLayout(self)
        layout.addLayout(top, 1)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.progress)
        layout.addLayout(bottom)

        saved_view = str(QSettings().value("models/view", "list"))
        self.view_switch.set_current_index(1 if saved_view == "cards" else 0)
        self.view_stack.setCurrentIndex(1 if saved_view == "cards" else 0)
        self.refresh()

    # =====================================================================
    # Tabla
    # =====================================================================
    def _source_text(self, entry: ModelEntry) -> str:
        return {"catalog": self.tr("Catálogo"), "huggingface": "Hugging Face", "module": self.tr("Catálogo de módulos"),
                "local": self.tr("Carpeta local"),
                "folder": self.tr("Carpeta de modelos")}.get(entry.source, entry.source)

    # --- Tipo y destino (modelos y módulos) ------------------------------------------------------------------------------
    def kind_text(self, entry: ModelEntry) -> str:
        return {"look": self.tr("Módulo de Look")}.get(entry.kind, self.tr("Modelo de profundidad"))

    def dest_path(self, entry: ModelEntry) -> Path:
        """La carpeta donde se instala la entrada: la de módulos para un módulo, la de modelos para los demás."""
        return modules.modules_dir() / (entry.path or "") if entry.is_module else self.manager.models_dir

    def _on_kind_filter(self, index: int) -> None:
        QSettings().setValue("models/kind_filter", self.KIND_FILTERS[index])
        self.refresh()

    def kind_visible(self, entry: ModelEntry) -> bool:
        wanted = self.KIND_FILTERS[self.kind_filter.current_index()]
        return wanted == "all" or (wanted == "depth" and not entry.is_module) or entry.kind == wanted

    def _state_text(self, entry: ModelEntry) -> str:
        if entry.is_module:
            if self.manager.is_installed(entry.id):
                return self.tr("Instalado · v{0}").format(entry.version) if entry.version else self.tr("Instalado")
            return self.tr("No instalado")
        if self.manager.is_installed(entry.id):
            gb = self.manager.disk_size(entry.id) / 1e9
            return self.tr("Instalado · {0:.1f} GB").format(gb)
        if entry.approx_gb:
            return self.tr("No descargado · {0:.1f} GB").format(entry.approx_gb)
        return self.tr("No descargado")

    def _gpu_vram_gb(self) -> float | None:
        """VRAM de la GPU de este equipo (se mide una vez; None si no hay GPU con VRAM medible)."""
        if not hasattr(self, "_vram_gb"):
            try:
                from deepshot.device import detect_device

                info = detect_device("auto")
                self._vram_gb, self._gpu_name = info.vram_gb, info.name
            except Exception:                      # un driver raro no debe impedir abrir el gestor
                self._vram_gb, self._gpu_name = None, ""
        return self._vram_gb

    def misfit_reason(self, entry: ModelEntry) -> str:
        """Texto «No cabe en este equipo…» si el modelo pide más VRAM de la que hay; vacío si cabe o no se sabe."""
        vram = self._gpu_vram_gb()
        if fits_vram(entry, vram) is not False:
            return ""
        return self.tr("No cabe en este equipo: necesita {0:.0f} GB de VRAM y tienes {1:.0f} GB.").format(
            entry.min_vram_gb, vram)

    def no_compute_reason(self, entry: ModelEntry) -> str:
        """Aviso si la app todavía no sabe calcular con este modelo (vacío si sí)."""
        if can_compute(entry):
            return ""
        return self.tr("Esta versión de la aplicación todavía no puede calcular la profundidad con «{0}»: se puede "
                       "descargar, pero no usar. No hace falta que lo descargues ahora.").format(entry.display_label)

    def _on_item_checked(self, item, column: int) -> None:
        """Marca o desmarca un modelo: sale o no en el selector del panel. El modelo activo no se puede ocultar."""
        if column not in (self.COL_NAME, self.COL_PRELOAD) or self._refreshing:
            return
        model_id = item.data(0, Qt.UserRole)
        if column == self.COL_PRELOAD:
            self._set_preload(model_id, item.checkState(self.COL_PRELOAD) == Qt.Checked)
            return
        enabled = item.checkState(self.COL_NAME) == Qt.Checked
        if not enabled and model_id == self.active_id:
            self._refreshing = True
            item.setCheckState(self.COL_NAME, Qt.Checked)
            self._refreshing = False
            self.status.setText(self.tr("El modelo activo no se puede ocultar: elige antes otro."))
            return
        self.manager.set_enabled(model_id, enabled)
        self.models_changed.emit()                         # el selector del panel se rehace al momento

    # --- Filtro por licencia ------------------------------------------------------------------------------------
    def shows_more_licenses(self) -> bool:
        """¿Se ven también los modelos de licencia no comercial o sin verificar? (decisión guardada del usuario)"""
        return QSettings().value("models/show_all_licenses", False, type=bool)

    def license_visible(self, entry: ModelEntry) -> bool:
        """Los modelos que la app aún no sabe usar (Marigold V2) no salen mientras no estén descargados. Del catálogo solo
        se ven por defecto los de licencia permisiva. Siempre se ven los que el usuario añadió, los
        que ya están instalados y el que se está usando: no se esconde nada que ya forme parte de su trabajo."""
        if entry.is_module:
            return True                       # los módulos no tienen licencias «de prueba»: se ven siempre (su licencia va en la lista)
        if not can_compute(entry) and not self.manager.is_installed(entry.id):
            return False                      # lo que la app aún no sabe usar no se ofrece (si ya está bajado, se ve para poder borrarlo)
        if self.shows_more_licenses() or entry.source != "catalog" or is_permissive(entry.license):
            return True
        return entry.id == self.active_id or self.manager.is_installed(entry.id)

    def _update_license_button(self) -> None:
        hidden = sum(1 for e in self.manager.all() if not self.license_visible(e))     # solo cuenta modelos
        if self.shows_more_licenses():
            self.btn_licenses.setText(self.tr("Ocultar licencias no comerciales"))
            self.btn_licenses.setToolTip(self.tr("Vuelve a enseñar solo los modelos cuya licencia permite el uso comercial "
                                                 "(los instalados y el activo se siguen viendo)."))
        else:
            self.btn_licenses.setText(self.tr("Mostrar más licencias…"))
            self.btn_licenses.setToolTip(self.tr(
                "Ahora solo se ven los modelos con licencia que permite uso comercial. Pulsa para ver también los de "
                "licencia no comercial o sin verificar ({0} ocultos); te pediremos confirmación.").format(hidden))

    def toggle_more_licenses(self) -> None:
        """Alterna el filtro. Para enseñar más licencias pide consentimiento: esos modelos no son para uso comercial."""
        if not self.shows_more_licenses():
            answer = QMessageBox.warning(
                self, self.tr("Más licencias"),
                self.tr("Vas a ver modelos con licencia NO comercial (p. ej. CC-BY-NC) o sin verificar. Sirven para "
                        "probar y comparar la calidad, pero NO se pueden usar para trabajos comerciales sin revisar su "
                        "licencia. Nunca se descargan solos.\n\n¿Mostrarlos?"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
            QSettings().setValue("models/show_all_licenses", True)
        else:
            QSettings().setValue("models/show_all_licenses", False)
        self.refresh()

    def refresh(self, select_id: str | None = None) -> None:
        select_id = select_id or self.selected_id()
        self._refreshing = True
        self.tree.clear()
        entries = [e for e in self.manager.all(with_modules=True) if self.kind_visible(e) and self.license_visible(e)]
        self._update_license_button()
        preload = self.manager.preload_id()
        for entry in entries:
            fam = ALL_FAMILIES.get(entry.family)
            name = ("● " if entry.id == self.active_id else "   ") + entry.display_label
            item = QTreeWidgetItem([name, "", "" if entry.is_module else (fam.label if fam else entry.family),
                                    self._source_text(entry), entry.license, self._state_text(entry), "",
                                    self.kind_text(entry), str(self.dest_path(entry))])
            item.setData(0, Qt.UserRole, entry.id)
            item.setToolTip(self.COL_DEST, self.tr("La carpeta donde se guarda esta entrada al instalarla"))
            if entry.is_module:
                item.setToolTip(self.COL_NAME, entry.description)
            else:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)   # casilla: ¿sale en el selector de modelos?
                item.setCheckState(self.COL_NAME, Qt.Checked if self.manager.is_enabled(entry.id)
                                   or entry.id == self.active_id else Qt.Unchecked)
            if entry.id == self.active_id:
                bold = QFont(item.font(0))
                bold.setBold(True)
                item.setFont(self.COL_NAME, bold)
                item.setToolTip(self.COL_NAME, self.tr("Modelo activo"))
            if not is_permissive(entry.license):
                item.setForeground(self.COL_LICENSE, QBrush(theme.color("warn")))
                item.setToolTip(self.COL_LICENSE, self.tr("Licencia no verificada como permisiva"))
            misfit = self.misfit_reason(entry)
            if not can_compute(entry):
                for col in (self.COL_NAME, self.COL_STATE):
                    item.setForeground(col, QBrush(theme.color("bad")))
                    item.setToolTip(col, self.no_compute_reason(entry))
                item.setText(self.COL_STATE, self.tr("Sin cálculo todavía · {0:.1f} GB").format(
                    self.manager.disk_size(entry.id) / 1e9 if self.manager.is_installed(entry.id) else entry.approx_gb or 0))
            elif misfit:              # no cabe en la GPU: rojo (el amarillo queda para la licencia y los avisos)
                for col in (self.COL_NAME, self.COL_STATE):
                    item.setForeground(col, QBrush(theme.color("bad")))
                    item.setToolTip(col, "\n\n".join(x for x in (misfit, entry.display_warning) if x))
            elif entry.warning:       # requisitos exigentes: el nombre y el estado en color de aviso, con el texto al pasar el ratón
                for col in (self.COL_NAME, self.COL_STATE):
                    item.setForeground(col, QBrush(theme.color("warn")))
                    item.setToolTip(col, entry.display_warning)
            if not entry.is_module and self.manager.is_installed(entry.id):     # «Al arrancar»: solo uno, y solo si está instalado
                item.setCheckState(self.COL_PRELOAD, Qt.Checked if entry.id == preload else Qt.Unchecked)
                item.setToolTip(self.COL_PRELOAD, self.tr("Carga este modelo en la memoria de la GPU al abrir la "
                                                          "aplicación. Solo uno a la vez."))
            if entry.id in self._updates:
                item.setText(self.COL_STATE, self._state_text(entry))
                item.setForeground(self.COL_STATE, QBrush(theme.color("warn")))
            self.tree.addTopLevelItem(item)
            if entry.repo_id:       # los de Hugging Face y los del catálogo llevan el enlace a su ficha; los de carpeta no
                self.tree.setItemWidget(item, self.COL_LINK, self._link_button(entry.repo_id))
            elif entry.url:         # los módulos, el enlace a su página (si la declaran en su module.json)
                self.tree.setItemWidget(item, self.COL_LINK, self._link_button(None, entry.url))
            if entry.id == select_id:
                item.setSelected(True)
        if not self._header_restored:      # solo la primera vez: después mandan los anchos que haya puesto el usuario
            for col in (self.COL_FAMILY, self.COL_SOURCE, self.COL_LICENSE, self.COL_STATE):
                self.tree.resizeColumnToContents(col)
            self._header_restored = True
        if not self.tree.selectedItems() and self.tree.topLevelItemCount():
            self.tree.topLevelItem(0).setSelected(True)
        self._refreshing = False
        self._current_id = self._tree_selected_id()
        self._rebuild_cards(entries, preload)
        self._update_buttons()

    def _tree_selected_id(self) -> str | None:
        items = self.tree.selectedItems() if hasattr(self, "tree") else []
        return items[0].data(0, Qt.UserRole) if items else None

    def selected_id(self) -> str | None:
        """El modelo seleccionado, en la lista o en las tarjetas (las dos vistas comparten la selección)."""
        return self._current_id if getattr(self, "_current_id", None) else self._tree_selected_id()

    def _on_tree_selection(self) -> None:
        if not self._refreshing:
            self._current_id = self._tree_selected_id()
            self._mark_current_card()
        self._update_buttons()

    def _on_view_changed(self, index: int) -> None:
        self.view_stack.setCurrentIndex(index)
        QSettings().setValue("models/view", "cards" if index == 1 else "list")
        self._relayout_cards()

    # --- Tarjetas ---------------------------------------------------------------------------------------------------
    def _card_data(self, entry: ModelEntry, preload: str | None) -> CardData:
        fam = ALL_FAMILIES.get(entry.family)
        installed = self.manager.is_installed(entry.id)
        if entry.is_module:
            size = self.manager.disk_size(entry.id)
            tags = " | ".join(t for t in (entry.license, "v" + entry.version if entry.version else "",
                                          f"{size / 1e6:.2f} MB" if installed else "") if t)
            return CardData(
                model_id=entry.id, title=entry.display_label, repo="", tags=tags, status=self._state_text(entry),
                status_tone="ok" if installed else "dim", installed=installed, update=False, active=False, enabled=True,
                preload=False, removable_files=installed, can_install=True, warn_license=not is_permissive(entry.license),
                kind_text=self.kind_text(entry), dest=str(self.dest_path(entry)), is_module=True,
                description=entry.description, url=entry.url)
        misfit = self.misfit_reason(entry)
        if not can_compute(entry):
            status, tone = self.tr("Sin cálculo todavía · {0:.1f} GB").format(
                self.manager.disk_size(entry.id) / 1e9 if installed else entry.approx_gb or 0), "bad"
        elif misfit and not installed:
            status, tone = self.tr("No cabe en este equipo"), "bad"
        elif entry.id in self._updates:
            status, tone = self.tr("Actualización disponible"), "warn"
        elif installed:
            status, tone = self.tr("Instalado · {0:.1f} GB").format(self.manager.disk_size(entry.id) / 1e9), "ok"
        elif entry.approx_gb:
            status, tone = self.tr("No descargado · {0:.1f} GB").format(entry.approx_gb), "dim"
        else:
            status, tone = self.tr("No descargado"), "dim"
        size = f"{self.manager.disk_size(entry.id) / 1e9:.1f} GB" if installed else (
            f"{entry.approx_gb:.1f} GB" if entry.approx_gb else "")
        tags = " | ".join(t for t in (fam.label if fam else entry.family, entry.license, size) if t)
        return CardData(
            model_id=entry.id, title=entry.display_label, repo=entry.repo_id or "", tags=tags, status=status,
            status_tone=tone, installed=installed, update=entry.id in self._updates, active=entry.id == self.active_id,
            enabled=self.manager.is_enabled(entry.id) or entry.id == self.active_id, preload=entry.id == preload,
            removable_files=installed, can_install=entry.source not in ("local", "folder"),
            warn_license=not is_permissive(entry.license), note=entry.display_warning, misfit=misfit,
            kind_text=self.kind_text(entry), dest=str(self.dest_path(entry)))

    def _rebuild_cards(self, entries: list[ModelEntry], preload: str | None) -> None:
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        self._cards = {}
        for entry in entries:
            card = ModelCard(self._card_data(entry, preload), self._link_icon, self._busy)
            card.selected.connect(self._select_card)
            card.useRequested.connect(self.activate)
            card.installRequested.connect(self.download)
            card.updateRequested.connect(self.update_model)
            card.uninstallRequested.connect(self.delete)
            card.enabledChanged.connect(self._card_enabled)
            card.preloadChanged.connect(self._set_preload)
            self._cards[entry.id] = card
        self._mark_current_card()
        self._relayout_cards()

    def _relayout_cards(self) -> None:
        """Reparte las tarjetas en columnas según el ancho disponible."""
        width = max(self.cards_scroll.viewport().width(), self.CARD_WIDTH)
        columns = max(1, width // self.CARD_WIDTH)
        for i, card in enumerate(self._cards.values()):
            self.cards_grid.addWidget(card, i // columns, i % columns)
        for col in range(columns):
            self.cards_grid.setColumnStretch(col, 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout_cards()

    def _select_card(self, model_id: str) -> None:
        self._current_id = model_id
        self._mark_current_card()
        self._refreshing = True                           # la lista sigue a la tarjeta, sin volver a disparar lo de la lista
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setSelected(item.data(0, Qt.UserRole) == model_id)
        self._refreshing = False
        self._update_buttons()

    def _mark_current_card(self) -> None:
        for model_id, card in self._cards.items():
            card.set_current(model_id == self._current_id)

    def _card_enabled(self, model_id: str, enabled: bool) -> None:
        if not enabled and model_id == self.active_id:
            self.status.setText(self.tr("El modelo activo no se puede ocultar: elige antes otro."))
            self.refresh()
            return
        self.manager.set_enabled(model_id, enabled)
        self.models_changed.emit()
        self.refresh()

    def _set_preload(self, model_id: str, on: bool) -> None:
        """«Cargar al arrancar»: solo un modelo a la vez (la GPU no admite varios)."""
        self.manager.set_preload(model_id if on else None)
        self.status.setText(self.tr("Se cargará al arrancar: {0}.").format(self.manager.get(model_id).display_label)
                            if on else self.tr("No se cargará ningún modelo al arrancar."))
        self.refresh()

    def _update_buttons(self) -> None:
        model_id = self.selected_id()
        entry = self.manager.get(model_id) if model_id else None
        installed = bool(entry) and self.manager.is_installed(entry.id)
        idle = not self._busy
        is_module = bool(entry) and entry.is_module
        self.btn_activate.setEnabled(idle and entry is not None and not is_module and entry.id != self.active_id)
        self.btn_download.setEnabled(idle and entry is not None and not installed
                                     and entry.source not in ("local", "folder"))
        # El botón desactivado explica por qué, en vez de quedarse mudo.
        if not idle:
            why = self.tr("Espera: hay una tarea de profundidad en marcha.")
        elif entry is None:
            why = self.tr("Elige primero un modelo de la lista.")
        elif installed:
            why = self.tr("Este modelo ya está instalado.")
        elif entry.source in ("local", "folder"):
            why = self.tr("Este modelo viene de una carpeta local: no hay nada que descargar.")
        else:
            why = ""
        self.btn_download.setToolTip(why or (self.tr("Instala el módulo en la carpeta de módulos") if is_module
                                             else self.tr("Descarga el modelo a la carpeta de modelos")))
        self.btn_update.setEnabled(idle and entry is not None and entry.id in self._updates)
        self.btn_delete.setEnabled(idle and entry is not None and (installed or entry.removable))
        self.btn_check_updates.setEnabled(idle)
        self.btn_add_hf.setEnabled(idle)
        self.btn_add_folder.setEnabled(idle)
        self.btn_adopt.setVisible(self.manager.legacy_dir() is not None)
        self.btn_adopt.setEnabled(idle)

    # =====================================================================
    # Acciones
    # =====================================================================
    def activate_selected(self) -> None:
        self.activate(self.selected_id())

    def activate(self, model_id: str | None) -> None:
        if not model_id or self._busy or model_id == self.active_id:
            return
        no_compute = self.no_compute_reason(self.manager.get(model_id))
        if no_compute:                                   # no se deja activar: solo daría un error al calcular
            QMessageBox.information(self, self.tr("Aviso sobre el modelo"), no_compute)
            return
        misfit = self.misfit_reason(self.manager.get(model_id))
        if misfit and QMessageBox.warning(
                self, self.tr("Aviso sobre el modelo"),
                self.tr("{0}\n\nPuede fallar por falta de memoria o ir muy lento. ¿Usarlo igualmente?").format(misfit),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.active_id = model_id
        self.active_changed.emit(model_id)
        entry = self.manager.get(model_id)
        if not self.manager.is_installed(model_id):
            self.status.setText(self.tr("«{0}» se descargará la primera vez que calcules una profundidad.")
                                .format(entry.display_label))
        self.refresh(model_id)

    def download_selected(self) -> None:
        self.download(self.selected_id())

    def download(self, model_id: str | None) -> None:
        if not model_id or self._busy:
            return
        entry = self.manager.get(model_id)
        if entry.is_module:
            self._install_module(entry)
            return
        no_compute = self.no_compute_reason(entry)
        if no_compute and QMessageBox.warning(          # lo primero: de nada sirve bajar decenas de GB que no se pueden usar
                self, self.tr("Aviso sobre el modelo"),
                no_compute + "\n\n" + self.tr("¿Descargarlo igualmente ({0:.0f} GB)?").format(entry.approx_gb or 0),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if not entry.warning and not self._confirm_license(entry):     # los que llevan aviso propio no repiten el genérico
            return
        notice = entry.display_warning
        if entry.min_vram_gb:                    # compara con la GPU de este equipo (solo informa; no bloquea nada)
            from deepshot.device import detect_device

            gpu = detect_device("auto")
            if gpu.vram_gb is None:
                notice += "\n\n" + self.tr("No se pudo medir la memoria de la GPU de este equipo.")
            elif gpu.vram_gb < entry.min_vram_gb:
                notice += "\n\n" + self.tr("Esta GPU ({0}) tiene {1:.0f} GB de memoria y los autores piden unos {2:.0f} GB: "
                                           "puede no funcionar bien aquí.").format(gpu.name, gpu.vram_gb, entry.min_vram_gb)
            else:
                notice += "\n\n" + self.tr("Esta GPU ({0}, {1:.0f} GB) cumple lo que piden sus autores (unos {2:.0f} GB)."
                                           ).format(gpu.name, gpu.vram_gb, entry.min_vram_gb)
        if notice and QMessageBox.warning(
                self, self.tr("Aviso sobre el modelo"),
                self.tr("{0}\n\n¿Descargar «{1}» ({2:.0f} GB)?").format(notice, entry.display_label,
                                                                     entry.approx_gb or 0),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._run(lambda report: self.manager.download(model_id, report),
                  self.tr("«{0}» descargado.").format(entry.display_label))

    def _install_module(self, entry: ModelEntry) -> None:
        """Instala un módulo: avisa de lo que necesita, pide la aprobación del código y lo da de alta como capa de Look."""
        missing = self.manager.missing_requirements(entry)
        if missing:
            QMessageBox.information(
                self, self.tr("Falta un modelo"),
                self.tr("El módulo «{0}» necesita estos modelos, que todavía no están instalados:\n\n{1}\n\n"
                        "Instálalos primero desde esta misma lista (con el filtro «Modelos») y vuelve a intentarlo."
                        ).format(entry.display_label, "\n".join("• " + m.display_label for m in missing)))
            return
        answer = QMessageBox.warning(
            self, self.tr("Instalar módulo"),
            self.tr("«{0}» es código Python que se ejecutará en tu equipo con tus permisos, igual que la propia "
                    "aplicación. Instálalo solo si confías en quien lo ha escrito.\n\nLicencia: {1}\nSe instalará "
                    "en: {2}\n\n¿Instalarlo?").format(entry.display_label, entry.license, self.dest_path(entry)),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return

        def after() -> None:                              # aprobado al instalar: el usuario acaba de confirmarlo
            installed = modules.installed_info(entry.path or "")
            if installed is not None:
                modules.approve(installed)
            modules.register_installed()
            self.models_changed.emit()
            self.refresh(entry.id)

        self._run(lambda report: self.manager.download(entry.id, report),
                  self.tr("Módulo «{0}» instalado: ya sale en «+ Añadir capa» del paso Look.").format(entry.display_label),
                  after=after)

    def delete_selected(self) -> None:
        self.delete(self.selected_id())

    def delete(self, model_id: str | None) -> None:
        if not model_id or self._busy:
            return
        if self.is_depth_busy():
            QMessageBox.information(self, self.windowTitle(),
                                    self.tr("Espera a que termine el cálculo de profundidad."))
            return
        entry = self.manager.get(model_id)
        if entry.is_module:
            if QMessageBox.question(self, self.tr("Desinstalar módulo"),
                                    self.tr("¿Desinstalar el módulo «{0}»?\nLas capas de tus proyectos que lo usen se "
                                            "conservan, pero no hacen nada hasta que lo vuelvas a instalar."
                                            ).format(entry.display_label)) != QMessageBox.Yes:
                return
            self.manager.delete(model_id)
            modules.register_installed()
            self.status.setText(self.tr("Módulo «{0}» desinstalado.").format(entry.display_label))
            self.models_changed.emit()
            self.refresh()
            return
        size = self.manager.disk_size(model_id) / 1e9
        text = self.tr("¿Borrar «{0}» ({1:.1f} GB)?").format(entry.display_label, size)
        if entry.source == "catalog":
            text += "\n" + self.tr("Seguirá en la lista y podrás volver a descargarlo.")
        if QMessageBox.question(self, self.tr("Borrar modelo"), text) != QMessageBox.Yes:
            return
        if self.estimator.loaded_model == model_id:
            self.estimator.unload()  # liberar la GPU y los archivos antes de borrar
        try:
            self.manager.delete(model_id)
        except OSError as exc:
            log.exception("No se pudo borrar %s", model_id)
            QMessageBox.warning(self, self.tr("Borrar modelo"),
                                self.tr("No se pudo borrar el modelo: {0}").format(exc.strerror or exc))
            return
        if entry.removable and model_id == self.active_id:
            from deepshot.models import DEFAULT_MODEL

            self.active_id = DEFAULT_MODEL
            self.active_changed.emit(DEFAULT_MODEL)
        self.status.setText(self.tr("«{0}» borrado.").format(entry.display_label))
        self.models_changed.emit()
        self.refresh()

    def update_selected(self) -> None:
        self.update_model(self.selected_id())

    def update_model(self, model_id: str | None) -> None:
        """Descarga la versión más nueva del modelo y borra la anterior al terminar bien."""
        if not model_id or self._busy or model_id not in self._updates:
            return
        entry = self.manager.get(model_id)
        if QMessageBox.question(self, self.tr("Actualizar modelo"),
                                self.tr("¿Actualizar «{0}»? Se descargará la versión nueva y se borrará la anterior al "
                                        "terminar.").format(entry.display_label)) != QMessageBox.Yes:
            return
        if self.estimator.loaded_model == model_id:
            self.estimator.unload()                       # suelta los archivos antes de sustituirlos

        def done() -> None:
            self._updates.discard(model_id)
            self.refresh(model_id)

        self._run(lambda report: self.manager.update(model_id, report),
                  self.tr("«{0}» actualizado.").format(entry.display_label), refresh_models=True, after=done)

    def check_updates(self) -> None:
        """Consulta (a mano) si los modelos instalados tienen una versión más nueva en Hugging Face."""
        found: list[set[str]] = []

        def done() -> None:
            self._updates = found[0] if found else set()
            self.refresh()
            self.status.setText(self.tr("Hay {0} actualización(es) disponible(s).").format(len(self._updates))
                                if self._updates else self.tr("Todos los modelos están al día."))

        def job(report) -> None:
            found.append(self.manager.check_updates(report))

        self._run(job, self.tr("Comprobación terminada."), after=done)

    def open_search(self) -> None:
        dialog = SearchDialog(self.manager, self)
        dialog.exec()
        if dialog.added_any:
            self.models_changed.emit()
            self.refresh()
            self.status.setText(self.tr("Modelos añadidos. Selecciona uno y pulsa «Descargar»."))

    @staticmethod
    def _url(repo_id: str) -> str:
        return f"https://huggingface.co/{repo_id}"

    def _link_button(self, repo_id: str | None, page: str | None = None) -> QToolButton:
        """Botón con icono que abre la ficha del modelo (licencia, autor, archivos) o la página del módulo en el
        navegador del sistema."""
        url = page or self._url(repo_id)
        button = QToolButton()
        button.setIcon(self._link_icon)
        button.setAutoRaise(True)  # sin fondo hasta pasar el ratón
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(self.tr("Abrir la página del módulo\n{0}").format(url) if page
                          else self.tr("Ver la ficha en Hugging Face\n{0}").format(url))
        button.setStyleSheet("QToolButton { background: transparent; border: none; padding: 1px; }"
                             f"QToolButton:hover {{ background: {theme.color('panel').name()}; border-radius: 3px; }}")
        button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
        return button

    # --- Carpeta de modelos -------------------------------------------------------------------------
    def _update_folder_label(self) -> None:
        self.folder_label.setText(self.tr("Carpeta de modelos: {0}").format(self.manager.models_dir))
        self.folder_label.setToolTip(self.tr("Aquí se guardan y se buscan los modelos"))

    def open_models_folder(self) -> None:
        self.manager.models_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.manager.models_dir)))

    def refresh_from_disk(self) -> None:
        """Vuelve a leer la carpeta: aparecen los modelos copiados a mano y desaparecen los que se quitaron."""
        self.refresh()
        self.models_changed.emit()
        found = len(self.manager.scan_folders())
        self.status.setText(self.tr("Lista actualizada. Modelos encontrados en la carpeta: {0}.").format(found))

    def adopt_legacy(self) -> None:
        old = self.manager.legacy_dir()
        if old is None:
            return
        if QMessageBox.question(self, self.tr("Mover modelos"),
                                self.tr("¿Mover los modelos de «{0}» a «{1}»?").format(old, self.manager.models_dir)
                                ) != QMessageBox.Yes:
            return
        if self.estimator.loaded_model:
            self.estimator.unload()                       # suelta los archivos antes de moverlos
        self._run(lambda report: self.manager.adopt_from(old, report), self.tr("Modelos movidos."),
                  refresh_models=True)

    def change_models_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.tr("Carpeta para los modelos"),
                                                  str(self.manager.models_dir))
        if not folder:
            return
        from deepshot.models import save_models_dir

        self.manager.set_models_dir(Path(folder))
        save_models_dir(Path(folder))
        modules.register_installed()                      # los módulos viven en «looks», dentro de la carpeta de modelos
        self._update_folder_label()
        self.refresh()
        self.models_changed.emit()
        self.status.setText(self.tr("Carpeta de modelos cambiada. Los modelos de la carpeta anterior no se han movido."))

    def install_from_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Carpeta del modelo (formato diffusers, con model_index.json)"))
        if not folder:
            return
        label, ok = QInputDialog.getText(self, self.tr("Nombre del modelo"),
                                         self.tr("Nombre con el que aparecerá en la lista:"),
                                         text=Path(folder).name)
        if not ok:
            return
        self._run(lambda report: self.manager.install_from_folder(Path(folder), label or None, report),
                  self.tr("Modelo instalado."), refresh_models=True)

    # =====================================================================
    # Licencias
    # =====================================================================
    def _confirm_license(self, entry: ModelEntry) -> bool:
        if is_permissive(entry.license):
            return True
        answer = QMessageBox.warning(
            self, self.tr("Licencia"),
            self.tr("La licencia de «{0}» es «{1}», que no está verificada como permisiva "
                    "(MIT, BSD, Apache 2.0). Revísala antes de usar el modelo.\n\n¿Continuar?")
            .format(entry.display_label, entry.license),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    # =====================================================================
    # Trabajo en segundo plano
    # =====================================================================
    def _run(self, job, success_message: str, refresh_models: bool = False, after=None) -> None:
        """Ejecuta `job(report)` en un hilo; la interfaz sigue respondiendo."""
        self._busy = True
        self._after = after
        self._refresh_models = refresh_models
        self._success_message = success_message
        self.progress.setRange(0, 0)
        self.progress.show()
        self._message = ""
        self._timer.start()
        self._update_buttons()

        def report(message: str, fraction: float | None) -> None:
            self._bridge.progress.emit(message, -1.0 if fraction is None else float(fraction))

        def worker() -> None:
            try:
                job(report)
                self._bridge.done.emit(success_message)
            except ModelError as exc:
                self._bridge.failed.emit(str(exc))
            except Exception:
                log.exception("Fallo en el gestor de modelos")
                self._bridge.failed.emit(self.tr("No se pudo completar la operación. "
                                                 "Consulta el panel Registro para ver el detalle."))

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, message: str, fraction: float) -> None:
        if fraction < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
        self._timer.update(None if fraction < 0 else fraction)
        self._message = message
        self.status.setText(with_time(message, self._timer))

    def _finish(self) -> None:
        self._timer.stop()
        self._busy = False
        self.progress.hide()
        self.refresh()

    def _on_done(self, message: str) -> None:
        self._finish()
        self.status.setText(message)
        if self._refresh_models:
            self.models_changed.emit()
        if self._after:
            self._after()

    def _on_failed(self, message: str) -> None:
        self._finish()
        self.status.setText(message)
        QMessageBox.warning(self, self.windowTitle(), message)

    def _header_menu(self, pos) -> None:
        """Clic derecho en la cabecera: elegir qué columnas se ven (el nombre no se puede ocultar)."""
        menu = QMenu(self)
        for col in range(self.tree.columnCount()):
            action = menu.addAction(self.tree.headerItem().text(col) or self.tr("Ficha"))
            action.setCheckable(True)
            action.setChecked(not self.tree.isColumnHidden(col))
            action.setEnabled(col != self.COL_NAME)
            action.toggled.connect(lambda shown, c=col: self.tree.setColumnHidden(c, not shown))
        menu.exec(self.tree.header().mapToGlobal(pos))

    def reject(self) -> None:
        self.accept()

    def accept(self) -> None:
        if self._busy:
            QMessageBox.information(self, self.windowTitle(),
                                    self.tr("Espera a que termine la operación en curso."))
            return
        QSettings().setValue("models/header", self.tree.header().saveState())
        super().accept()


class SearchDialog(QDialog):
    """Lista los modelos compatibles publicados en Hugging Face y permite añadirlos."""

    COL_REPO, COL_LINK, COL_ORIGIN, COL_LICENSE, COL_DOWNLOADS, COL_STATE = range(6)

    def __init__(self, manager: ModelManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Añadir desde Hugging Face"))
        self.setMinimumSize(760, 440)
        self.manager = manager
        self.added_any = False
        self._busy = False
        self._results: list[SearchResult] = []

        self._bridge = _Bridge(self)
        self._bridge.progress.connect(lambda msg, _f: self.status.setText(msg))
        self._bridge.done.connect(self._on_done)
        self._bridge.failed.connect(self._on_failed)

        intro = QLabel(self.tr(
            "Modelos públicos de Hugging Face que la app sabe ejecutar ({0}). "
            "Los «oficiales» son de los autores originales; los de terceros suelen ser copias "
            "o variantes sin verificar. El icono de cada fila abre su ficha en Hugging Face "
            "(licencia, autor, archivos).").format(", ".join(f.label for f in FAMILIES.values())))
        intro.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([self.tr("Repositorio"), self.tr("Ficha"), self.tr("Autor"),
                                   self.tr("Licencia"), self.tr("Descargas"), self.tr("Estado")])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.header().setSectionResizeMode(self.COL_REPO, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(self.COL_LINK, QHeaderView.Fixed)
        self.tree.header().resizeSection(self.COL_LINK, 46)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemDoubleClicked.connect(lambda *_: self.add_selected())
        self._link_icon = QIcon(str(resource("icons", "external-link.svg")))

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setRange(0, 0)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setProperty("dim", True)

        self.btn_add = QPushButton(self.tr("Añadir a la lista"))
        self.btn_add.setObjectName("Primary")
        self.btn_add.clicked.connect(self.add_selected)
        self.btn_retry = QPushButton(self.tr("Buscar de nuevo"))
        self.btn_retry.clicked.connect(self.search)
        close = QPushButton(self.tr("Cerrar"))
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.btn_retry)
        buttons.addWidget(self.btn_add)
        buttons.addWidget(close)

        # Campo manual, para un repositorio que no salga en la lista.
        self.manual = QLineEdit()
        self.manual.setPlaceholderText(self.tr("organización/modelo"))
        self.manual.returnPressed.connect(self.add_manual)
        self.btn_manual = QPushButton(self.tr("Añadir"))
        self.btn_manual.clicked.connect(self.add_manual)
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel(self.tr("¿No está en la lista? Escribe el repositorio:")))
        manual_row.addWidget(self.manual, 1)
        manual_row.addWidget(self.btn_manual)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.tree, 1)
        layout.addLayout(manual_row)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)

        self.search()

    # --- Búsqueda ----------------------------------------------------------------
    def search(self) -> None:
        """Consulta Hugging Face en segundo plano (no congela la ventana)."""
        self._start(self.tr("Buscando en Hugging Face…"))

        def worker() -> None:
            try:
                self._results = self.manager.search_compatible()
                self._bridge.done.emit("search")
            except ModelError as exc:
                self._bridge.failed.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _fill(self) -> None:
        self.tree.clear()
        for r in self._results:
            listed = self.manager.is_listed(r.repo_id)
            item = QTreeWidgetItem([
                r.repo_id,
                "",  # columna del icono de enlace
                self.tr("Oficial") if r.official else self.tr("Terceros"),
                r.license,
                f"{r.downloads:,}".replace(",", "."),
                self.tr("En la lista") if listed else "",
            ])
            item.setData(0, Qt.UserRole, r.repo_id)
            item.setData(0, Qt.UserRole + 1, listed)
            item.setToolTip(self.COL_REPO, self._url(r.repo_id))
            if not listed:
                item.setForeground(self.COL_REPO, QBrush(theme.color("link")))  # aspecto de enlace
            item.setTextAlignment(self.COL_DOWNLOADS, Qt.AlignRight | Qt.AlignVCenter)
            if r.official:
                bold = QFont(item.font(0))
                bold.setBold(True)
                item.setFont(self.COL_ORIGIN, bold)
            if not is_permissive(r.license):
                item.setForeground(self.COL_LICENSE, QBrush(theme.color("warn")))
                item.setToolTip(self.COL_LICENSE, self.tr("Licencia no verificada como permisiva"))
            if listed:  # ya está: en gris
                for col in range(self.tree.columnCount()):
                    item.setForeground(col, QBrush(theme.color("text_disabled")))
            self.tree.addTopLevelItem(item)
            self.tree.setItemWidget(item, self.COL_LINK, self._link_button(r.repo_id))
        for col in (self.COL_ORIGIN, self.COL_LICENSE, self.COL_DOWNLOADS, self.COL_STATE):
            self.tree.resizeColumnToContents(col)

    # --- Añadir ------------------------------------------------------------------
    def _selected_new(self) -> list[str]:
        return [i.data(0, Qt.UserRole) for i in self.tree.selectedItems() if not i.data(0, Qt.UserRole + 1)]

    def add_selected(self) -> None:
        repos = self._selected_new()
        if not repos or self._busy:
            return
        # Aviso si hay licencias no permisivas o modelos de terceros.
        risky = [r for r in self._results
                 if r.repo_id in repos and (not is_permissive(r.license) or not r.official)]
        if risky:
            lines = "\n".join(f"• {r.repo_id} — {r.license}" + ("" if r.official else self.tr(" (terceros)"))
                              for r in risky)
            answer = QMessageBox.warning(
                self, self.tr("Revisa antes de añadir"),
                self.tr("Estos modelos tienen una licencia no verificada como permisiva o no son "
                        "de los autores originales:\n\n{0}\n\n¿Añadirlos igualmente?").format(lines),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self._start(self.tr("Añadiendo…"))

        def worker() -> None:
            errors = []
            for repo in repos:
                try:
                    self.manager.add_from_huggingface(repo, lambda m, _f: self._bridge.progress.emit(m, -1.0))
                    self.added_any = True
                except ModelError as exc:
                    errors.append(str(exc))
            if errors:
                self._bridge.failed.emit("\n".join(errors))
            else:
                self._bridge.done.emit("added")

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _url(repo_id: str) -> str:
        return f"https://huggingface.co/{repo_id}"

    def _link_button(self, repo_id: str) -> QToolButton:
        """Botón con icono que abre la ficha del modelo en el navegador del sistema."""
        button = QToolButton()
        button.setIcon(self._link_icon)
        button.setAutoRaise(True)  # sin fondo hasta pasar el ratón
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(self.tr("Ver la ficha en Hugging Face\n{0}").format(self._url(repo_id)))
        button.setStyleSheet("QToolButton { background: transparent; border: none; padding: 1px; }"
                             f"QToolButton:hover {{ background: {theme.color('panel').name()}; border-radius: 3px; }}")
        button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._url(repo_id))))
        return button

    def add_manual(self) -> None:
        """Añade un repositorio escrito a mano (se comprueba familia y licencia)."""
        repo = self.manual.text().strip()
        if not repo or self._busy:
            return
        self._start(self.tr("Comprobando {0}…").format(repo))

        def worker() -> None:
            try:
                entry = self.manager.add_from_huggingface(repo, lambda m, _f: self._bridge.progress.emit(m, -1.0))
                self.added_any = True
                self._last_manual = entry
                self._bridge.done.emit("manual")
            except ModelError as exc:
                self._bridge.failed.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    # --- Estado ------------------------------------------------------------------
    def _start(self, message: str) -> None:
        self._busy = True
        self.progress.show()
        self.status.setText(message)
        self._update_buttons()

    def _stop(self) -> None:
        self._busy = False
        self.progress.hide()
        self._fill()
        self._update_buttons()

    def _on_done(self, what: str) -> None:
        self._stop()
        if what == "search":
            self.status.setText(self.tr("{0} modelos compatibles encontrados.").format(len(self._results)))
        elif what == "manual":
            entry = self._last_manual
            self.manual.clear()
            self.status.setText(self.tr("«{0}» añadido. Descárgalo desde el gestor de modelos.").format(entry.repo_id))
            if not is_permissive(entry.license):
                keep = QMessageBox.warning(
                    self, self.tr("Licencia"),
                    self.tr("La licencia de «{0}» es «{1}», que no está verificada como permisiva. "
                            "¿Mantenerlo en la lista?").format(entry.repo_id, entry.license),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if keep != QMessageBox.Yes:
                    self.manager.delete(entry.id)
                    self.status.setText(self.tr("«{0}» descartado.").format(entry.repo_id))
            self._fill()
        else:
            self.status.setText(self.tr("Añadidos. Descárgalos desde el gestor de modelos."))

    def _on_failed(self, message: str) -> None:
        self._stop()
        self.status.setText(message)
        QMessageBox.warning(self, self.windowTitle(), message)

    def _update_buttons(self) -> None:
        self.btn_add.setEnabled(not self._busy and bool(self._selected_new()))
        self.btn_retry.setEnabled(not self._busy)
        self.btn_manual.setEnabled(not self._busy)

    def reject(self) -> None:
        self.accept()

    def accept(self) -> None:
        if self._busy:
            return  # esperar a que termine la búsqueda o la instalación
        super().accept()
