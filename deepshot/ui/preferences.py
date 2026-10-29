"""Ventana de Preferencias: inicio, apariencia (tema, redondeo, densidad) y disposición de los paneles.

Los cambios de apariencia se ven al momento (la ventana principal los aplica en caliente) y se guardan al cambiarlos.
"""

from __future__ import annotations

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication, QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepshot import depth, i18n, models, themes
from deepshot.ui import layouts, theme
from deepshot.ui.icons import set_icon, set_tab_icons

DENSITY_LABELS = (("compact", QT_TRANSLATE_NOOP("Preferences", "Compacta")),
                  ("standard", QT_TRANSLATE_NOOP("Preferences", "Normal")),
                  ("comfortable", QT_TRANSLATE_NOOP("Preferences", "Cómoda")))


ICON_STYLE_LABELS = (("color", QT_TRANSLATE_NOOP("Preferences", "En color")),
                     ("mono", QT_TRANSLATE_NOOP("Preferences", "Monocromo (más sobrio)")))


class PreferencesDialog(QDialog):
    def __init__(self, window, parent: QWidget | None = None) -> None:
        super().__init__(parent or window)
        self.window_ = window
        self.setWindowTitle(self.tr("Preferencias"))
        self.setMinimumWidth(460)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._general_tab(), self.tr("General"))
        self.tabs.addTab(self._appearance_tab(), self.tr("Apariencia"))
        self.tabs.addTab(self._layout_tab(), self.tr("Disposición"))
        self.tabs.setIconSize(QSize(18, 18))
        set_tab_icons(self.tabs, ["sliders", "effect", "compare"])          # general · apariencia · disposición
        close = QPushButton(self.tr("Cerrar"))
        close.setToolTip(self.tr("Los cambios ya están aplicados y guardados"))
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addLayout(bottom)

    @staticmethod
    def _heading(icon_name: str, text: str) -> QToolButton:
        """Título de sección con su icono (un botón plano que no se pulsa: así el icono se repinta con el tema)."""
        heading = QToolButton()
        set_icon(heading, icon_name)
        heading.setText(text)
        heading.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        heading.setIconSize(QSize(18, 18))
        heading.setAutoRaise(True)
        heading.setStyleSheet("QToolButton { background: transparent; border: none; padding: 2px 0; }")   # un título, no un botón
        heading.setFocusPolicy(Qt.NoFocus)
        heading.setAttribute(Qt.WA_TransparentForMouseEvents)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        return heading

    # --- General -----------------------------------------------------------------------------------
    def _general_tab(self) -> QWidget:
        page = QWidget()
        self.show_start = QCheckBox(self.tr("Mostrar la pantalla de inicio al abrir la aplicación"))
        self.show_start.setToolTip(self.tr("Pregunta si abrir un proyecto reciente o empezar uno nuevo. "
                                           "También está en Archivo › Pantalla de inicio."))
        self.show_start.setChecked(self.window_._start_enabled())
        self.show_start.toggled.connect(lambda on: self.window_.settings.setValue("startup/show_start", on))
        # Modelos de prueba (licencias no comerciales o más antiguos): para comparar calidad internamente.
        self.show_test_models = QCheckBox(self.tr("Mostrar modelos de prueba en el gestor de modelos"))
        self.show_test_models.setToolTip(self.tr("Añade a la lista modelos para comparar calidad: Depth Anything Base y "
                                                 "Large, MiDaS, ZoeDepth y Depth Pro. Algunos tienen licencia NO "
                                                 "comercial. No forman parte de la oferta final de la aplicación."))
        self.show_test_models.setChecked(models.dev_models_enabled())
        self.show_test_models.toggled.connect(self._test_models_toggled)
        # Idioma de la interfaz: se guarda al momento y se aplica al reiniciar (los textos ya creados no se retraducen).
        self.language_combo = QComboBox()
        self.language_combo.addItem(self.tr("Automático (el del sistema)"), "auto")
        self.language_combo.addItem("Español", "es")             # el nombre de cada idioma va en ese idioma
        self.language_combo.addItem("English", "en")
        self.language_combo.setToolTip(self.tr("Idioma de los menús, paneles y mensajes. Se aplica al reiniciar la "
                                               "aplicación."))
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(i18n.saved_choice())))
        self.language_note = QLabel(self.tr("Reinicia la aplicación para aplicar el idioma."))
        self.language_note.setProperty("tone", "warn")
        self.language_note.hide()
        self.language_combo.currentIndexChanged.connect(self._language_changed)
        # Caché de mapas de profundidad: se acumula (un PNG por foto, modelo y calidad); aquí se ve cuánto ocupa y se vacía.
        self.cache_label = QLabel()
        self.cache_label.setProperty("dim", True)
        self.cache_button = QPushButton(self.tr("Vaciar caché…"))
        set_icon(self.cache_button, "trash")
        self.cache_button.setToolTip(self.tr("Borra los mapas de profundidad guardados para no recalcularlos. Es seguro: "
                                             "se vuelven a calcular al abrir la foto. Los proyectos guardados no se tocan."))
        self.cache_button.clicked.connect(self._clear_cache)
        self._update_cache_label()
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.cache_label, 1)
        cache_row.addWidget(self.cache_button)
        layout = QVBoxLayout(page)
        layout.addWidget(self.show_start)
        layout.addWidget(self.show_test_models)
        layout.addSpacing(8)
        layout.addWidget(self._heading("depth", self.tr("Caché de mapas de profundidad")))
        layout.addLayout(cache_row)
        layout.addSpacing(8)
        layout.addWidget(self._heading("layers", self.tr("Idioma de la interfaz")))
        layout.addWidget(self.language_combo)
        layout.addWidget(self.language_note)
        layout.addStretch()
        return page

    def _update_cache_label(self) -> None:
        files, size = depth.cache_size()
        self.cache_label.setText(self.tr("{0} mapas · {1:.1f} MB").format(files, size / 1e6))
        self.cache_button.setEnabled(files > 0)

    def _clear_cache(self) -> None:
        files, size = depth.cache_size()
        answer = QMessageBox.question(
            self, self.tr("Vaciar caché"),
            self.tr("Se borrarán {0} mapas de profundidad ({1:.1f} MB). Se volverán a calcular cuando los necesites. "
                    "¿Continuar?").format(files, size / 1e6))
        if answer != QMessageBox.Yes:
            return
        count, freed = depth.clear_cache()
        self._update_cache_label()
        self.window_.statusBar().showMessage(self.tr("Caché vaciada: {0} archivos, {1:.1f} MB liberados.").format(
            count, freed / 1e6), 6000)
        try:
            self.window_._refresh_runs_list()          # la lista de cálculos de la foto lee de la caché
        except Exception:
            pass

    def _test_models_toggled(self, on: bool) -> None:
        self.window_.settings.setValue("models/show_test_models", on)
        self.window_._fill_model_combo()                 # la lista del panel de profundidad se rehace al momento

    def _language_changed(self, _index: int) -> None:
        choice = str(self.language_combo.currentData())
        i18n.save_choice(choice)
        self.language_note.setVisible(i18n.resolve(choice) != i18n.active())

    # --- Apariencia --------------------------------------------------------------------------------
    def _appearance_tab(self) -> QWidget:
        page = QWidget()
        self.theme_combo = QComboBox()
        self.theme_combo.setToolTip(self.tr("Colores de la interfaz. Se aplican al instante."))
        self.density_combo = QComboBox()
        for key, label in DENSITY_LABELS:
            self.density_combo.addItem(QCoreApplication.translate("Preferences", label), key)
        self.density_combo.setToolTip(self.tr("Cuánto aire llevan los botones, campos y listas"))
        self.icons_combo = QComboBox()
        for key, label in ICON_STYLE_LABELS:
            self.icons_combo.addItem(QCoreApplication.translate("Preferences", label), key)
        self.icons_combo.setToolTip(self.tr("Iconos de colores (un color por tipo de herramienta o acción) o en una sola "
                                            "tinta, más sobrios"))
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(0, 10)
        self.radius_slider.setSingleStep(1)
        self.radius_slider.setPageStep(1)          # el valor por defecto (10) hacía saltar de 0 a 10 al clicar la barra
        self.radius_slider.setTracking(True)
        self.radius_slider.setToolTip(self.tr("Redondeo de las esquinas de botones y campos (0 = rectas)"))
        self.radius_label = QLabel()
        self.radius_label.setMinimumWidth(40)
        self.themes_button = QPushButton(self.tr("Abrir la carpeta de temas"))
        set_icon(self.themes_button, "folder")
        self.themes_button.setToolTip(self.tr("Cada tema propio es un archivo .json con colores; copia uno aquí "
                                              "y pulsa «Actualizar lista»"))
        self.themes_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(themes.themes_dir()))))
        self.reload_button = QPushButton(self.tr("Actualizar lista"))
        set_icon(self.reload_button, "refresh")
        self.reload_button.setToolTip(self.tr("Vuelve a leer los temas de la carpeta"))
        self.reload_button.clicked.connect(self._fill_themes)
        self._fill_themes()
        self.density_combo.setCurrentIndex(max(0, self.density_combo.findData(theme.current_appearance().density)))
        self.radius_slider.setValue(theme.current_appearance().radius)
        self.icons_combo.setCurrentIndex(max(0, self.icons_combo.findData(theme.current_appearance().icons)))
        self._update_radius_label()
        self.theme_combo.currentIndexChanged.connect(self._changed)
        self.density_combo.currentIndexChanged.connect(self._changed)
        self.icons_combo.currentIndexChanged.connect(self._changed)
        self.radius_slider.valueChanged.connect(self._changed)

        layout = QVBoxLayout(page)
        layout.addWidget(self._heading("effect", self.tr("Tema de color")))
        layout.addWidget(self.theme_combo)
        row = QHBoxLayout()
        row.addWidget(self.themes_button)
        row.addWidget(self.reload_button)
        layout.addLayout(row)
        layout.addWidget(self._heading("sliders", self.tr("Densidad de los controles")))
        layout.addWidget(self.density_combo)
        layout.addWidget(self._heading("photo", self.tr("Estilo de los iconos")))
        layout.addWidget(self.icons_combo)
        layout.addWidget(self._heading("draw_area", self.tr("Redondeo de las esquinas")))
        radius_row = QHBoxLayout()
        radius_row.addWidget(self.radius_slider, 1)
        radius_row.addWidget(self.radius_label)
        layout.addLayout(radius_row)
        layout.addStretch()
        return page

    def _fill_themes(self) -> None:
        current = theme.current_theme().key
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        for t in themes.all_themes():
            label = QCoreApplication.translate("Themes", t.label) if t.factory else t.label
            self.theme_combo.addItem(label if t.factory else f"{label}  ({self.tr('propio')})", t.key)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(current)))
        self.theme_combo.blockSignals(False)

    def _update_radius_label(self) -> None:
        self.radius_label.setText(f"{self.radius_slider.value()} px")

    def _changed(self, *_args) -> None:
        self._update_radius_label()
        self.window_.apply_appearance(
            str(self.theme_combo.currentData()),
            themes.Appearance(radius=self.radius_slider.value(), density=str(self.density_combo.currentData()),
                              icons=str(self.icons_combo.currentData())))

    # --- Disposición -------------------------------------------------------------------------------
    def _layout_tab(self) -> QWidget:
        page = QWidget()
        self.layout_list = QListWidget()
        self.layout_list.setToolTip(self.tr("Disposiciones de los paneles: las de fábrica y las tuyas"))
        self.apply_layout_button = QPushButton(self.tr("Aplicar"))
        set_icon(self.apply_layout_button, "import")
        self.apply_layout_button.clicked.connect(self._apply_layout)
        self.save_layout_button = QPushButton(self.tr("Guardar la actual como…"))
        set_icon(self.save_layout_button, "save")
        self.save_layout_button.setToolTip(self.tr("Guarda dónde están ahora los paneles y su tamaño"))
        self.save_layout_button.clicked.connect(self._save_layout)
        self.delete_layout_button = QPushButton(self.tr("Borrar"))
        set_icon(self.delete_layout_button, "trash")
        self.delete_layout_button.setToolTip(self.tr("Borra una disposición propia (las de fábrica no se borran)"))
        self.delete_layout_button.clicked.connect(self._delete_layout)
        self.reset_layout_button = QPushButton(self.tr("Restablecer disposición"))
        set_icon(self.reset_layout_button, "refresh")
        self.reset_layout_button.setToolTip(self.tr("Vuelve a la disposición de edición del principio"))
        self.reset_layout_button.clicked.connect(self.window_.reset_layout)
        self.layout_list.itemDoubleClicked.connect(lambda _i: self._apply_layout())
        self.layout_list.itemSelectionChanged.connect(self._update_layout_buttons)
        self._fill_layouts()
        row = QHBoxLayout()
        for button in (self.apply_layout_button, self.save_layout_button, self.delete_layout_button):
            row.addWidget(button)
        layout = QVBoxLayout(page)
        layout.addWidget(self.layout_list, 1)
        layout.addLayout(row)
        layout.addWidget(self.reset_layout_button)
        return page

    def _fill_layouts(self) -> None:
        self.layout_list.clear()
        for key, label in layouts.factory_names():
            item = QListWidgetItem(QCoreApplication.translate("Layouts", label))
            item.setData(Qt.UserRole, ("factory", key))
            self.layout_list.addItem(item)
        for name in layouts.list_user():
            item = QListWidgetItem(f"{name}  ({self.tr('propia')})")
            item.setData(Qt.UserRole, ("user", name))
            self.layout_list.addItem(item)
        self.layout_list.setCurrentRow(0)
        self._update_layout_buttons()

    def _selected_layout(self) -> tuple | None:
        items = self.layout_list.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def _update_layout_buttons(self) -> None:
        chosen = self._selected_layout()
        self.apply_layout_button.setEnabled(chosen is not None)
        self.delete_layout_button.setEnabled(chosen is not None and chosen[0] == "user")

    def _apply_layout(self) -> None:
        chosen = self._selected_layout()
        if chosen is not None:
            self.window_.apply_layout(*chosen)

    def _save_layout(self) -> None:
        name, ok = QInputDialog.getText(self, self.tr("Guardar disposición"), self.tr("Nombre de la disposición:"))
        if not ok or not name.strip():
            return
        if layouts.exists_user(name):
            answer = QMessageBox.question(self, self.tr("Guardar disposición"),
                                          self.tr("Ya existe «{0}». ¿Sustituirla?").format(name.strip()))
            if answer != QMessageBox.Yes:
                return
        self.window_.save_layout(name)
        self._fill_layouts()

    def _delete_layout(self) -> None:
        chosen = self._selected_layout()
        if chosen is None or chosen[0] != "user":
            return
        answer = QMessageBox.question(self, self.tr("Borrar disposición"),
                                      self.tr("¿Borrar la disposición «{0}»?").format(chosen[1]))
        if answer == QMessageBox.Yes:
            layouts.delete_user(chosen[1])
            self._fill_layouts()
