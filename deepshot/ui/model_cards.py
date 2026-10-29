"""Tarjeta de un modelo para el gestor de modelos (vista «Tarjetas»): nombre, repositorio, etiquetas, estado, casillas y
botones de cada modelo. No sabe nada del disco ni de Hugging Face: recibe los datos ya preparados y avisa con señales."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget



@dataclass
class CardData:
    """Todo lo que enseña una tarjeta (la ventana del gestor lo calcula y la tarjeta solo lo dibuja)."""

    model_id: str
    title: str
    repo: str                 # «org/modelo» o texto vacío si no es de Hugging Face
    tags: str                 # «Familia | licencia | tamaño»
    status: str
    status_tone: str          # "ok" | "warn" | "bad" | "dim"
    installed: bool
    update: bool              # hay una versión más nueva
    active: bool              # es el modelo que se usa
    enabled: bool             # sale en el selector del panel
    preload: bool             # se carga al arrancar
    removable_files: bool     # tiene archivos que desinstalar
    can_install: bool         # se puede descargar (no es de carpeta)
    warn_license: bool        # licencia no comercial o sin verificar
    note: str = ""            # aviso largo (se ve al pasar el ratón)
    misfit: str = ""          # si no cabe en la GPU de este equipo: el motivo (el título y el estado salen en rojo)
    kind_text: str = ""       # tipo de entrada: «Modelo de profundidad», «Módulo de Look»…
    dest: str = ""            # carpeta donde se instala
    is_module: bool = False   # los módulos no se «usan» ni se ocultan del selector: solo se instalan y se quitan
    description: str = ""     # módulos: qué hacen
    url: str = ""             # módulos: su página web


class ModelCard(QFrame):
    """Una tarjeta. Señales con el id del modelo; las casillas avisan con su nuevo valor."""

    selected = Signal(str)
    useRequested = Signal(str)
    installRequested = Signal(str)
    updateRequested = Signal(str)
    uninstallRequested = Signal(str)
    enabledChanged = Signal(str, bool)
    preloadChanged = Signal(str, bool)

    def __init__(self, data: CardData, link_icon: QIcon, busy: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data = data
        self.setObjectName("ModelCard")
        self.setProperty("active", data.active)
        self.setProperty("current", False)
        self.setMinimumWidth(200)
        if data.note or data.misfit:
            self.setToolTip("\n\n".join(x for x in (data.misfit, data.note) if x))
        id_ = data.model_id

        title = QLabel(data.title)
        title.setWordWrap(True)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        if data.misfit:
            title.setProperty("tone", "bad")
        elif data.warn_license:
            title.setProperty("tone", "warn")
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(title, 1)
        self.link_button: QToolButton | None = None
        if data.repo or data.url:                                  # enlace a la ficha del modelo (HF) o a la página del módulo
            self.link_button = QToolButton()
            self.link_button.setIcon(link_icon)
            self.link_button.setAutoRaise(True)
            self.link_button.setCursor(Qt.PointingHandCursor)
            url = data.url or f"https://huggingface.co/{data.repo}"
            self.link_button.setToolTip(self.tr("Abrir la página del módulo\n{0}").format(url) if data.url
                                        else self.tr("Ver la ficha en Hugging Face\n{0}").format(url))
            self.link_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
            head.addWidget(self.link_button, 0, Qt.AlignTop)

        repo = QLabel(data.kind_text if data.is_module else (data.repo or self.tr("(carpeta local)")))
        repo.setProperty("dim", True)
        repo.setWordWrap(True)
        repo.setTextInteractionFlags(Qt.TextSelectableByMouse)
        tags = QLabel(data.tags)
        tags.setWordWrap(True)
        tags.setProperty("dim", True)
        status = QLabel(data.status)
        status.setWordWrap(True)
        status.setProperty("tone", data.status_tone)
        description = QLabel(data.description)
        description.setWordWrap(True)
        description.setVisible(bool(data.description))
        dest = QLabel(self.tr("Se instala en: {0}").format(data.dest))
        dest.setWordWrap(True)
        dest.setProperty("dim", True)
        dest.setTextInteractionFlags(Qt.TextSelectableByMouse)
        dest.setToolTip(self.tr("La carpeta donde se guarda esta entrada al instalarla"))
        dest.setVisible(bool(data.dest))

        self.enabled_check = QCheckBox(self.tr("Mostrar en el selector"))
        self.enabled_check.setToolTip(self.tr("Si está marcado, el modelo sale en el selector de modelos del panel de "
                                              "profundidad. Desmarcarlo no lo borra: solo lo oculta."))
        self.enabled_check.setChecked(data.enabled)
        self.enabled_check.toggled.connect(lambda on: self.enabledChanged.emit(id_, on))
        self.preload_check = QCheckBox(self.tr("Cargar al arrancar"))
        self.preload_check.setToolTip(self.tr("Carga este modelo en la memoria de la GPU al abrir la aplicación, para "
                                              "que el primer cálculo sea más rápido. Solo uno a la vez."))
        self.preload_check.setChecked(data.preload)
        self.preload_check.setEnabled(data.installed)
        self.preload_check.toggled.connect(lambda on: self.preloadChanged.emit(id_, on))
        self.enabled_check.setVisible(not data.is_module)          # un módulo no sale en el selector de modelos
        self.preload_check.setVisible(not data.is_module)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        self.install_button = self.use_button = self.update_button = self.uninstall_button = None
        if not data.installed:
            self.install_button = QPushButton(self.tr("Instalar"))
            self.install_button.setToolTip(self.tr("Instala el módulo en la carpeta de módulos") if data.is_module
                                           else self.tr("Descarga el modelo a la carpeta de modelos"))
            self.install_button.setEnabled(data.can_install and not busy)
            self.install_button.clicked.connect(lambda: self.installRequested.emit(id_))
            buttons.addWidget(self.install_button)
        else:
            if not data.is_module:
                self.use_button = QPushButton(self.tr("Usar"))
                self.use_button.setObjectName("Primary")
                self.use_button.setToolTip(self.tr("Usa este modelo para calcular la profundidad"))
                self.use_button.setEnabled(not data.active and not busy)
                self.use_button.clicked.connect(lambda: self.useRequested.emit(id_))
                buttons.addWidget(self.use_button)
            if data.update:
                self.update_button = QPushButton(self.tr("Actualizar"))
                self.update_button.setToolTip(self.tr("Hay una versión más nueva en Hugging Face: la descarga y borra la "
                                                      "anterior cuando termina bien."))
                self.update_button.setEnabled(not busy)
                self.update_button.clicked.connect(lambda: self.updateRequested.emit(id_))
                buttons.addWidget(self.update_button)
            if data.removable_files:
                self.uninstall_button = QPushButton(self.tr("Desinstalar"))
                self.uninstall_button.setToolTip(self.tr("Borra los archivos del módulo de la carpeta de módulos")
                                                 if data.is_module
                                                 else self.tr("Borra los archivos del modelo de la carpeta de modelos"))
                self.uninstall_button.setEnabled(not busy)
                self.uninstall_button.clicked.connect(lambda: self.uninstallRequested.emit(id_))
                buttons.addWidget(self.uninstall_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(4)
        layout.addLayout(head)
        for widget in (repo, tags, description, status, dest, self.enabled_check, self.preload_check):
            layout.addWidget(widget)
        layout.addStretch(1)
        layout.addLayout(buttons)

    def set_current(self, current: bool) -> None:
        """Marca la tarjeta seleccionada (borde) sin cambiar nada más."""
        self.setProperty("current", current)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:       # clic en cualquier parte de la tarjeta = seleccionarla
        self.selected.emit(self.data.model_id)
        super().mousePressEvent(event)

