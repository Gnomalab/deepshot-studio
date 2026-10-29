"""Diálogo «Acerca de»: pestaña General y pestaña Créditos.

La pestaña Créditos reúne las atribuciones: la técnica que inspira la app (cita propuesta por su autor
en Zenodo), el modelo de profundidad (autores y artículos que pide citar) y las bibliotecas de terceros.
Es el ÚNICO sitio de la interfaz donde se nombra la técnica de referencia (ver CLAUDE.md).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget, QTextBrowser, QVBoxLayout, QWidget

ZENODO_DOI = "https://doi.org/10.5281/zenodo.23103263"


class AboutDialog(QDialog):
    """Acerca de la aplicación. `info` lleva los datos de versión (app, ver, tagline, url, org, author, date)."""

    def __init__(self, info: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Acerca de {0}").format(info["app"]))
        self.resize(560, 520)
        tabs = QTabWidget()
        tabs.addTab(self._page(self._general_html(info)), self.tr("General"))
        tabs.addTab(self._page(self._credits_html()), self.tr("Créditos"))
        self.tabs = tabs
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    @staticmethod
    def _page(html: str) -> QTextBrowser:
        view = QTextBrowser()
        view.setOpenExternalLinks(True)          # los enlaces se abren en el navegador del sistema
        view.setHtml(html)
        view.setFocusPolicy(Qt.NoFocus)
        return view

    def _general_html(self, info: dict[str, str]) -> str:
        return self.tr(
            "<h3>{app} {ver}</h3>"
            "<p><i>{app} — {tagline}</i></p>"
            "<p>Imágenes creadas a partir del mapa de profundidad de una fotografía. "
            "Todo se procesa en tu equipo.</p>"
            "<p>By <a href='{url}'>{org}</a> · {author}<br>Versión del {date}</p>"
            # Mención de la técnica de referencia: aquí y en Créditos, nunca en otra parte de la interfaz.
            "<p>Inspirado en la técnica Vathography de Karim Joseph Nassar "
            "(<a href='https://vathography.com'>vathography.com</a>). "
            "Proyecto independiente, sin afiliación con el autor.</p>"
        ).format(**info)

    def _credits_html(self) -> str:
        return self.tr(
            "<h3>Técnica</h3>"
            "<p>Inspirada en la práctica <b>Vathography</b>, creada por <b>Karim Joseph Nassar</b> "
            "(licencia CC BY 4.0). Cita:<br>"
            "<i>Nassar, K. J. (2026). Vathography: The Art of a Photograph's Hidden Volume (Version 1.2). "
            "Zenodo.</i> <a href='{doi}'>{doi}</a></p>"
            "<p>{app} es un proyecto independiente: no usa código ni recursos de su add-on de Blender "
            "y no tiene afiliación con el autor.</p>"
            "<h3>Modelo de profundidad</h3>"
            "<p><b>Marigold</b> (licencia Apache 2.0): Bingxin Ke, Anton Obukhov, Shengyu Huang, "
            "Nando Metzger, Rodrigo Caye Daudt y Konrad Schindler, ETH Zürich (Photogrammetry and Remote "
            "Sensing). Se ejecuta con la biblioteca <i>diffusers</i> de Hugging Face.</p>"
            "<ul>"
            "<li>«Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation», CVPR 2024 "
            "(<a href='https://arxiv.org/abs/2312.02145'>arXiv 2312.02145</a>).</li>"
            "<li>«Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis» "
            "(<a href='https://arxiv.org/abs/2505.09358'>arXiv 2505.09358</a>).</li>"
            "</ul>"
            "<h3>Bibliotecas</h3>"
            "<p>Interfaz: Qt for Python / PySide6 (LGPL 3.0, enlazada dinámicamente). Cálculo: PyTorch, NumPy, "
            "SciPy, Pillow y OpenEXR. Cada una conserva su propia licencia: la lista completa está en el "
            "archivo <code>THIRD_PARTY_LICENSES.md</code> que acompaña a la aplicación.</p>"
        ).format(doi=ZENODO_DOI, app="Deepshot Studio")
