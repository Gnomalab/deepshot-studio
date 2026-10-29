"""Diálogo Acerca de: pestañas, créditos obligatorios y que la técnica solo se nombre ahí."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from deepshot.app import create_app  # noqa: E402
from deepshot.ui.about import ZENODO_DOI, AboutDialog  # noqa: E402

INFO = {"app": "Deepshot Studio", "tagline": "t", "ver": "0.1", "url": "https://x.es", "org": "Org",
        "author": "Autor", "date": "2026"}


@pytest.fixture(scope="module")
def app():
    return create_app([])


def test_about_has_general_and_credits_tabs_with_required_citations(app):
    dlg = AboutDialog(INFO)
    assert dlg.tabs.count() == 2
    general = dlg.tabs.widget(0).toPlainText()
    credits = dlg.tabs.widget(1).toPlainText()
    assert "Deepshot Studio 0.1" in general and "sin afiliación" in general
    for needed in ("Karim Joseph Nassar", "CC BY 4.0", ZENODO_DOI, "Bingxin Ke", "ETH Zürich",
                   "Apache 2.0", "2312.02145", "2505.09358", "PySide6", "THIRD_PARTY_LICENSES"):
        assert needed in credits, needed
    dlg.deleteLater()


def test_reference_technique_is_named_only_in_about_and_legacy_constant():
    """Regla de CLAUDE.md: «Vathography» solo en Acerca de y en LEGACY_APP_NAME."""
    root = Path(__file__).resolve().parents[1] / "deepshot"
    hits = {p.relative_to(root).as_posix() for p in root.rglob("*.py") if "vathography" in p.read_text("utf-8").lower()}
    assert hits <= {"ui/about.py", "__init__.py"}, hits
