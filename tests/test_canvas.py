"""Visor: modo comparativo A/B con divisor."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402

from deepshot.app import create_app  # noqa: E402
from deepshot.ui.canvas import CanvasView  # noqa: E402


def _img(w, h, color):
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor(color))
    return img


def test_compare_clips_b_to_its_side():
    create_app([])
    c = CanvasView()
    c.resize(400, 300)
    c.set_compare(_img(200, 100, "red"), _img(50, 25, "blue"), ("A", "B"))  # B se escala a A
    assert c.is_comparing()
    assert c._item_b.sceneBoundingRect() == c._item.sceneBoundingRect()  # B cubre A exactamente

    c.set_split(0.25)  # divisor vertical: B ocupa del 25 % a la derecha
    r = c._clip.rect()
    assert (r.left(), r.width(), r.height()) == (50, 150, 100)

    c.set_split_orientation(Qt.Vertical)  # divisor horizontal: B abajo
    c.set_split(0.5)
    r = c._clip.rect()
    assert (r.top(), r.height(), r.width()) == (50, 50, 200)

    c.set_split(2.0)  # fuera de rango: se limita
    assert c.split() == 1.0

    c.set_image(_img(200, 100, "green"))  # volver a imagen única quita B
    assert not c.is_comparing()
