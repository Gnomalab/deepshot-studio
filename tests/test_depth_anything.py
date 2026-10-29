"""Depth Anything V2: convención de la salida (0 = cerca) y tamaño, con un modelo falso (sin descargar pesos)."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from deepshot.depth import DepthEstimator  # noqa: E402


class FakeModel(torch.nn.Module):
    """Imita a Depth Anything: profundidad inversa, más grande = más cerca (aquí, más a la derecha)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values):
        b, _, h, w = pixel_values.shape
        assert h % 14 == 0 and w % 14 == 0                                   # múltiplos del parche
        ramp = torch.linspace(0, 10, w).expand(b, h, w)
        return type("Out", (), {"predicted_depth": ramp})()


def test_depth_anything_output_is_near_zero_and_matches_the_photo():
    est = DepthEstimator()
    est._pipe, est._family = FakeModel(), "depth-anything"
    image = Image.new("RGB", (200, 100), "gray")
    depth = est._run_depth_anything(image, 140)
    assert depth.shape == (100, 200) and depth.dtype == np.float32
    assert 0.0 <= depth.min() and depth.max() <= 1.0
    assert depth[:, -1].mean() < 0.05 and depth[:, 0].mean() > 0.95          # lo cercano (derecha) sale en 0
