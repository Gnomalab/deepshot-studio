"""Refinado guiado: alinea bordes sin alterar zonas lisas."""

import numpy as np

from deepshot.refine import box_filter, guided_filter_color, refine_depth


def test_box_filter_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.random((9, 11)).astype(np.float32)
    r = 2
    got = box_filter(x, r)
    for i in range(9):
        for j in range(11):
            win = x[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
            assert abs(got[i, j] - win.mean()) < 1e-5


def test_constant_depth_stays_constant():
    photo = np.random.default_rng(1).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    depth = np.full((60, 80), 0.4, dtype=np.float32)
    out = refine_depth(depth, photo, work_size=80, radius=3)
    assert np.allclose(out, 0.4, atol=1e-3)


def _step_scene(h=64, w=96, edge=48):
    """Foto con un borde nítido en x=edge y mapa con ese borde emborronado."""
    photo = np.zeros((h, w, 3), dtype=np.uint8)
    photo[:, edge:] = (200, 60, 40)      # objeto rojo (cerca) a la derecha
    photo[:, :edge] = (40, 90, 200)      # fondo azul (lejos)
    x = np.arange(w, dtype=np.float32)
    blurred = 1.0 / (1.0 + np.exp((x - edge) / 4.0))  # transición suave de 1 (lejos) a 0 (cerca)
    depth = np.tile(blurred, (h, 1)).astype(np.float32)
    return photo, depth


def test_refine_sharpens_depth_edge_to_photo_edge():
    photo, depth = _step_scene()
    out = refine_depth(depth, photo, work_size=96, radius=4, eps=1e-4)
    row_before, row_after = depth[32], out[32]
    # Lejos del borde, cambia poco; junto al borde, el salto es mucho más brusco.
    jump_before = row_before[46] - row_before[50]
    jump_after = row_after[46] - row_after[50]
    assert jump_after > jump_before * 1.3  # el borde se vuelve claramente más brusco
    assert abs(row_after[5] - row_before[5]) < 0.05


def test_strength_zero_is_identity_and_output_in_range():
    photo, depth = _step_scene()
    assert refine_depth(depth, photo, strength=0) is depth
    out = refine_depth(depth, photo, strength=0.5, work_size=96)
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1


def test_full_resolution_guided_filter_shapes():
    photo, depth = _step_scene()
    q = guided_filter_color(photo.astype(np.float32) / 255, depth, 3, 1e-3)
    assert q.shape == depth.shape
