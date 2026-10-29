"""Fase 2: el procesado se reproduce exactamente desde un JSON de ajustes."""

import numpy as np

from deepshot.areas import Area, Scene
from deepshot.project import Composite, Project, render
from deepshot.tonemap import Band, ramp_preset


def _scene_data(h=40, w=60):
    rng = np.random.default_rng(3)
    depth = np.clip(np.tile(np.linspace(0.1, 0.9, w, dtype=np.float32), (h, 1))
                    + rng.normal(0, 0.02, (h, w)).astype(np.float32), 0, 1)
    photo = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return depth, photo


def _full_project() -> Project:
    p = Project(ramp=ramp_preset("color"), base="ramp")
    p.ramp.range_min, p.ramp.range_max = 0.1, 0.9
    p.bands = [Band(0.3, 0.1, 0.05, (1.0, 0.9, 0.7), 0.8), Band(0.7, 0.05, 0.0, (0, 0, 0), 0.5)]
    p.scene = Scene(relief=4.0, fov_deg=40)
    p.areas = [Area("Luz", "ellipsoid", (0.2, 0.1, 2.0), (0.5, 0.6, 0.8), (0, 15, 30), 0.7, 0.4, 0.3),
               Area("Sombra", "box", (-0.5, -0.2, 3.0), (0.3, 0.3, 0.5), (10, 0, 0), -1.0, 0.2, 0.0)]
    p.composite = Composite(mode="soft_light", opacity=0.6, photo_grayscale=True, mask="areas")
    return p


def test_json_roundtrip_reproduces_render(tmp_path):
    depth, photo = _scene_data()
    project = _full_project()
    first = render(project, depth, photo)

    path = tmp_path / "ajustes del deepshot.json"
    project.save(path)
    loaded = Project.load(path)

    assert loaded.to_dict() == project.to_dict()
    assert np.array_equal(render(loaded, depth, photo), first)


def test_render_ranges_and_defaults():
    depth, photo = _scene_data()
    out = render(Project(), depth, photo)            # por defecto: rampa blanco->negro, sin foto
    assert out.shape == depth.shape + (3,) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert np.allclose(out[..., 0], 1.0 - depth, atol=1e-5)


def test_photo_opacity_zero_ignores_photo_and_full_normal_is_photo():
    depth, photo = _scene_data()
    p = Project()
    assert np.array_equal(render(p, depth, photo), render(p, depth, None))
    p.composite = Composite(mode="normal", opacity=1.0)
    assert np.allclose(render(p, depth, photo), photo / 255.0, atol=1e-6)


def test_unknown_keys_are_ignored_for_forward_compatibility():
    data = _full_project().to_dict()
    data["campo_futuro"] = 123
    data["composite"]["otro_campo"] = "x"
    assert Project.from_dict(data).composite.mode == "soft_light"
