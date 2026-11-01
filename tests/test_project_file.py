"""Archivo de proyecto (.deepshot): guardar, abrir, fotos movidas y archivos dañados."""

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from deepshot import project_file as pf


def make_photo(path, size=(40, 30), color=(80, 90, 100)):
    Image.new("RGB", size, color).save(path)
    return path


def entry(photo, depth=None, project=None, mask=None):
    return pf.entry_for_photo(photo, project, depth, "external", None) if mask is None else \
        pf.PhotoEntry(str(photo), photo.name, photo.stat().st_size, pf.sha256_of(photo), "external", project, depth, mask)


def test_roundtrip_keeps_depth_settings_mask_and_current(tmp_path):
    a, b = make_photo(tmp_path / "a.png"), make_photo(tmp_path / "b.png", color=(1, 2, 3))
    depth = np.linspace(0, 1, 40 * 30, dtype=np.float32).reshape(30, 40)
    settings = {"base": "ramp", "composite": {"mask": "painted", "painted_mask": "x.png"}}
    target = tmp_path / "trabajo.deepshot"
    pf.save_project(target, [entry(a, depth, settings, mask=b"PNGDATA"), entry(b)], current=1, app_version="9.9")
    loaded = pf.load_project(target)
    assert loaded.current == 1 and loaded.app_version == "9.9" and len(loaded.photos) == 2
    first, second = loaded.photos
    assert first.project == settings and first.mask == b"PNGDATA" and first.source == "external"
    assert np.abs(first.depth - depth).max() < 2e-5                          # PNG de 16 bits: error < 1/65535
    assert second.depth is None and second.project is None and second.mask is None
    assert first.sha256 == pf.sha256_of(a) and first.size == a.stat().st_size


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    a = make_photo(tmp_path / "a.png")
    target = tmp_path / "p.deepshot"
    pf.save_project(target, [entry(a)])
    pf.save_project(target, [entry(a), entry(a)])                           # sobrescribir
    assert len(pf.load_project(target).photos) == 2
    assert [p.name for p in tmp_path.glob("*.tmp")] == []


def test_locate_photo_original_next_to_project_missing_and_changed(tmp_path):
    a = make_photo(tmp_path / "a.png")
    e = entry(a)
    proj = tmp_path / "p.deepshot"
    assert pf.locate_photo(e, proj) == (a, False)                           # ruta original
    moved = tmp_path / "otra"
    moved.mkdir()
    a.rename(moved / "a.png")
    assert pf.locate_photo(e, moved / "p.deepshot") == (moved / "a.png", False)   # junto al proyecto
    assert pf.locate_photo(e, proj) == (None, False)                         # no está en ningún sitio
    make_photo(moved / "a.png", color=(250, 0, 0))                           # mismo nombre, contenido distinto
    found, changed = pf.locate_photo(e, moved / "p.deepshot")
    assert found == moved / "a.png" and changed


@pytest.mark.parametrize("content", [b"esto no es un zip", b""])
def test_invalid_files_raise_project_error(tmp_path, content):
    bad = tmp_path / "x.deepshot"
    bad.write_bytes(content)
    with pytest.raises(pf.ProjectFileError):
        pf.load_project(bad)
    with pytest.raises(pf.ProjectFileError):
        pf.load_project(tmp_path / "no_existe.deepshot")


def test_zip_without_manifest_or_from_a_newer_version_is_refused(tmp_path):
    no_manifest = tmp_path / "a.deepshot"
    with zipfile.ZipFile(no_manifest, "w") as z:
        z.writestr("otra_cosa.txt", "hola")
    with pytest.raises(pf.ProjectFileError, match="manifiesto"):
        pf.load_project(no_manifest)
    newer = tmp_path / "b.deepshot"
    with zipfile.ZipFile(newer, "w") as z:
        z.writestr(pf.MANIFEST, json.dumps({"deepshot_project": pf.PROJECT_VERSION + 1, "photos": []}))
    with pytest.raises(pf.ProjectFileError, match="más reciente"):
        pf.load_project(newer)


def test_manifest_pointing_to_missing_members_does_not_crash(tmp_path):
    odd = tmp_path / "c.deepshot"
    manifest = {"deepshot_project": 1, "current": 7,
                "photos": [{"path": "z.png", "depth": "depth/0.png", "mask": "../../evil.png"}]}
    with zipfile.ZipFile(odd, "w") as z:
        z.writestr(pf.MANIFEST, json.dumps(manifest))
    loaded = pf.load_project(odd)
    assert loaded.photos[0].depth is None and loaded.photos[0].mask is None      # nada se lee fuera del zip
    assert loaded.current == 0                                                    # índice fuera de rango: se acota


def test_photo_size_applies_exif_rotation(tmp_path):
    """Una foto de móvil en vertical (guardada apaisada + etiqueta EXIF) mide lo mismo que la ve la app."""
    from PIL import Image

    img = Image.new("RGB", (40, 20), "gray")
    exif = Image.Exif()
    exif[0x0112] = 6                                   # girar 90°: la app la ve de 20 × 40
    path = tmp_path / "vertical.jpg"
    img.save(path, exif=exif)
    assert pf.photo_size(path) == (20, 40)
    plain = tmp_path / "plain.png"
    img.save(plain)
    assert pf.photo_size(plain) == (40, 20)


def test_roundtrip_keeps_pre_and_post_fx_with_their_map_limits(tmp_path):
    from deepshot.look import LookLayer
    from deepshot.project import Project

    photo = make_photo(tmp_path / "a.png")
    pre = LookLayer.new("saturation")
    pre.params["amount"] = -1.0
    pre.limit, pre.mask_id, pre.mask_invert = "mask", "subject", True
    post = LookLayer.new("levels")
    post.params["black"] = 0.2
    post.limit, post.mask_id = "mask", "subject"
    project = Project(pre=[pre], look=[post])
    target = tmp_path / "t.deepshot"
    pf.save_project(target, [entry(photo, None, project.to_dict())], current=0, app_version="1")
    back = Project.from_dict(pf.load_project(target).photos[0].project)
    assert [(l.kind, l.limit, l.mask_id, l.mask_invert) for l in back.pre] == [("saturation", "mask", "subject", True)]
    assert [(l.kind, l.limit, l.mask_id, l.mask_invert) for l in back.look] == [("levels", "mask", "subject", False)]
    assert back.pre[0].params["amount"] == -1.0 and back.look[0].params["black"] == 0.2


def test_project_can_carry_its_photos_and_open_without_the_originals(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "datos"))
    a = make_photo(tmp_path / "a.png", color=(5, 6, 7))
    target = tmp_path / "con fotos.deepshot"
    pf.save_project(target, [entry(a)], embed_photos=False)
    assert pf.load_project(target).photos[0].embedded == ""                  # por defecto no se copian
    pf.save_project(target, [entry(a)], embed_photos=True)
    e = pf.load_project(target).photos[0]
    assert e.embedded.startswith("photo/0_")
    original = a.read_bytes()
    a.unlink()                                                              # el proyecto se abre en un equipo sin la foto
    assert pf.locate_photo(e, target) == (None, False)
    copy = pf.extract_embedded(target, e)
    assert copy is not None and copy.read_bytes() == original and copy.name == "a.png"
    assert pf.extract_embedded(target, e) == copy                           # una segunda vez se reutiliza
    e.sha256 = "0" * 64                                                     # huella que no coincide: no se da por buena
    assert pf.extract_embedded(target, e) is None
