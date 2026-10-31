"""Reencuadre previo: el rectángulo, la lectura de la foto recortada y la caché de profundidad por encuadre."""

import numpy as np
from PIL import Image

from deepshot import depth, framing


def test_normalize_orders_clamps_and_ignores_full_or_tiny_crops():
    assert framing.normalize(None) is None
    assert framing.normalize((0, 0, 1, 1)) is None                          # toda la foto = sin encuadre
    assert framing.normalize((0.5, 0.5, 0.505, 0.9)) is None                # demasiado fino
    assert framing.normalize((0.8, 0.9, 0.2, 0.1)) == (0.2, 0.1, 0.8, 0.9)   # se ordena
    assert framing.normalize((-1, -1, 0.5, 2)) == (0.0, 0.0, 0.5, 1.0)       # se recorta al rango
    assert framing.normalize("x") is None


def test_box_size_and_aspect():
    box = framing.to_box((0.25, 0.5, 0.75, 1.0), (400, 200))
    assert box == (100, 100, 300, 200) and framing.cropped_size((0.25, 0.5, 0.75, 1.0), (400, 200)) == (200, 100)
    assert framing.to_box(None, (400, 200)) == (0, 0, 400, 200)
    crop = framing.with_aspect(None, 1.0, (400, 200))                       # cuadrado dentro de 400×200
    w, h = framing.cropped_size(crop, (400, 200))
    assert abs(w - h) <= 1 and h == 200
    sq = framing.with_aspect((0.0, 0.0, 0.5, 0.5), 16 / 9, (1600, 900))
    w, h = framing.cropped_size(sq, (1600, 900))
    assert abs(w / h - 16 / 9) < 0.02 and 0 <= sq[0] and sq[2] <= 1


def test_load_photo_crops_without_touching_the_file_and_the_cache_key_follows_the_crop(tmp_path):
    path = tmp_path / "foto.png"
    arr = np.zeros((100, 200, 3), np.uint8)
    arr[:, 100:] = 255                                                      # mitad izquierda negra, derecha blanca
    Image.fromarray(arr).save(path)
    right = depth.load_photo(path, (0.5, 0.0, 1.0, 1.0))
    assert right.size == (100, 100) and np.asarray(right).min() == 255
    assert depth.load_photo(path).size == (200, 100)                        # sin encuadre, la foto entera
    assert Image.open(path).size == (200, 100)                              # el archivo original no cambia
    plain = depth.DepthSettings()
    assert "crop" not in plain.key()                                        # las cachés de antes siguen valiendo
    framed = depth.DepthSettings(crop=(0.5, 0.0, 1.0, 1.0))
    assert framed.key() != plain.key() and framed.raw_key() != plain.raw_key()
    assert depth.cache_path(path, framed) != depth.cache_path(path, plain)


def test_photo_info_reads_a_plain_file_without_failing(tmp_path):
    from deepshot import photo_info

    path = tmp_path / "a.png"
    Image.new("RGB", (30, 20)).save(path)
    info = photo_info.read(path)
    assert (info.width, info.height, info.megapixels) == (30, 20, 0.0) and info.camera == "" and info.gps == ""
    missing = photo_info.read(tmp_path / "no.jpg")                          # archivo que no existe: vacío, sin error
    assert missing.width == 0 and missing.name == "no.jpg"
    assert photo_info.human_size(2048) == "2.0 KB"


def test_project_file_keeps_the_crop_per_photo(tmp_path):
    from deepshot import project_file

    entry = project_file.PhotoEntry(path="/x/a.jpg", crop=[0.1, 0.2, 0.6, 0.9])
    target = tmp_path / "p.deepshot"
    project_file.save_project(target, [entry, project_file.PhotoEntry(path="/x/b.jpg")])
    loaded = project_file.load_project(target)
    assert loaded.photos[0].crop == [0.1, 0.2, 0.6, 0.9] and loaded.photos[1].crop is None


def test_crop_dialog_applies_proportions_and_resets():
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from deepshot.ui.crop_dialog import CropDialog

    app = QApplication.instance() or QApplication([])
    image = QImage(400, 200, QImage.Format_RGB888)
    dialog = CropDialog(image, None)
    assert dialog.crop() is None                                            # sin tocar nada: foto entera
    dialog.aspect_combo.setCurrentIndex(dialog.aspect_combo.findData("1:1"))
    crop = dialog.crop()
    w, h = framing.cropped_size(crop, (400, 200))
    assert abs(w - h) <= 1 and h == 200                                     # cuadrado de 200 × 200
    dialog.aspect_combo.setCurrentIndex(dialog.aspect_combo.findData("16:9"))
    w, h = framing.cropped_size(dialog.crop(), (400, 200))
    assert abs(w / h - 16 / 9) < 0.03
    assert "px" in dialog.label.text()
    dialog._reset()
    assert dialog.crop() is None
    dialog.deleteLater()
    app.processEvents()


def test_discovered_runs_follow_the_crop(tmp_path, monkeypatch):
    from deepshot import depth_runs

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    photo = tmp_path / "f.png"
    Image.new("RGB", (40, 30)).save(photo)
    crop = (0.1, 0.1, 0.6, 0.6)
    plain = depth.settings_for(model="marigold-v1-0", quality="fast")
    depth.cache_path(photo, plain).write_bytes(b"x")                         # un mapa de la foto entera en la caché
    assert len(depth_runs.discover(photo, ["marigold-v1-0"])) == 1
    assert depth_runs.discover(photo, ["marigold-v1-0"], crop) == []          # con encuadre no se ofrece el mapa de la foto entera
    from dataclasses import replace

    depth.cache_path(photo, replace(plain, crop=crop)).write_bytes(b"x")
    found = depth_runs.discover(photo, ["marigold-v1-0"], crop)
    assert len(found) == 1 and found[0].settings.crop == crop
