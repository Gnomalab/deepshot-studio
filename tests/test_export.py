"""Exportación: PNG 8 / PNG 16 / EXR 32, efecto o mapa de profundidad."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepshot import export as ex
from deepshot.areas import Area
from deepshot.depth import load_photo
from deepshot.project import Project, render


def read_png16(path: Path) -> np.ndarray:
    """Lector independiente de PNG de 16 bits sin filtros (como los que escribe write_png16)."""
    raw = Path(path).read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, ihdr = 8, b"", None
    while pos < len(raw):
        (n,) = struct.unpack(">I", raw[pos:pos + 4])
        tag, payload = raw[pos + 4:pos + 8], raw[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", raw[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(tag + payload) & 0xFFFFFFFF
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + n
    w, h, depth, ctype, *_ = ihdr
    assert depth == 16
    channels = 1 if ctype == 0 else 3
    rows = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * channels * 2)
    assert (rows[:, 0] == 0).all()
    data = rows[:, 1:].copy().view(">u2").astype(np.uint16)
    return data.reshape(h, w) if channels == 1 else data.reshape(h, w, 3)


def test_png16_gray_is_exact_and_opens_in_pillow(tmp_path):
    data = (np.arange(40 * 70).reshape(40, 70) * 11 % 65536).astype(np.uint16)
    ex.write_png16(tmp_path / "g.png", data)
    assert np.array_equal(read_png16(tmp_path / "g.png"), data)
    with Image.open(tmp_path / "g.png") as img:
        assert img.size == (70, 40) and img.mode.startswith("I")
        assert np.array_equal(np.asarray(img).astype(np.uint16), data)


def test_png16_rgb_is_exact_with_many_row_blocks(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 65536, size=(200, 33, 3), dtype=np.uint16)    # >64 filas: varios bloques IDAT
    ex.write_png16(tmp_path / "c.png", data)
    assert np.array_equal(read_png16(tmp_path / "c.png"), data)
    with Image.open(tmp_path / "c.png") as img:
        assert img.size == (33, 200)


def test_png16_rejects_wrong_arrays(tmp_path):
    with pytest.raises(ValueError):
        ex.write_png16(tmp_path / "x.png", np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        ex.write_png16(tmp_path / "x.png", np.zeros((4, 4, 4), dtype=np.uint16))


def test_save_image_all_formats_roundtrip(tmp_path):
    gray = np.tile(np.linspace(0, 1, 50, dtype=np.float32), (30, 1))
    color = np.stack([gray, 1 - gray, gray * 0.5], -1)
    ex.save_image(gray, tmp_path / "g8.png", "png8")
    with Image.open(tmp_path / "g8.png") as img:
        assert img.mode == "L" and abs(int(np.asarray(img)[0, 25]) - round(gray[0, 25] * 255)) <= 1
    ex.save_image(np.repeat(gray[..., None], 3, -1), tmp_path / "g8rgb.png", "png8")   # gris guardado como gris
    with Image.open(tmp_path / "g8rgb.png") as img:
        assert img.mode == "L"
    ex.save_image(color, tmp_path / "c8.png", "png8")
    with Image.open(tmp_path / "c8.png") as img:
        assert img.mode == "RGB"
    ex.save_image(color, tmp_path / "c16.png", "png16")
    assert np.abs(read_png16(tmp_path / "c16.png") / 65535.0 - color).max() < 1e-4
    ex.save_image(gray, tmp_path / "g16.png", "png16")
    assert read_png16(tmp_path / "g16.png").ndim == 2

    import OpenEXR

    hdr = color * 3.0 - 0.5                                   # en EXR se conserva el valor sin recortar
    ex.save_image(hdr, tmp_path / "c.exr", "exr32")
    back = OpenEXR.File(str(tmp_path / "c.exr")).channels()["RGB"].pixels
    assert back.dtype == np.float32 and np.abs(back - hdr).max() == 0
    ex.save_image(gray, tmp_path / "z.exr", "exr32")
    assert OpenEXR.File(str(tmp_path / "z.exr")).channels()["Z"].pixels.shape == gray.shape
    with pytest.raises(ValueError):
        ex.save_image(gray, tmp_path / "x.bmp", "bmp")


def test_free_path_does_not_overwrite(tmp_path):
    first = ex.free_path(tmp_path, "foto", ".png")
    first.write_bytes(b"x")
    assert ex.free_path(tmp_path, "foto", ".png").name == "foto_2.png"
    (tmp_path / "foto_2.png").write_bytes(b"x")
    assert ex.free_path(tmp_path, "foto", ".png").name == "foto_3.png"
    assert ex.free_path(tmp_path, 'a/b:c*', ".png").name == "a_b_c_.png"      # caracteres no válidos en Windows


def make_scene(tmp_path):
    photo = tmp_path / "foto original.png"
    arr = np.zeros((60, 80, 3), dtype=np.uint8)
    arr[..., 0] = 200
    Image.fromarray(arr).save(photo)
    depth = np.tile(np.linspace(0.05, 0.95, 80, dtype=np.float32), (60, 1))
    project = Project()
    project.areas.append(Area(center=(0.2, 0.0, 0.3), scale=(0.3, 0.3, 0.3)))
    return photo, depth, project


@pytest.mark.parametrize("fmt", ["png8", "png16", "exr32"])
def test_export_effect_matches_render_and_writes_settings(tmp_path, fmt):
    photo, depth, project = make_scene(tmp_path)
    opts = ex.ExportOptions("effect", fmt, tmp_path / "salida nueva", "foto original_deepshot")
    steps = []
    written = ex.export_effect(project, depth, photo, opts, progress=lambda m, f: steps.append(f))
    assert [p.suffix for p in written] == [ex.EXTENSIONS[fmt], ".json"] and all(p.is_file() for p in written)
    assert steps == sorted(steps) and steps[-1] == 1.0
    expected = render(project, depth, photo)
    if fmt == "exr32":
        import OpenEXR

        got = OpenEXR.File(str(written[0])).channels()["RGB"].pixels
        assert np.abs(got - expected).max() < 1e-6
    elif fmt == "png16":
        got = read_png16(written[0])
        got = got.astype(np.float32) / 65535.0
        assert np.abs((got if got.ndim == 3 else got[..., None]) - (expected if got.ndim == 3 else expected[..., :1])).max() < 2e-5
    else:
        with Image.open(written[0]) as img:
            got = np.asarray(img).astype(np.float32) / 255.0
        ref = expected[..., 0] if got.ndim == 2 else expected
        assert np.abs(got - ref).max() <= 0.5 / 255 + 1e-6
    assert Project.load(written[1]).to_dict() == project.to_dict()            # los ajustes se pueden recargar


def test_export_effect_without_settings_file_and_with_painted_mask(tmp_path):
    photo, depth, project = make_scene(tmp_path)
    project.composite.opacity, project.composite.mask = 1.0, "painted"
    mask = np.zeros((30, 40), dtype=np.uint8)                       # máscara a otra resolución: se reescala
    mask[:, :20] = 255
    opts = ex.ExportOptions("effect", "png8", tmp_path, "m", save_settings=False)
    written = ex.export_effect(project, depth, photo, opts, painted_mask=mask)
    assert len(written) == 1
    with Image.open(written[0]) as img:
        arr = np.asarray(img.convert("RGB"))
    assert arr[30, 5, 0] > 190 and arr[30, 5, 1] < 20               # lado pintado: la foto (roja)
    assert abs(int(arr[30, 75, 0]) - int(arr[30, 75, 1])) < 3       # lado sin pintar: gris de la rampa


def test_export_depth_near_bright_and_inverted(tmp_path):
    depth = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (10, 1))
    near = ex.export_depth(depth, ex.ExportOptions("depth", "png16", tmp_path, "prof"))
    arr = read_png16(near)
    assert arr[0, 0] == 65535 and arr[0, -1] == 0                  # cerca (0) = blanco
    far = ex.export_depth(depth, ex.ExportOptions("depth", "png16", tmp_path, "prof", near_bright=False))
    assert far.name == "prof_2.png" and read_png16(far)[0, 0] == 0
    exr = ex.export_depth(depth, ex.ExportOptions("depth", "exr32", tmp_path, "prof"))
    import OpenEXR

    z = OpenEXR.File(str(exr)).channels()["Z"].pixels
    assert z.shape == depth.shape and z[0, 0] == pytest.approx(1.0)


def test_invalid_options_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        ex.ExportOptions("nada", "png8", tmp_path, "x").validate()
    with pytest.raises(ValueError):
        ex.ExportOptions("effect", "tga", tmp_path, "x").validate()
    with pytest.raises(ValueError):
        ex.ExportOptions("effect", "png8", tmp_path, "  ").validate()


# --- Mapa de profundidad: original o con las zonas movidas ---------------------------------------------------
def moved_control():
    from deepshot.depth_control import DepthControl

    control = DepthControl(enabled=True)
    control.near.center, control.near.target = 0.1, 0.3                    # lo cercano se lleva hacia el medio
    return control


@pytest.mark.parametrize("fmt", ["png8", "png16", "exr32"])
def test_export_depth_original_or_with_moved_zones(tmp_path, fmt):
    depth = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (10, 1))
    control = moved_control()
    moved = control.apply(depth)
    assert not np.allclose(moved, depth)                                   # el control de la prueba sí cambia el mapa

    def load(path):
        if fmt == "png8":
            return np.asarray(Image.open(path), dtype=np.float32) / 255.0
        if fmt == "png16":
            return read_png16(path).astype(np.float32) / 65535.0
        import OpenEXR

        return OpenEXR.File(str(path)).channels()["Z"].pixels

    tol = {"png8": 1 / 255, "png16": 1 / 65535, "exr32": 1e-6}[fmt]
    original = ex.export_depth(depth, ex.ExportOptions("depth", fmt, tmp_path, "o", near_bright=False),
                               depth_control=control)                      # depth_zones apagado: mapa original
    assert np.abs(load(original) - depth).max() <= tol + 1e-6
    zones = ex.export_depth(depth, ex.ExportOptions("depth", fmt, tmp_path, "z", near_bright=False, depth_zones=True),
                            depth_control=control)
    assert np.abs(load(zones) - moved).max() <= tol + 1e-6                 # igual a project.depth_control.apply
    bright = ex.export_depth(depth, ex.ExportOptions("depth", fmt, tmp_path, "b", depth_zones=True),
                             depth_control=control)                        # y «cerca = claro» sigue valiendo
    assert np.abs(load(bright) - (1.0 - moved)).max() <= tol + 1e-6


def test_export_depth_zones_without_control_or_with_it_off_is_the_original(tmp_path):
    from deepshot.depth_control import DepthControl

    depth = np.tile(np.linspace(0, 1, 50, dtype=np.float32), (5, 1))
    opts = ex.ExportOptions("depth", "png16", tmp_path, "a", near_bright=False, depth_zones=True)
    a = read_png16(ex.export_depth(depth, opts))                           # sin control
    opts_off = ex.ExportOptions("depth", "png16", tmp_path, "b", near_bright=False, depth_zones=True)
    b = read_png16(ex.export_depth(depth, opts_off, depth_control=DepthControl()))   # control apagado
    assert np.array_equal(a, b) and np.abs(a.astype(np.float32) / 65535.0 - depth).max() <= 1 / 65535 + 1e-6


@pytest.mark.parametrize("fmt", ["png8", "png16", "exr32"])
def test_export_depth_in_color_uses_the_palette_and_keeps_gray_default(tmp_path, fmt):
    from deepshot import depth_view as dv

    depth = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (10, 1))
    opts = ex.ExportOptions(content="depth", format=fmt, folder=tmp_path, name="color", depth_palette="rainbow")
    path = ex.export_depth(depth, opts)
    if fmt == "png8":
        got = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    elif fmt == "png16":
        got = read_png16(path).astype(np.float32) / 65535.0
    else:
        import OpenEXR

        got = OpenEXR.File(str(path)).channels()["RGB"].pixels
    assert got.shape == (10, 100, 3)                                      # tres canales, no gris
    tol = 0.01 if fmt != "exr32" else 1e-5
    assert np.allclose(got, dv.colorize_float(depth, "rainbow"), atol=tol)
    assert not np.allclose(got[..., 0], got[..., 2])                       # de verdad tiene color
    # Sin paleta sigue saliendo el mapa en gris de siempre.
    plain = ex.export_depth(depth, ex.ExportOptions(content="depth", format="png16", folder=tmp_path, name="gris"))
    assert read_png16(plain).ndim == 2
    with pytest.raises(ValueError):
        ex.export_depth(depth, ex.ExportOptions(content="depth", folder=tmp_path, name="x", depth_palette="no_existe"))


# --- Imagen final: JPG, TIFF, WebP, tamaño y metadatos --------------------------------------------------------------
def _final_setup(tmp_path):
    from deepshot.look import LookLayer

    photo = tmp_path / "foto final.jpg"
    exif = Image.Exif()
    exif[0x010F] = "Camara de prueba"                       # Make
    exif[0x0112] = 6                                        # orientación: la foto se gira al abrirla
    Image.new("RGB", (80, 60), (90, 120, 160)).save(photo, exif=exif)
    depth = np.tile(np.linspace(0.1, 0.9, 60, dtype=np.float32), (80, 1))   # (alto 80, ancho 60) tras el giro
    project = Project()
    layer = LookLayer.new("levels")
    layer.params["black"] = 0.3
    project.look = [layer]
    return photo, depth, project


@pytest.mark.parametrize("fmt", ["jpg", "tiff8", "tiff16", "png8", "png16", "webp"])
def test_export_final_writes_every_format_with_the_look(tmp_path, fmt):
    photo, depth, project = _final_setup(tmp_path)
    opts = ex.ExportOptions("final", fmt, tmp_path / "out", "final", quality=90, dpi=240)
    path = ex.export_effect(project, depth, photo, opts)[0]
    assert path.suffix == ex.EXTENSIONS[fmt] and path.exists()
    with Image.open(path) as img:
        assert img.size == (60, 80) and img.convert("RGB").size == (60, 80)
        got = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    expected = render(project, depth, load_photo(photo))
    assert np.abs(got - expected).max() < 0.08                # con el Look y, en JPG / WebP, la pérdida de la compresión
    plain = render(project, depth, load_photo(photo), with_look=False)
    assert np.abs(got - plain).max() > 0.05                   # el Look cambia la imagen


def test_export_effect_leaves_the_look_out_and_final_puts_it_in(tmp_path):
    photo, depth, project = _final_setup(tmp_path)
    effect = ex.export_effect(project, depth, photo, ex.ExportOptions("effect", "png8", tmp_path / "e", "e",
                                                                       save_settings=False))[0]
    final = ex.export_effect(project, depth, photo, ex.ExportOptions("final", "png8", tmp_path / "f", "f"))[0]
    a = np.asarray(Image.open(effect).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(final).convert("RGB"), dtype=np.int16)
    assert np.abs(a - b).max() > 10


def test_target_size_modes_keep_the_proportion_and_are_clamped():
    assert ex.target_size(4000, 3000, "original", 0) == (4000, 3000)
    assert ex.target_size(4000, 3000, "percent", 50) == (2000, 1500)
    assert ex.target_size(4000, 3000, "width", 1000) == (1000, 750)
    assert ex.target_size(4000, 3000, "height", 600) == (800, 600)
    assert ex.target_size(4000, 3000, "long_side", 2000) == (2000, 1500)
    assert ex.target_size(4000, 3000, "percent", 1_000_000) == (ex.MAX_SIDE, ex.MAX_SIDE * 3 // 4)
    assert ex.target_size(10, 10, "percent", 0.001) == (1, 1)


def test_export_final_resizes_and_stores_dpi_and_metadata(tmp_path):
    photo, depth, project = _final_setup(tmp_path)
    opts = ex.ExportOptions("final", "jpg", tmp_path / "o", "tam", size_mode="long_side", size_value=40, dpi=150)
    path = ex.export_effect(project, depth, photo, opts)[0]
    with Image.open(path) as img:
        assert img.size == (30, 40) and round(img.info["dpi"][0]) == 150
        exif = img.getexif()
        assert exif.get(0x010F) == "Camara de prueba" and exif.get(0x0112) == 1     # orientación ya aplicada: derecha
    opts2 = ex.ExportOptions("final", "jpg", tmp_path / "o", "sin", keep_metadata=False)
    with Image.open(ex.export_effect(project, depth, photo, opts2)[0]) as img:
        assert 0x010F not in img.getexif()


def test_final_format_and_size_are_validated(tmp_path):
    with pytest.raises(ValueError):
        ex.ExportOptions("final", "exr32", tmp_path, "x").validate()
    with pytest.raises(ValueError):
        ex.ExportOptions("final", "jpg", tmp_path, "x", size_mode="otro").validate()
    with pytest.raises(ValueError):
        ex.ExportOptions("final", "jpg", tmp_path, "x", size_mode="width", size_value=0).validate()


def test_tiff16_keeps_more_than_8_bits(tmp_path):
    data = np.zeros((4, 4, 3), dtype=np.uint16)
    data[..., 0] = 1000
    data[..., 1] = 40000
    data[..., 2] = 65535
    ex.write_tiff16(tmp_path / "t.tif", data, 300)
    raw = (tmp_path / "t.tif").read_bytes()
    assert raw[:4] == b"II*\x00" and np.frombuffer(raw[8:8 + 6], dtype="<u2").tolist() == [1000, 40000, 65535]


def test_export_file_names_describe_how_the_file_was_made():
    from deepshot.ui.export_panel import name_suffix

    def options(**kw):
        return ex.ExportOptions(**{"content": "depth", "format": "png16", "folder": Path("."), "name": "x", **kw})

    assert name_suffix(options(content="effect")) == "_effect"
    assert name_suffix(options(content="final", format="jpg")) == "_final"
    assert name_suffix(options()) == "_depth_gray"
    assert name_suffix(options(near_bright=False)) == "_depth_gray_inverted"
    assert name_suffix(options(depth_palette="thermal")) == "_depth_thermal"
    assert name_suffix(options(depth_palette="rainbow", depth_zones=True, near_bright=False)) == "_depth_rainbow_zones"
