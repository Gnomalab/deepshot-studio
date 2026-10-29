"""Temas de color, aspecto de los controles y plantillas de disposición."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from deepshot import themes  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))


def luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_every_factory_theme_has_every_color_as_valid_hex_and_readable_text():
    assert set(themes.FACTORY_THEMES) == {"dark", "light", "midnight"}
    for key, theme in themes.FACTORY_THEMES.items():
        assert set(theme.colors) >= set(themes.COLOR_KEYS), key
        assert all(themes.HEX.match(theme.colors[k]) for k in themes.COLOR_KEYS), key
        assert contrast(theme.color("text"), theme.color("window")) >= 7, key          # texto principal
        assert contrast(theme.color("text"), theme.color("base")) >= 7, key             # y en los campos
        assert contrast(theme.color("text_dim"), theme.color("window")) >= 4.5, key     # textos de ayuda
        assert contrast(theme.color("text_dim"), theme.color("panel")) >= 3.5, key      # ... sobre las secciones
        assert contrast(theme.color("on_accent"), theme.color("accent")) >= 3.5, key    # texto sobre el acento
        assert contrast(theme.color("icon_idle"), theme.color("window")) >= 3, key      # iconos en reposo
        assert contrast(theme.color("icon_active"), theme.color("button")) >= 3, key     # ... y activos


def test_qss_is_fully_filled_for_every_theme_and_appearance_and_changes_with_them():
    dark_default = themes.build_qss(themes.FACTORY_THEMES["dark"])
    assert "$" not in dark_default and "#303030" in dark_default and "#4772b3" in dark_default
    assert "border-radius: 3px" in dark_default
    for key, theme in themes.FACTORY_THEMES.items():
        for density in themes.DENSITIES:
            for radius in (0, 3, 10):
                qss = themes.build_qss(theme, themes.Appearance(radius=radius, density=density))
                assert "$" not in qss, (key, density, radius)
                assert theme.color("window") in qss and theme.color("accent") in qss
                assert f"border-radius: {radius}px" in qss
    light = themes.build_qss(themes.FACTORY_THEMES["light"])
    assert "#303030" not in light and "#dcdcdc" in light
    compact = themes.tokens(themes.FACTORY_THEMES["dark"], themes.Appearance(density="compact"))
    comfortable = themes.tokens(themes.FACTORY_THEMES["dark"], themes.Appearance(density="comfortable"))
    assert int(compact["pad_v"]) < int(comfortable["pad_v"]) and int(compact["min_h"]) < int(comfortable["min_h"])


def test_appearance_values_are_clamped_and_bad_ones_replaced():
    assert themes.Appearance(radius=99).validated().radius == 10 and themes.Appearance(radius=-4).validated().radius == 0
    assert themes.Appearance(density="raro").validated().density == "standard"
    assert themes.Appearance().icons == "color" and themes.Appearance(icons="raro").validated().icons == "color"
    assert themes.Appearance(icons="mono").validated().icons == "mono"


def test_every_icon_has_a_color_version_with_a_category_and_valid_svg():
    from PySide6.QtCore import QByteArray
    from PySide6.QtSvg import QSvgRenderer

    from deepshot.ui import icons, icons_color

    assert set(icons.available()) == set(icons_color.BUILDERS) == set(icons_color.CATEGORY)
    assert set(icons_color.CATEGORY.values()) <= set(icons_color.PALETTE_DARK) == set(icons_color.PALETTE_LIGHT)
    for name in icons.available():
        for dark in (True, False):
            assert QSvgRenderer(QByteArray(icons_color.svg(name, dark))).isValid(), name


def test_user_themes_partial_colors_bad_values_and_broken_files(tmp_path):
    folder = themes.themes_dir()
    (folder / "rojo.json").write_text(json.dumps({"name": "Rojo", "base": "light",
                                                    "colors": {"accent": "#c0392b", "window": "no es un color",
                                                               "inventado": "#123456"}}), encoding="utf-8")
    (folder / "roto.json").write_text("{ esto no es json", encoding="utf-8")
    (folder / "lista.json").write_text("[1, 2, 3]", encoding="utf-8")
    found = {t.label: t for t in themes.user_themes()}
    assert set(found) == {"Rojo"}                                              # los rotos se ignoran sin romper la lista
    red = found["Rojo"]
    assert red.color("accent") == "#c0392b" and not red.factory and red.key == "user:rojo"
    assert red.color("window") == themes.LIGHT["window"]                       # color inválido: el de su tema base
    assert "inventado" not in red.colors
    assert themes.get_theme("user:rojo") is not None and themes.get_theme("user:rojo").label == "Rojo"
    assert themes.get_theme("no existe").key == "dark"                         # tema que ya no está: el de por defecto
    path = themes.save_user_theme("Verde", "dark", {"accent": "#2e8b57", "text": "xx"})
    assert json.loads(path.read_text(encoding="utf-8"))["colors"] == {"accent": "#2e8b57"}
    assert {t.label for t in themes.all_themes()} >= {"Verde", "Rojo"}
    with pytest.raises(ValueError):
        themes.save_user_theme("   ", "dark", {})
    assert "$" not in themes.build_qss(themes.get_theme("user:rojo"))
