"""Presets de fábrica: los de mapas de profundidad «de calle»."""

import pytest

from deepshot import presets


def test_street_depth_map_presets_are_dark_near_and_hazy_far():
    """Los presets «de calle»: cerca casi negro, lejos gris claro (nunca blanco puro) y siempre creciente."""
    import numpy as np

    from deepshot.project import Project, render

    depth = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (8, 1))               # 0 = cerca → 1 = lejos
    for key in ("depth_street", "depth_street_detail", "depth_soft", "depth_fog"):
        data = presets.apply_to_project_dict("style", Project().to_dict(), presets.builtin("style", key))
        row = render(Project.from_dict(data), depth, None)[0, :, 0]
        assert row[0] < 0.1 and row[-1] > 0.85, key
        assert (np.diff(row) >= -1e-4).all(), key                                  # sin inversiones de tono
    for key in ("street", "soft_map", "dense_fog"):
        assert presets.builtin("ramp", key)["ramp"]["stops"]
    assert {"street", "soft_map", "dense_fog"} <= {k for k, _ in presets.builtin_names("ramp")}


def test_lens_presets_only_change_the_bokeh_character_and_turn_the_blur_on():
    from deepshot.project import Project

    names = dict(presets.builtin_names("lens"))
    assert {"portrait", "portrait_classic", "macro", "cinema", "hard_discs", "landscape"} <= set(names)
    base = Project().to_dict()
    base["lens"].update(focus=0.62, sharp_range=0.11, target="photo", enabled=False)
    for key in names:
        data = presets.builtin("lens", key)
        assert set(data) == {"lens"} and set(data["lens"]) <= set(presets.LENS_PRESET_FIELDS)
        out = presets.apply_to_project_dict("lens", base, data)
        assert out["lens"]["enabled"] is True
        assert out["lens"]["focus"] == 0.62 and out["lens"]["target"] == "photo"      # lo de la foto no se toca
        for field, value in data["lens"].items():
            assert out["lens"][field] == value
        import json

        assert {k: out[k] for k in out if k != "lens"} == json.loads(json.dumps({k: base[k] for k in base if k != "lens"}))
    assert presets.builtin("lens", "portrait_classic")["lens"]["cat_eye"] > 0
    with pytest.raises(KeyError):
        presets.builtin("lens", "no_existe")


def test_lens_preset_saved_by_the_user_roundtrips_without_focus_or_target(tmp_path, monkeypatch):
    from deepshot.project import Project

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path))
    project = Project()
    project.lens.aperture, project.lens.cat_eye, project.lens.focus = 1.6, 0.4, 0.9
    data = presets.from_project("lens", project)
    assert set(data["lens"]) == set(presets.LENS_PRESET_FIELDS) and "focus" not in data["lens"]
    presets.save_user("lens", "Mi bokeh", data)
    assert "Mi bokeh" in presets.list_user("lens")
    loaded = presets.load_user("lens", "Mi bokeh")
    out = presets.apply_to_project_dict("lens", Project().to_dict(), loaded)
    assert out["lens"]["aperture"] == 1.6 and out["lens"]["cat_eye"] == 0.4 and out["lens"]["focus"] == 0.35
    presets.delete_user("lens", "Mi bokeh")


# --- Áreas que trae un estilo ------------------------------------------------------------------------------
def _apply(project_dict, key):
    return presets.apply_to_project_dict("style", project_dict, presets.builtin("style", key))


def _names(project_dict):
    return [a["name"] for a in project_dict["areas"]]


def test_style_area_disappears_when_another_style_without_areas_is_applied():
    from deepshot.project import Project

    base = Project().to_dict()
    with_area = _apply(base, "studio_light")
    assert _names(with_area) == ["Foco"] and with_area["areas"][0]["from_style"]       # marcada como de estilo
    without = _apply(with_area, "haze")
    assert without["areas"] == []                                                       # el fallo de Juanjo: ya no queda
    again = _apply(without, "studio_light")
    assert _names(again) == ["Foco"]                                                    # y vuelve a poder aplicarse


def test_hand_made_areas_survive_style_changes_and_new_style_areas_are_added_next_to_them():
    from deepshot.project import Project

    base = Project().to_dict()
    base["areas"] = [{"name": "Mía", "shape": "ellipsoid", "center": [0.1, 0.2, 0.3], "scale": [0.5, 0.5, 0.5]}]
    first = _apply(base, "studio_light")
    assert _names(first) == ["Mía", "Foco"]                                             # la tuya se queda; la del estilo, al lado
    second = _apply(first, "haze")
    assert _names(second) == ["Mía"] and not second["areas"][0].get("from_style")
    third = _apply(_apply(second, "studio_light"), "studio_light")                      # el mismo estilo dos veces: no se duplica
    assert _names(third) == ["Mía", "Foco"]


def test_editing_a_style_area_makes_it_the_users_and_it_is_kept():
    from deepshot.project import Project

    with_area = _apply(Project().to_dict(), "studio_light")
    with_area["areas"][0]["strength"] = 0.3                                             # el usuario la edita
    after = _apply(with_area, "haze")
    assert _names(after) == ["Foco"] and after["areas"][0]["strength"] == 0.3 and after["areas"][0]["from_style"] == ""
    assert _names(_apply(after, "contrast")) == ["Foco"]                                # y ya no se quita nunca
    renamed = _apply(Project().to_dict(), "studio_light")
    renamed["areas"][0]["name"] = "Luz principal"                                       # hasta cambiar el nombre cuenta
    assert _names(_apply(renamed, "haze")) == ["Luz principal"]


def test_two_styles_with_areas_replace_each_other_and_old_projects_without_the_mark_keep_theirs():
    from deepshot.project import Project

    custom = {"ramp": presets.builtin("ramp", "linear")["ramp"], "base": "black",
              "areas": [{"name": "Otra", "center": [0.0, 0.0, 0.2]}]}
    with_foco = _apply(Project().to_dict(), "studio_light")
    swapped = presets.apply_to_project_dict("style", with_foco, custom)
    assert _names(swapped) == ["Otra"]                                                  # la del estilo anterior sale
    assert swapped["areas"][0]["scale"] == [0.4, 0.4, 0.4]                              # y la nueva viene completa
    old = Project().to_dict()
    old["areas"] = [{k: v for k, v in a.items() if k != "from_style"} for a in with_foco["areas"]]   # proyecto antiguo
    assert _names(_apply(old, "haze")) == ["Foco"]                                      # sin marca: es del usuario


def test_area_mark_survives_json_projects_and_is_not_saved_inside_styles(tmp_path):
    from deepshot.project import Project

    marked = Project.from_dict(_apply(Project().to_dict(), "studio_light"))
    assert marked.areas[0].from_style
    path = tmp_path / "p.json"
    marked.save(path)
    assert Project.load(path).areas[0].from_style == marked.areas[0].from_style
    legacy = Project.from_dict({"areas": [{"name": "Vieja", "center": [0, 0, 0.5]}]})
    assert legacy.areas[0].from_style == ""
    saved = presets.from_project("style", marked)
    assert all("from_style" not in a for a in saved["areas"])                           # el estilo guardado va limpio
    reapplied = _apply(Project().to_dict(), "haze")
    reapplied = presets.apply_to_project_dict("style", reapplied, saved)
    assert reapplied["areas"][0]["from_style"] and _names(reapplied) == ["Foco"]


# --- Importar y exportar estilos (compartir con otras personas) ---------------------------------------------
def _style_data():
    return presets.builtin("style", "depth_bands")


def test_style_roundtrips_through_export_and_import_and_name_clashes_get_a_number(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    presets.save_user("style", "Mi estilo", _style_data())
    shared = tmp_path / "mi.deepshotstyle"
    presets.export_user("style", "Mi estilo", shared)
    assert presets.load_file(shared, "style")["name"] == "Mi estilo"
    assert presets.import_user("style", shared) == "Mi estilo (2)"             # ya había uno: se conservan los dos
    assert presets.import_user("style", shared, "Mi estilo") == "Mi estilo"    # o se sustituye
    assert presets.list_user("style") == ["Mi estilo", "Mi estilo (2)"]
    assert presets.load_user("style", "Mi estilo (2)")["bands"] == presets.load_user("style", "Mi estilo")["bands"]


def test_importing_rejects_files_that_are_not_valid_presets_of_that_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    bad = tmp_path / "raro.json"
    bad.write_text("no es json", encoding="utf-8")
    with pytest.raises(ValueError):
        presets.read_importable(bad, "style")
    other = tmp_path / "rampa.json"
    presets.save_file(other, "ramp", "Una rampa", presets.builtin("ramp", "haze"))
    with pytest.raises(ValueError):
        presets.read_importable(other, "style")                                # es de otro tipo
    broken = tmp_path / "roto.json"
    broken.write_text('{"deepshot_preset": 1, "kind": "style", "ramp": {"stops": "x"}}', encoding="utf-8")
    with pytest.raises(ValueError):
        presets.read_importable(broken, "style")
    assert presets.list_user("style") == []


def test_preset_bar_imports_a_shared_style_and_it_appears_in_the_list(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from deepshot.app import create_app
    from deepshot.ui.preset_bar import PresetBar

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    create_app([])
    shared = tmp_path / "ajeno.deepshotstyle"
    presets.save_file(shared, "style", "Estilo de un amigo", _style_data())
    bar = PresetBar("style", "Cargar estilo…", lambda: _style_data())
    assert bar.import_file(str(shared)) == "Estilo de un amigo"
    assert any(bar.combo.itemData(i) == ("user", "Estilo de un amigo") for i in range(bar.combo.count()))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)                   # la pregunta de «ya existe» se cierra: cancela
    assert bar.import_file(str(shared)) is None
    assert presets.list_user("style") == ["Estilo de un amigo"]
    assert bar.export_file("Estilo de un amigo", str(tmp_path / "salida.deepshotstyle"))
    assert (tmp_path / "salida.deepshotstyle").is_file()
