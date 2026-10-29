"""Migración de la carpeta de datos del nombre anterior de la app al nuevo."""

from deepshot.paths import _migrate_legacy


def test_migrate_legacy_moves_folder(tmp_path):
    old, new = tmp_path / "Antigua", tmp_path / "Nueva"
    (old / "models").mkdir(parents=True)
    (old / "models" / "marca.complete.json").write_text("{}", encoding="utf-8")
    _migrate_legacy(old, new)
    assert not old.exists()
    assert (new / "models" / "marca.complete.json").exists()   # los modelos se conservan


def test_migrate_legacy_keeps_existing_new_folder(tmp_path):
    old, new = tmp_path / "Antigua", tmp_path / "Nueva"
    old.mkdir()
    new.mkdir()
    _migrate_legacy(old, new)                                   # no pisa la carpeta nueva
    assert old.exists() and new.exists()
