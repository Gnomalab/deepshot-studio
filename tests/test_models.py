"""Gestor de modelos (sin red): instalación desde carpeta, seguridad y borrado."""

import json

import pytest

from deepshot import models as m


@pytest.fixture
def manager(tmp_path):
    return m.ModelManager(tmp_path / "models")


def _fake_marigold(folder, *, fp16=True, pickle_too=True, class_name="MarigoldDepthPipeline"):
    """Carpeta con la estructura de un modelo diffusers (pesos falsos)."""
    (folder / "unet").mkdir(parents=True)
    (folder / "model_index.json").write_text(json.dumps({"_class_name": class_name}))
    (folder / "unet" / "config.json").write_text("{}")
    (folder / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 100)
    if fp16:
        (folder / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x" * 50)
    if pickle_too:
        (folder / "unet" / "diffusion_pytorch_model.bin").write_bytes(b"pickle")
    (folder / "README.md").write_text("hola")
    return folder


def test_catalog_has_default_and_only_safe_patterns(manager):
    ids = [e.id for e in manager.all()]
    assert m.DEFAULT_MODEL in ids
    for entry in m.MODELS_CATALOG:
        # Los .pt solo se aceptan si el modelo los declara como tensores del autor (tensor_files, leídos sin ejecutar código).
        assert not any(p.endswith(tuple(m.UNSAFE_EXTENSIONS)) for p in entry.patterns if p not in entry.tensor_files)
        assert all(p.endswith(".pt") for p in entry.tensor_files)


def test_select_safe_files_prefers_fp16_and_drops_pickle():
    names = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.bin",
        "vae/diffusion_pytorch_model.safetensors",  # sin fp16 en esta carpeta: se queda
        "README.md",
    ]
    assert m.select_safe_files(names) == [
        "model_index.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]


def test_install_from_folder_copies_only_safe_files(manager, tmp_path):
    src = _fake_marigold(tmp_path / "Mi modelo")
    entry = manager.install_from_folder(src)

    assert entry.source == "local" and entry.family == "marigold-diffusers"
    snap = manager.snapshot(entry.id)
    assert snap is not None
    copied = sorted(p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file())
    assert copied == ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"]
    assert manager.variant(snap) == "fp16"

    # Persiste: un gestor nuevo sobre la misma carpeta lo sigue viendo.
    again = m.ModelManager(manager.models_dir)
    assert again.is_installed(entry.id)


def test_install_twice_gets_unique_ids(manager, tmp_path):
    src = _fake_marigold(tmp_path / "modelo")
    a = manager.install_from_folder(src)
    b = manager.install_from_folder(src)
    assert a.id != b.id


def test_rejects_invalid_folders(manager, tmp_path):
    empty = tmp_path / "vacia"
    empty.mkdir()
    with pytest.raises(m.ModelError):
        manager.install_from_folder(empty)

    other = _fake_marigold(tmp_path / "otra familia", class_name="StableDiffusionPipeline")
    with pytest.raises(m.ModelError, match="familia"):
        manager.install_from_folder(other)

    only_pickle = tmp_path / "solo pickle"
    (only_pickle / "unet").mkdir(parents=True)
    (only_pickle / "model_index.json").write_text(json.dumps({"_class_name": "MarigoldDepthPipeline"}))
    (only_pickle / "unet" / "diffusion_pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(m.ModelError, match="safetensors"):
        manager.install_from_folder(only_pickle)


def test_delete_local_model_removes_files_and_entry(manager, tmp_path):
    entry = manager.install_from_folder(_fake_marigold(tmp_path / "borrable"))
    folder = manager.snapshot(entry.id)
    manager.delete(entry.id)
    assert not folder.exists()
    assert entry.id not in [e.id for e in manager.all()]


def test_delete_catalog_model_keeps_it_listed(manager):
    # Simula una descarga completa del catálogo.
    snap = manager.models_dir / "models--prs-eth--marigold-depth-v1-0" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model_index.json").write_text("{}")
    (manager.models_dir / f"{m.DEFAULT_MODEL}.complete.json").write_text(
        json.dumps({"snapshot": snap.relative_to(manager.models_dir).as_posix()}))
    assert manager.is_installed(m.DEFAULT_MODEL)

    manager.delete(m.DEFAULT_MODEL)
    assert not manager.is_installed(m.DEFAULT_MODEL)
    assert m.DEFAULT_MODEL in [e.id for e in manager.all()]


def test_license_check():
    assert m.is_permissive("Apache-2.0")
    assert not m.is_permissive("openrail++")
    assert not m.is_permissive(None)


def test_catalog_includes_lcm(manager):
    lcm = manager.get("marigold-lcm-v1-0")
    assert lcm.repo_id == "prs-eth/marigold-depth-lcm-v1-0" and m.is_permissive(lcm.license)


def test_search_compatible_filters_and_sorts(manager, monkeypatch):
    """Búsqueda con Hugging Face simulado: oficiales primero, sin repos restringidos."""
    from types import SimpleNamespace

    import huggingface_hub

    fake = [
        SimpleNamespace(id="someone/copia", tags=["license:openrail++"], downloads=900, likes=0, gated=False),
        SimpleNamespace(id="prs-eth/marigold-depth-v1-0", tags=["license:apache-2.0"], downloads=10,
                        likes=5, gated=False),
        SimpleNamespace(id="other/privado", tags=[], downloads=5000, likes=0, gated="manual"),
    ]
    calls = []

    class FakeApi:
        def list_models(self, **kwargs):
            calls.append(kwargs)
            return iter(fake)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    results = manager.search_compatible()

    assert calls[0]["filter"] == "diffusers:MarigoldDepthPipeline"
    assert [r.repo_id for r in results] == ["prs-eth/marigold-depth-v1-0", "someone/copia"]
    assert results[0].official and results[0].license == "apache-2.0"
    assert not results[1].official and results[1].license == "openrail++"
    assert manager.is_listed("prs-eth/marigold-depth-v1-0")
    assert not manager.is_listed("someone/copia")


def test_depth_anything_is_in_the_catalog_with_its_own_index_file(tmp_path):
    from deepshot.models import ALL_FAMILIES, ModelManager

    manager = ModelManager(tmp_path)
    entry = manager.get("depth-anything-v2-small")
    assert entry.family == "depth-anything" and entry.license == "apache-2.0"
    assert ALL_FAMILIES["depth-anything"].index_file == "config.json"
    assert manager.snapshot(entry.id) is None                               # sin descargar
    snap = tmp_path / "models--x" / "snap"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")                                  # sin model_index.json: vale igual
    (tmp_path / f"{entry.id}.complete.json").write_text(json.dumps({"snapshot": "models--x/snap"}))
    assert manager.snapshot(entry.id) == snap


def test_non_commercial_models_carry_a_warning_and_are_never_downloaded_by_themselves(tmp_path, monkeypatch):
    from deepshot import depth as d

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path))
    manager = m.ModelManager(tmp_path / "models")
    for model_id in ("depth-anything-v2-base", "depth-anything-v2-large"):
        entry = manager.get(model_id)
        assert entry.license == "cc-by-nc-4.0" and not m.is_permissive(entry.license)
        assert "NO para uso comercial" in entry.warning and entry.family == "depth-anything"
    estimator = d.DepthEstimator(device="cpu", manager=manager)
    with pytest.raises(m.ModelError):                                  # calcular con él sin tenerlo no lo descarga solo
        estimator.load("depth-anything-v2-large")


# --- Carpeta de modelos junto al programa y modelos soltados a mano ------------------------------------------------
def _drop_model(folder, model_type="dpt", weights=True):
    """Una carpeta como la que un usuario copiaría a mano: config.json + pesos (el contenido de los pesos da igual)."""
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    if weights:
        (folder / "model.safetensors").write_bytes(b"0")


def test_models_dir_resolution_order_and_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSHOT_MODELS_DIR", raising=False)
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "datos"))
    assert m.resolve_models_dir() == tmp_path / "datos" / "models"                 # datos forzados (pruebas, portátil)
    monkeypatch.setenv("DEEPSHOT_MODELS_DIR", str(tmp_path / "mios"))
    assert m.resolve_models_dir() == tmp_path / "mios"                              # la variable de modelos manda
    monkeypatch.delenv("DEEPSHOT_MODELS_DIR")
    monkeypatch.delenv("DEEPSHOT_DATA_DIR")
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ajustes"))
    monkeypatch.setattr(m, "program_dir", lambda: tmp_path / "programa")
    assert m.resolve_models_dir() == tmp_path / "programa" / "models"              # junto al programa, por defecto
    m.save_models_dir(tmp_path / "externa")
    assert m.resolve_models_dir() == tmp_path / "externa"                           # la elegida en el gestor
    m.save_models_dir(None)
    assert m.resolve_models_dir() == tmp_path / "programa" / "models"


def test_models_dropped_by_hand_appear_in_the_list_and_unsafe_or_foreign_ones_do_not(tmp_path):
    manager = m.ModelManager(tmp_path / "models")
    _drop_model(manager.models_dir / "mi-midas", "dpt")
    _drop_model(manager.models_dir / "mi-zoe", "zoedepth")
    _drop_model(manager.models_dir / "mi-da", "depth_anything")
    _drop_model(manager.models_dir / "sin-pesos", "dpt", weights=False)             # sin .safetensors: no se acepta
    _drop_model(manager.models_dir / "otro-modelo", "bert")                          # no es de profundidad
    (manager.models_dir / "models--org--cache").mkdir()                              # la caché de la app no cuenta
    (manager.models_dir / "local").mkdir()
    found = {e.label: e for e in manager.scan_folders()}
    assert set(found) == {"mi-midas", "mi-zoe", "mi-da"}
    assert found["mi-midas"].family == "transformers-depth" and found["mi-da"].family == "depth-anything"
    entry = found["mi-midas"]
    assert entry.source == "folder" and entry.id == "dir-mi-midas" and entry.removable
    assert entry.id in {e.id for e in manager.all()} and manager.is_installed(entry.id)   # sale en la lista y está instalado
    assert manager.disk_size(entry.id) == 1 + len(json.dumps({"model_type": "dpt"}))
    assert manager.download(entry.id) == manager.snapshot(entry.id)                  # ya está: no descarga nada
    manager.delete(entry.id)
    assert not (manager.models_dir / "mi-midas").exists()


def test_legacy_models_are_found_and_moved_without_overwriting(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "datos"))
    old = m.app_data_dir() / "models"
    (old / "models--org--x").mkdir(parents=True)
    (old / "models--org--x" / "a.bin").write_bytes(b"1")
    (old / "marigold-v1-0.complete.json").write_text("{}", encoding="utf-8")
    (old / "xet").mkdir()
    new = m.ModelManager(tmp_path / "nuevos")
    assert new.legacy_dir() == old
    (new.models_dir / "marigold-v1-0.complete.json").write_text("{\"ya\": 1}", encoding="utf-8")   # ya existe: no se pisa
    moved = new.adopt_from(old)
    assert moved == 1 and (new.models_dir / "models--org--x" / "a.bin").is_file()
    assert json.loads((new.models_dir / "marigold-v1-0.complete.json").read_text(encoding="utf-8")) == {"ya": 1}
    assert not (new.models_dir / "xet").exists()
    assert m.ModelManager(old).legacy_dir() is None                                  # la actual no es «anterior»


def test_model_manager_dialog_has_a_page_link_per_hugging_face_model_and_refreshes_dropped_folders(monkeypatch, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QToolButton

    from deepshot.app import create_app
    from deepshot.depth import DepthEstimator
    from deepshot.ui.model_manager import ModelManagerDialog

    create_app([])
    monkeypatch.setattr(ModelManagerDialog, "shows_more_licenses", lambda self: True)    # aquí se prueban todos
    monkeypatch.setenv("DEEPSHOT_DEV_MODELS", "1")
    manager = m.ModelManager(tmp_path / "models")
    dialog = ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), m.DEFAULT_MODEL, lambda: False)
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toString()) or True))

    def row(model_id):
        return next(dialog.tree.topLevelItem(i) for i in range(dialog.tree.topLevelItemCount())
                    if dialog.tree.topLevelItem(i).data(0, Qt.UserRole) == model_id)

    button = dialog.tree.itemWidget(row("marigold-v1-0"), dialog.COL_LINK)
    assert isinstance(button, QToolButton) and "huggingface.co/prs-eth/marigold-depth-v1-0" in button.toolTip()
    button.click()
    assert opened == ["https://huggingface.co/prs-eth/marigold-depth-v1-0"]
    assert dialog.tree.itemWidget(row("depth-pro"), dialog.COL_LINK) is not None
    # Un modelo soltado a mano: aparece al actualizar la lista y no lleva enlace (no es de Hugging Face).
    _drop_model(manager.models_dir / "mi-modelo", "dpt")
    assert not any(dialog.tree.topLevelItem(i).data(0, Qt.UserRole) == "dir-mi-modelo"
                   for i in range(dialog.tree.topLevelItemCount()))
    dialog.btn_refresh.click()
    assert dialog.tree.itemWidget(row("dir-mi-modelo"), dialog.COL_LINK) is None
    assert "1" in dialog.status.text()
    for widget in (dialog.btn_refresh, dialog.btn_open_folder, dialog.btn_change_folder, dialog.btn_adopt):
        assert widget.toolTip()
    assert str(manager.models_dir) in dialog.folder_label.text()
    dialog.btn_open_folder.click()
    assert opened[-1].endswith("models") or "models" in opened[-1]
    # Casilla de cada modelo: sale o no en el selector. El modelo activo no se puede ocultar y la elección se recuerda.
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ajustes"))
    QSettings().remove("models/disabled")
    events = []
    dialog.models_changed.connect(lambda: events.append(1))
    row("marigold-lcm-v1-0").setCheckState(dialog.COL_NAME, Qt.Unchecked)
    assert manager.disabled_ids() == {"marigold-lcm-v1-0"} and not manager.is_enabled("marigold-lcm-v1-0") and events
    row(m.DEFAULT_MODEL).setCheckState(dialog.COL_NAME, Qt.Unchecked)               # el activo: se queda marcado
    assert row(m.DEFAULT_MODEL).checkState(dialog.COL_NAME) == Qt.Checked and manager.is_enabled(m.DEFAULT_MODEL)
    dialog.refresh()
    assert row("marigold-lcm-v1-0").checkState(dialog.COL_NAME) == Qt.Unchecked     # se mantiene al rehacer la lista
    manager.set_enabled("marigold-lcm-v1-0", True)
    assert manager.disabled_ids() == set()
    QSettings().remove("models/disabled")
    dialog.deleteLater()


# --- Gestor en tarjetas, actualizar y cargar al arrancar -------------------------------------------------------------
def _install_fake(manager, model_id, snapshot_name, revision):
    """Deja un modelo del catálogo como «instalado» sin descargar nada: carpeta con su archivo índice y su marcador."""
    entry = manager.get(model_id)
    snap = manager.models_dir / snapshot_name
    snap.mkdir(parents=True, exist_ok=True)
    index = m.ALL_FAMILIES[entry.family].index_file
    (snap / index).parent.mkdir(parents=True, exist_ok=True)
    (snap / index).write_text("{}", encoding="utf-8")
    (manager.models_dir / f"{model_id}.complete.json").write_text(
        json.dumps({"model": entry.repo_id, "revision": revision, "snapshot": snapshot_name, "extra": {}}), encoding="utf-8")


def test_updates_are_detected_by_revision_and_the_old_version_is_removed_after_updating(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import huggingface_hub

    manager = m.ModelManager(tmp_path / "models")
    _install_fake(manager, "marigold-v1-0", "snap-vieja", "vieja")
    _install_fake(manager, "depth-anything-v2-small", "snap-da", "igual")
    assert manager.installed_revision("marigold-v1-0") == "vieja"
    latest = {"prs-eth/marigold-depth-v1-0": "nueva", "depth-anything/Depth-Anything-V2-Small-hf": "igual"}
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info",
                        lambda self, repo, **kw: SimpleNamespace(sha=latest[repo], siblings=[]))
    assert manager.check_updates() == {"marigold-v1-0"}                     # solo el que cambió de versión
    latest["prs-eth/marigold-depth-v1-0"] = "vieja"
    assert manager.check_updates() == set()
    # Tras actualizar, la carpeta de la versión anterior se borra y la nueva se queda.
    new_snap = manager.models_dir / "snap-nueva"
    new_snap.mkdir()
    manager._remove_old_revision(manager.get("marigold-v1-0"), manager._read_marker("marigold-v1-0"), keep=new_snap)
    assert not (manager.models_dir / "snap-vieja").exists() and new_snap.exists()
    manager._remove_old_revision(manager.get("depth-anything-v2-small"),
                                 manager._read_marker("depth-anything-v2-small"),
                                 keep=manager.models_dir / "snap-da")      # la misma carpeta: no se borra
    assert (manager.models_dir / "snap-da").exists()


def test_preload_setting_is_one_installed_model_at_a_time(tmp_path):
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ajustes"))
    manager = m.ModelManager(tmp_path / "models")
    assert manager.preload_id() is None
    manager.set_preload("marigold-v1-0")
    assert manager.preload_id() is None                                      # no está instalado: no se precarga
    _install_fake(manager, "marigold-v1-0", "snap", "r")
    assert manager.preload_id() == "marigold-v1-0"
    manager.set_preload("depth-anything-v2-small")                           # otro: sustituye al anterior
    assert manager.preload_id() is None                                      # (este no está instalado)
    manager.set_preload(None)
    assert QSettings().value("models/preload", "") == ""


def test_model_manager_cards_view_mirrors_the_list_with_buttons_checkboxes_and_updates(monkeypatch, tmp_path):
    import time

    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    from deepshot.app import create_app
    from deepshot.depth import DepthEstimator
    from deepshot.ui.model_manager import ModelManagerDialog

    create_app([])
    monkeypatch.setattr(ModelManagerDialog, "shows_more_licenses", lambda self: True)    # aquí se prueban todos
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ajustes"))
    monkeypatch.setenv("DEEPSHOT_DEV_MODELS", "1")
    manager = m.ModelManager(tmp_path / "models")
    _install_fake(manager, "marigold-v1-0", "snap-mg", "vieja")
    dialog = ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), "marigold-v1-0", lambda: False)
    dialog.view_switch.set_current_index(1)                                   # a la vista de tarjetas
    dialog._on_view_changed(1)
    assert dialog.view_stack.currentIndex() == 1 and QSettings().value("models/view") == "cards"
    assert set(dialog._cards) == {e.id for e in manager.all(with_modules=True) if m.can_compute(e) or e.is_helper}   # una tarjeta por modelo o módulo que se puede usar (y por modelo de apoyo)
    helper = dialog._cards["sam2-1-hiera-small"]
    assert helper.use_button is None and not helper.enabled_check.isVisibleTo(helper) and helper.install_button is not None
    installed, missing = dialog._cards["marigold-v1-0"], dialog._cards["depth-pro"]
    assert installed.use_button is not None and installed.uninstall_button is not None and installed.install_button is None
    assert missing.install_button is not None and missing.use_button is None
    assert installed.link_button is not None and "huggingface.co/prs-eth/marigold-depth-v1-0" in installed.link_button.toolTip()
    assert not installed.use_button.isEnabled()                                # ya es el activo
    assert installed.property("active") is True
    # Clic en una tarjeta: se selecciona y la lista la sigue; las acciones de la derecha actúan sobre ella.
    installed.selected.emit("depth-pro")
    dialog._select_card("depth-pro")
    assert dialog.selected_id() == "depth-pro" and dialog._cards["depth-pro"].property("current") is True
    assert dialog.btn_download.isEnabled()
    # Casillas de la tarjeta.
    dialog._cards["marigold-lcm-v1-0"].enabled_check.setChecked(False)
    assert not manager.is_enabled("marigold-lcm-v1-0")
    dialog._cards["marigold-v1-0"].preload_check.setChecked(True)
    assert manager.preload_id() == "marigold-v1-0"
    assert not dialog._cards["depth-pro"].preload_check.isEnabled()            # no instalado: no se puede precargar
    # Actualización disponible: aparece «Actualizar» y, al pulsarlo con confirmación, llama a update del gestor.
    dialog._updates = {"marigold-v1-0"}
    dialog.refresh()
    card = dialog._cards["marigold-v1-0"]
    assert card.update_button is not None and card.data.status_tone == "warn"
    dialog._select_card("marigold-v1-0")
    assert dialog.btn_update.isEnabled()
    calls = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(m.ModelManager, "update", lambda self, model_id, progress=None: calls.append(model_id))
    card.update_button.click()
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        QApplication.instance().processEvents()
        time.sleep(0.02)
    assert calls == ["marigold-v1-0"]
    # La vista de lista también lleva la columna «Al arrancar» con su casilla.
    dialog._on_view_changed(0)
    assert dialog.view_stack.currentIndex() == 0
    item = next(dialog.tree.topLevelItem(i) for i in range(dialog.tree.topLevelItemCount())
                if dialog.tree.topLevelItem(i).data(0, Qt.UserRole) == "marigold-v1-0")
    assert item.checkState(dialog.COL_PRELOAD) == Qt.Checked
    QSettings().remove("models/preload")
    QSettings().remove("models/disabled")
    dialog.deleteLater()


def test_model_list_columns_can_be_resized_moved_hidden_and_remembered(monkeypatch, tmp_path):
    from deepshot import depth as d
    from deepshot.ui.model_manager import ModelManagerDialog

    manager = m.ModelManager(tmp_path / "models")
    dialog = ModelManagerDialog(manager, d.DepthEstimator(device="cpu", manager=manager), m.DEFAULT_MODEL, lambda: False)
    header = dialog.tree.header()
    assert header.sectionsMovable() and header.sectionResizeMode(dialog.COL_NAME) == header.ResizeMode.Interactive
    header.resizeSection(dialog.COL_NAME, 333)
    dialog.refresh()                                                        # refrescar no deshace el ancho elegido
    assert header.sectionSize(dialog.COL_NAME) == 333
    dialog.tree.setColumnHidden(dialog.COL_FAMILY, True)
    header.moveSection(header.visualIndex(dialog.COL_STATE), 1)
    dialog.accept()
    again = ModelManagerDialog(manager, d.DepthEstimator(device="cpu", manager=manager), m.DEFAULT_MODEL, lambda: False)
    assert again.tree.isColumnHidden(dialog.COL_FAMILY) and again.tree.header().sectionSize(dialog.COL_NAME) == 333
    assert again.tree.header().visualIndex(dialog.COL_STATE) == 1


def test_fits_vram_compares_the_declared_minimum_with_the_gpu():
    v1, v2 = m.MODELS_CATALOG[0], next(e for e in m.MODELS_CATALOG if e.id == "marigold-v2-log")
    small = next(e for e in m.MODELS_CATALOG if e.id == "depth-anything-v2-small")
    assert m.fits_vram(v1, 8.0) is True and m.fits_vram(v2, 8.0) is False
    assert m.fits_vram(v2, None) is None and m.fits_vram(small, 2.0) is None     # sin dato: no se juzga


def test_model_that_does_not_fit_shows_red_with_the_reason_and_asks_before_using(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    from deepshot.app import create_app
    from deepshot.depth import DepthEstimator
    from deepshot.ui import theme
    from deepshot.ui.model_manager import ModelManagerDialog

    create_app([])
    monkeypatch.setattr(m, "UNIMPLEMENTED_FAMILIES", set())               # aquí se prueba solo lo de la memoria
    manager = m.ModelManager(tmp_path / "models")
    dialog = ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), m.DEFAULT_MODEL, lambda: False)
    dialog._vram_gb, dialog._gpu_name = 8.0, "GPU de prueba"
    dialog.refresh()
    assert "17 GB de VRAM" in dialog.misfit_reason(manager.get("marigold-v2-log")) and "8 GB" in dialog.misfit_reason(
        manager.get("marigold-v2-log"))
    assert dialog.misfit_reason(manager.get("marigold-v1-0")) == ""
    row = next(dialog.tree.topLevelItem(i) for i in range(dialog.tree.topLevelItemCount())
               if dialog.tree.topLevelItem(i).data(0, Qt.UserRole) == "marigold-v2-log")
    assert row.foreground(dialog.COL_NAME).color() == theme.color("bad")
    assert dialog._cards["marigold-v2-log"].data.status_tone == "bad"
    asked = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: asked.append(a) or QMessageBox.No))
    dialog.activate("marigold-v2-log")
    assert asked and dialog.active_id == m.DEFAULT_MODEL                 # dijo que no: sigue el modelo de antes
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog.activate("marigold-v2-log")
    assert dialog.active_id == "marigold-v2-log"


def test_license_filter_hides_non_commercial_models_until_the_user_consents(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QMessageBox

    from deepshot.app import create_app
    from deepshot.depth import DepthEstimator
    from deepshot.ui.model_manager import ModelManagerDialog

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ini"))
    create_app([])
    monkeypatch.setenv("DEEPSHOT_DEV_MODELS", "1")
    manager = m.ModelManager(tmp_path / "models")
    dialog = ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), m.DEFAULT_MODEL, lambda: False)

    def ids():
        return {dialog.tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(dialog.tree.topLevelItemCount())}

    assert "marigold-v1-0" in ids() and "midas-3-1-beit-large" in ids()          # permisivas: a la vista
    assert not ids() & {"depth-anything-v2-base", "depth-anything-v2-large", "depth-pro"}   # no comerciales: ocultas
    assert set(dialog._cards) == ids()                                           # las tarjetas igual que la lista
    assert "Mostrar más" in dialog.btn_licenses.text()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.No))
    dialog.toggle_more_licenses()                                                # sin consentimiento no cambia
    assert "depth-pro" not in ids()
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog.toggle_more_licenses()
    assert {"depth-anything-v2-base", "depth-pro"} <= ids() and "Ocultar" in dialog.btn_licenses.text()
    dialog.toggle_more_licenses()                                                # ocultar no pregunta
    assert "depth-pro" not in ids()
    dialog.active_id = "depth-pro"                                               # el modelo en uso nunca se esconde
    dialog.refresh()
    assert "depth-pro" in ids()


def test_a_model_the_app_cannot_compute_yet_warns_before_download_and_cannot_be_used(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from deepshot.app import create_app
    from deepshot.depth import DepthEstimator
    from deepshot.ui.model_manager import ModelManagerDialog

    create_app([])
    manager = m.ModelManager(tmp_path / "models")
    dialog = ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), m.DEFAULT_MODEL, lambda: False)
    v2, v1 = manager.get("marigold-v2-log"), manager.get("marigold-v1-0")
    assert not m.can_compute(v2) and m.can_compute(v1)
    assert "marigold-v2-log" not in dialog._cards                                     # sin descargar, no se ofrece
    monkeypatch.setattr(manager, "is_installed", lambda mid: mid == "marigold-v2-log")  # descargado: se ve, para poder borrarlo
    dialog.refresh()
    assert "todavía no puede calcular" in dialog.no_compute_reason(v2) and dialog.no_compute_reason(v1) == ""
    assert dialog._cards["marigold-v2-log"].data.status_tone == "bad"
    assert "Sin cálculo" in dialog._cards["marigold-v2-log"].data.status
    asked, started = [], []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: asked.append(a[2]) or QMessageBox.No))
    monkeypatch.setattr(dialog, "_run", lambda *a, **k: started.append(1))
    dialog.download("marigold-v2-log")
    assert asked and "todavía no puede calcular" in asked[0] and started == []        # dijo que no: no descarga
    infos = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a[2])))
    dialog.activate("marigold-v2-log")
    assert infos and dialog.active_id == m.DEFAULT_MODEL                              # no se deja activar
