"""Módulos descargables: formato, instalación en «looks», aprobación del código, tipos de efecto y entradas del gestor."""

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402

from deepshot import look, modules  # noqa: E402
from deepshot import models as m  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.look import KINDS, LookLayer, apply_look  # noqa: E402

SHIPPED = Path(__file__).resolve().parent.parent / "modules_catalog"
SAMPLE_CODE = "def process(image, depth, params):\n    return image * (1.0 - params['amount'])\n"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Carpetas de módulos y ajustes propias de cada prueba, y sin módulos cargados al empezar ni al terminar."""
    create_app([])
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("DEEPSHOT_MODULES_DIR", str(tmp_path / "instalados"))
    monkeypatch.setenv("DEEPSHOT_MODULES_CATALOG", str(tmp_path / "catalogo"))
    (tmp_path / "catalogo").mkdir()
    look.unregister_module_kinds()
    yield tmp_path
    look.unregister_module_kinds()


def _image(h=24, w=32, seed=1):
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def _make_module(root: Path, module_id="oscurecer", code=SAMPLE_CODE, **manifest):
    folder = root / module_id
    folder.mkdir(parents=True)
    data = {"id": module_id, "name": "Oscurecer", "kind": "look",
            "params": [{"name": "amount", "label": "Cantidad", "default": 0.0, "min": 0.0, "max": 1.0,
                        "tooltip": "Cuánto oscurece"}], **manifest}
    (folder / "module.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "module.py").write_text(code, encoding="utf-8")
    return folder


@pytest.fixture
def catalog(env):
    """Un módulo de muestra en el catálogo local (la app no trae ninguno: los efectos básicos son de serie)."""
    return _make_module(env / "catalogo", homepage="https://ejemplo.com/oscurecer", description="Oscurece la imagen.")


@pytest.fixture
def loaded(catalog):
    modules.approve(modules.install("oscurecer"))
    assert not modules.register_installed().failed


# --- Catálogo de la app ----------------------------------------------------------------------------------------------
def test_el_catalogo_de_la_app_no_trae_modulos_de_ejemplo_solo_las_instrucciones():
    assert (SHIPPED / "README.md").is_file()
    assert modules._scan(SHIPPED) == []                                # grano, nitidez y viñeta son efectos de serie
    for kind in ("grain", "sharpen", "vignette"):
        assert kind in KINDS and not KINDS[kind].module_id


# --- Formato -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("change, fragment", [
    ({"id": "Mal Id"}, "id"),
    ({"kind": "otro"}, "tipo"),
    ({"params": [{"name": "a", "min": 1, "max": 0}]}, "rango"),
    ({"params": [{"name": "a"}, {"name": "a"}]}, "repetido"),
    ({"params": [{"name": "A!"}]}, "no vale"),
    ({"requires_models": "marigold"}, "requires_models"),
])
def test_un_module_json_incoherente_se_rechaza(tmp_path, change, fragment):
    folder = _make_module(tmp_path / "x", "ok_id")
    data = json.loads((folder / "module.json").read_text(encoding="utf-8"))
    data.update(change)
    if "id" not in change:
        data["id"] = "ok_id"
    (folder / "module.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(modules.ModuleError) as exc:
        modules.read_info(folder)
    assert fragment in str(exc.value)


def test_la_carpeta_debe_llamarse_como_el_id_y_hace_falta_module_py(tmp_path):
    folder = _make_module(tmp_path / "x", "uno")
    other = folder.rename(folder.parent / "otro")
    with pytest.raises(modules.ModuleError):
        modules.read_info(other)
    (other.parent / "uno").mkdir()
    (other.parent / "uno" / "module.json").write_text(json.dumps({"id": "uno"}), encoding="utf-8")
    with pytest.raises(modules.ModuleError):
        modules.read_info(other.parent / "uno")                      # sin module.py


def test_la_pagina_del_modulo_solo_vale_si_es_un_enlace_web(tmp_path):
    ok = modules.read_info(_make_module(tmp_path / "a", "uno", homepage="https://ejemplo.com/x"))
    assert ok.homepage == "https://ejemplo.com/x"
    for bad in ("file:///C:/algo", "C:\\carpeta", "javascript:alert(1)", ""):
        assert modules.read_info(_make_module(tmp_path / ("b" + str(abs(hash(bad)))), "uno", homepage=bad)).homepage == ""


# --- Dónde viven -----------------------------------------------------------------------------------------------------
def test_los_modulos_se_instalan_en_looks_dentro_de_la_carpeta_de_modelos(tmp_path, monkeypatch, catalog):
    monkeypatch.delenv("DEEPSHOT_MODULES_DIR")
    manager = m.ModelManager(tmp_path / "modelos")
    assert modules.modules_dir() == tmp_path / "modelos" / "looks"
    info = modules.install("oscurecer")
    assert info.path == tmp_path / "modelos" / "looks" / "oscurecer"
    other = m.ModelManager(tmp_path / "otros")                          # cambiar la carpeta de modelos mueve «looks» con ella
    assert modules.modules_dir() == tmp_path / "otros" / "looks" and modules.installed() == []
    assert manager.models_dir == tmp_path / "modelos"


def test_looks_no_se_confunde_con_un_modelo_de_profundidad(tmp_path):
    manager = m.ModelManager(tmp_path / "modelos")
    looks = tmp_path / "modelos" / "looks" / "uno"
    looks.mkdir(parents=True)
    (looks / "config.json").write_text("{}")
    (looks / "pesos.safetensors").write_bytes(b"x")                    # un módulo con pesos propios
    assert "looks" not in {e.label for e in manager.scan_folders()}


# --- Instalar, aprobar y cargar ---------------------------------------------------------------------------------------
def test_instalar_no_ejecuta_nada_hasta_aprobar(catalog):
    info = modules.install("oscurecer")
    assert [i.id for i in modules.installed()] == ["oscurecer"] and info.path == modules.modules_dir() / "oscurecer"
    report = modules.register_installed()
    assert [i.id for i in report.pending] == ["oscurecer"] and not report.loaded
    assert "module.oscurecer" not in KINDS                             # sin aprobar, su código no se importa
    modules.approve(info)
    report = modules.register_installed()
    assert report.loaded == ["oscurecer"] and not report.pending
    assert KINDS["module.oscurecer"].module_id == "oscurecer"


def test_editar_un_modulo_aprobado_pide_aprobarlo_otra_vez(catalog):
    info = modules.install("oscurecer")
    modules.approve(info)
    assert modules.is_approved(info)
    (info.path / "module.py").write_text(SAMPLE_CODE + "\n# cambio\n", encoding="utf-8")
    assert not modules.is_approved(info)
    assert [i.id for i in modules.register_installed().pending] == ["oscurecer"]


def test_desinstalar_quita_el_tipo_la_carpeta_y_la_aprobacion(loaded):
    assert "module.oscurecer" in KINDS
    modules.uninstall("oscurecer")
    assert "module.oscurecer" not in KINDS and not (modules.modules_dir() / "oscurecer").exists()
    assert "oscurecer" not in modules._approvals()


def test_un_modulo_que_falla_al_cargarse_o_al_aplicarse_no_rompe_nada(tmp_path):
    bad_import = _make_module(tmp_path / "src", "malo_import", code="raise RuntimeError('boom')\n")
    bad_run = _make_module(tmp_path / "src", "malo_run", code="def process(image, depth, params):\n    raise ValueError('x')\n")
    no_func = _make_module(tmp_path / "src", "sin_funcion", code="x = 1\n")
    for folder in (bad_import, bad_run, no_func):
        modules.approve(modules.install_from_folder(folder))
    report = modules.register_installed()
    assert set(report.failed) == {"malo_import", "sin_funcion"} and report.loaded == ["malo_run"]
    layer = LookLayer.new("module.malo_run")
    layer.params["amount"] = 0.5
    img = _image()
    assert np.allclose(apply_look(img, [layer]), img)                  # el efecto que falla se omite


def test_los_tipos_de_serie_no_se_pueden_pisar():
    with pytest.raises(ValueError):
        look.register_kind(look.LayerKind("levels", "x", (), lambda i, d, p: i))


def test_un_efecto_de_un_modulo_se_aplica_como_cualquier_otro(loaded):
    layer = LookLayer.new("module.oscurecer")
    assert layer.is_identity()
    layer.params["amount"] = 0.5
    img = _image()
    assert np.allclose(apply_look(img, [layer]), img * 0.5, atol=1e-6)


def test_proyecto_con_efecto_de_un_modulo_no_instalado_lo_conserva(tmp_path):
    from deepshot.project import Project, render

    p = Project()
    p.look = [LookLayer(kind="module.fantasma", params={"x": 3.0})]
    p.save(tmp_path / "p.json")
    q = Project.load(tmp_path / "p.json")
    assert q.look[0].missing and q.look[0].params == {"x": 3.0}
    q.validate()                                                        # y no impide abrir ni procesar el proyecto
    render(q, np.tile(np.linspace(0, 1, 16, dtype=np.float32), (8, 1)))


# --- Entradas del gestor de modelos --------------------------------------------------------------------------------------
def test_los_modulos_salen_en_el_gestor_solo_si_se_piden(tmp_path, catalog):
    manager = m.ModelManager(tmp_path / "models")
    assert not any(e.is_module for e in manager.all())
    entries = {e.id: e for e in manager.all(with_modules=True) if e.is_module}
    assert set(entries) == {"module:oscurecer"}
    entry = entries["module:oscurecer"]
    assert entry.kind == "look" and entry.source == "module" and not entry.removable and not manager.is_installed(entry.id)
    assert entry.url == "https://ejemplo.com/oscurecer" and entry.description == "Oscurece la imagen."


def test_instalar_y_borrar_un_modulo_desde_el_gestor(tmp_path, catalog):
    manager = m.ModelManager(tmp_path / "models")
    manager.download("module:oscurecer")
    assert manager.is_installed("module:oscurecer") and manager.disk_size("module:oscurecer") > 0
    assert manager.snapshot("module:oscurecer") == modules.modules_dir() / "oscurecer"
    manager.delete("module:oscurecer")
    assert not manager.is_installed("module:oscurecer")
    assert "module:oscurecer" in {e.id for e in manager.all(with_modules=True)}   # sigue en la lista: se puede reinstalar


def test_un_modulo_avisa_de_los_modelos_que_le_faltan(tmp_path):
    _make_module(tmp_path / "catalogo", "con_modelo", requires_models=["depth-anything-v2-small"])
    manager = m.ModelManager(tmp_path / "models")
    entry = manager.get("module:con_modelo")
    assert [e.id for e in manager.missing_requirements(entry)] == ["depth-anything-v2-small"]


def _dialog(tmp_path, monkeypatch):
    from deepshot.depth import DepthEstimator
    from deepshot.ui.model_manager import ModelManagerDialog

    monkeypatch.setattr(ModelManagerDialog, "shows_more_licenses", lambda self: True)
    manager = m.ModelManager(tmp_path / "models")
    return manager, ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), "marigold-v1-0", lambda: False)


def test_el_gestor_muestra_tipo_destino_y_enlace_filtra_e_instala_y_desinstala_modulos(monkeypatch, tmp_path, catalog):
    import time

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    _make_module(tmp_path / "catalogo", "sin_pagina", name="Sin página")
    manager, dialog = _dialog(tmp_path, monkeypatch)

    def rows():
        return {dialog.tree.topLevelItem(i).data(0, Qt.UserRole): dialog.tree.topLevelItem(i)
                for i in range(dialog.tree.topLevelItemCount())}

    def wait(condition, seconds=10):
        deadline = time.time() + seconds
        while not condition() and time.time() < deadline:
            QApplication.instance().processEvents()
            time.sleep(0.02)
        return condition()

    assert dialog.windowTitle() == "Gestor de modelos y módulos"
    module, model = rows()["module:oscurecer"], rows()["marigold-v1-0"]
    assert module.text(dialog.COL_KIND) == "Módulo de Look" and model.text(dialog.COL_KIND) == "Modelo de profundidad"
    assert module.text(dialog.COL_DEST) == str(modules.modules_dir() / "oscurecer")     # «Se instala en»
    assert model.text(dialog.COL_DEST) == str(manager.models_dir)
    assert module.text(dialog.COL_STATE) == "No instalado" and module.data(dialog.COL_NAME, Qt.CheckStateRole) is None
    link = dialog.tree.itemWidget(module, dialog.COL_LINK)                              # enlace a su página
    assert link is not None and "https://ejemplo.com/oscurecer" in link.toolTip()
    assert dialog.tree.itemWidget(rows()["module:sin_pagina"], dialog.COL_LINK) is None
    assert dialog._cards["module:oscurecer"].link_button is not None and dialog._cards["module:sin_pagina"].link_button is None
    dialog.kind_filter.set_current_index(2)                                           # solo módulos de Look
    dialog._on_kind_filter(2)
    assert set(rows()) == {"module:oscurecer", "module:sin_pagina"} and set(dialog._cards) == set(rows())
    dialog.kind_filter.set_current_index(1)                                           # solo modelos
    dialog._on_kind_filter(1)
    assert rows() and not any(k.startswith("module:") for k in rows())
    dialog.kind_filter.set_current_index(0)
    dialog._on_kind_filter(0)

    dialog.tree.setCurrentItem(rows()["module:oscurecer"])                            # como un clic en su fila
    assert dialog.selected_id() == "module:oscurecer"
    assert dialog.btn_download.isEnabled() and not dialog.btn_activate.isEnabled() and not dialog.btn_delete.isEnabled()
    asked = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: asked.append(a[2]) or QMessageBox.No))
    dialog.download_selected()                                                        # «No» al aviso del código: no se instala
    assert "código Python" in asked[0] and not manager.is_installed("module:oscurecer")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog.download_selected()
    assert wait(lambda: not dialog._busy and manager.is_installed("module:oscurecer"))
    assert wait(lambda: "module.oscurecer" in KINDS)                                  # aprobado y dado de alta como efecto
    assert modules.is_approved(modules.installed_info("oscurecer"))
    dialog.refresh("module:oscurecer")
    assert rows()["module:oscurecer"].text(dialog.COL_STATE).startswith("Instalado") and dialog.btn_delete.isEnabled()
    card = dialog._cards["module:oscurecer"]
    assert card.uninstall_button is not None and card.use_button is None

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog.delete_selected()
    assert not manager.is_installed("module:oscurecer") and "module.oscurecer" not in KINDS
    assert "module:oscurecer" in rows()                                               # sigue en la lista para reinstalarlo


def test_un_modulo_con_modelos_pendientes_no_se_instala_y_lo_explica(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    _make_module(tmp_path / "catalogo", "con_modelo", name="Con modelo", requires_models=["depth-anything-v2-small"])
    manager, dialog = _dialog(tmp_path, monkeypatch)
    infos = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a[2])))
    dialog.download("module:con_modelo")
    assert infos and "Depth Anything" in infos[0] and not manager.is_installed("module:con_modelo")


def test_el_panel_de_efectos_dibuja_los_ajustes_de_un_modulo_y_conserva_los_que_no_tienen_modulo(loaded):
    from deepshot.project import Project
    from deepshot.ui.look_panel import LookPanel

    panel = LookPanel()
    project = Project()
    project.look = [LookLayer.new("module.oscurecer"), LookLayer(kind="module.fantasma", params={"x": 1.0})]
    panel.bind(project)
    assert set(panel.param_sliders) == {"amount"} and all(s.toolTip() for s in panel.param_sliders.values())
    panel.param_sliders["amount"].setValue(0.7, emit=True)
    assert project.look[0].params["amount"] == pytest.approx(0.7)
    panel.list.setCurrentRow(1)                                                       # el efecto sin su módulo
    assert panel.missing_label.isVisibleTo(panel) and not panel.common.isVisibleTo(panel)
    assert "no instalado" in panel.list.item(1).text() and panel.param_sliders == {}
    panel.deleteLater()
