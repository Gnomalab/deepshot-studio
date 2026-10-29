"""Módulos descargables: formato, instalación, aprobación del código, tipos de capa y entradas del gestor."""

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


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Carpetas de módulos y ajustes propias de cada prueba, y sin módulos cargados al empezar ni al terminar."""
    create_app([])
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("DEEPSHOT_MODULES_DIR", str(tmp_path / "instalados"))
    monkeypatch.setenv("DEEPSHOT_MODULES_CATALOG", str(tmp_path / "catalogo"))
    shutil.copytree(SHIPPED, tmp_path / "catalogo")
    look.unregister_module_kinds()
    yield tmp_path
    look.unregister_module_kinds()


def _image(h=24, w=32, seed=1):
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def _make_module(root: Path, module_id="mi_modulo", code="def process(image, depth, params):\n    return image * params['k']\n",
                 **manifest):
    folder = root / module_id
    folder.mkdir(parents=True)
    data = {"id": module_id, "name": "Mi módulo", "kind": "look",
            "params": [{"name": "k", "label": "K", "default": 1.0, "min": 0.0, "max": 1.0}], **manifest}
    (folder / "module.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "module.py").write_text(code, encoding="utf-8")
    return folder


# --- Formato -------------------------------------------------------------------------------------------------------
def test_los_tres_modulos_del_catalogo_son_validos():
    infos = {i.id: i for i in modules.catalog()}
    assert set(infos) == {"film_grain", "sharpen", "vignette"}
    for info in infos.values():
        assert info.kind == "look" and info.license and info.version and info.params
        assert all(p.tooltip for p in info.params)                  # todo control lleva su ayuda


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


# --- Instalar, aprobar y cargar ---------------------------------------------------------------------------------------
def test_instalar_no_ejecuta_nada_hasta_aprobar():
    info = modules.install("film_grain")
    assert [i.id for i in modules.installed()] == ["film_grain"] and info.path == modules.modules_dir() / "film_grain"
    report = modules.register_installed()
    assert [i.id for i in report.pending] == ["film_grain"] and not report.loaded
    assert "module.film_grain" not in KINDS                            # sin aprobar, su código no se importa
    modules.approve(info)
    report = modules.register_installed()
    assert report.loaded == ["film_grain"] and not report.pending
    assert KINDS["module.film_grain"].module_id == "film_grain"


def test_editar_un_modulo_aprobado_pide_aprobarlo_otra_vez():
    info = modules.install("sharpen")
    modules.approve(info)
    assert modules.is_approved(info)
    (info.path / "module.py").write_text((info.path / "module.py").read_text(encoding="utf-8") + "\n# cambio\n",
                                         encoding="utf-8")
    assert not modules.is_approved(info)
    assert [i.id for i in modules.register_installed().pending] == ["sharpen"]


def test_desinstalar_quita_el_tipo_la_carpeta_y_la_aprobacion():
    info = modules.install("vignette")
    modules.approve(info)
    modules.register_installed()
    assert "module.vignette" in KINDS
    modules.uninstall("vignette")
    assert "module.vignette" not in KINDS and not (modules.modules_dir() / "vignette").exists()
    assert "vignette" not in modules._approvals()


def test_un_modulo_que_falla_al_cargarse_o_al_aplicarse_no_rompe_nada(tmp_path):
    bad_import = _make_module(tmp_path / "src", "malo_import", code="raise RuntimeError('boom')\n")
    bad_run = _make_module(tmp_path / "src", "malo_run", code="def process(image, depth, params):\n    raise ValueError('x')\n")
    no_func = _make_module(tmp_path / "src", "sin_funcion", code="x = 1\n")
    for folder in (bad_import, bad_run, no_func):
        modules.approve(modules.install_from_folder(folder))
    report = modules.register_installed()
    assert set(report.failed) == {"malo_import", "sin_funcion"} and report.loaded == ["malo_run"]
    layer = LookLayer.new("module.malo_run")
    layer.params["k"] = 0.5
    img = _image()
    assert np.allclose(apply_look(img, [layer]), img)                  # la capa que falla se omite


def test_los_tipos_de_serie_no_se_pueden_pisar():
    with pytest.raises(ValueError):
        look.register_kind(look.LayerKind("levels", "x", (), lambda i, d, p: i))


# --- Capas de los módulos de ejemplo ----------------------------------------------------------------------------------
@pytest.fixture
def loaded():
    for info in modules.catalog():
        modules.approve(modules.install(info.id))
    assert not modules.register_installed().failed


def test_las_capas_de_ejemplo_no_hacen_nada_por_defecto_y_cambian_la_imagen_al_tocarlas(loaded):
    img = _image(40, 60)
    for module_id, change in (("film_grain", {"amount": 0.8}), ("sharpen", {"amount": 1.5}),
                              ("vignette", {"amount": 0.9})):
        layer = LookLayer.new("module.%s" % module_id)
        assert layer.is_identity() and np.allclose(apply_look(img, [layer]), img)
        layer.params.update(change)
        out = apply_look(img, [layer])
        assert out.shape == img.shape and out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0
        assert not np.allclose(out, img, atol=1e-3)


def test_el_grano_es_siempre_el_mismo_y_la_vineta_oscurece_los_bordes(loaded):
    img = np.full((40, 60, 3), 0.5, dtype=np.float32)
    grain = LookLayer.new("module.film_grain")
    grain.params["amount"] = 1.0
    assert np.array_equal(apply_look(img, [grain]), apply_look(img, [grain]))     # reproducible
    vignette = LookLayer.new("module.vignette")
    vignette.params["amount"] = 1.0
    out = apply_look(img, [vignette])
    assert out[20, 30, 0] == pytest.approx(0.5) and out[0, 0, 0] < 0.3


def test_proyecto_con_capa_de_un_modulo_no_instalado_la_conserva(tmp_path):
    from deepshot.project import Project

    p = Project()
    layer = LookLayer(kind="module.fantasma", params={"x": 3.0})
    p.look = [layer]
    p.save(tmp_path / "p.json")
    q = Project.load(tmp_path / "p.json")
    assert q.look[0].missing and q.look[0].params == {"x": 3.0}
    q.validate()                                                        # y no impide abrir ni procesar el proyecto
    from deepshot.project import render

    render(q, np.tile(np.linspace(0, 1, 16, dtype=np.float32), (8, 1)))


# --- Entradas del gestor de modelos --------------------------------------------------------------------------------------
def test_los_modulos_salen_en_el_gestor_solo_si_se_piden(tmp_path):
    manager = m.ModelManager(tmp_path / "models")
    assert not any(e.is_module for e in manager.all())
    entries = {e.id: e for e in manager.all(with_modules=True) if e.is_module}
    assert set(entries) == {"module:film_grain", "module:sharpen", "module:vignette"}
    entry = entries["module:film_grain"]
    assert entry.kind == "look" and entry.source == "module" and not entry.removable and not manager.is_installed(entry.id)


def test_instalar_y_borrar_un_modulo_desde_el_gestor(tmp_path):
    manager = m.ModelManager(tmp_path / "models")
    manager.download("module:sharpen")
    assert manager.is_installed("module:sharpen") and manager.disk_size("module:sharpen") > 0
    assert manager.snapshot("module:sharpen") == modules.modules_dir() / "sharpen"
    manager.delete("module:sharpen")
    assert not manager.is_installed("module:sharpen")
    assert "module:sharpen" in {e.id for e in manager.all(with_modules=True)}   # sigue en la lista: se puede reinstalar


def test_un_modulo_avisa_de_los_modelos_que_le_faltan(tmp_path):
    folder = _make_module(tmp_path / "src", "con_modelo", requires_models=["depth-anything-v2-small"])
    (tmp_path / "catalogo").mkdir(exist_ok=True)
    shutil.copytree(folder, tmp_path / "catalogo" / "con_modelo")
    manager = m.ModelManager(tmp_path / "models")
    entry = manager.get("module:con_modelo")
    assert [e.id for e in manager.missing_requirements(entry)] == ["depth-anything-v2-small"]


def test_el_gestor_muestra_tipo_y_destino_filtra_e_instala_y_desinstala_modulos(monkeypatch, tmp_path):
    import time

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    from deepshot.depth import DepthEstimator
    from deepshot.ui.model_manager import ModelManagerDialog

    monkeypatch.setattr(ModelManagerDialog, "shows_more_licenses", lambda self: True)
    manager = m.ModelManager(tmp_path / "models")
    dialog = ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), "marigold-v1-0", lambda: False)

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
    grain, model = rows()["module:film_grain"], rows()["marigold-v1-0"]
    assert grain.text(dialog.COL_KIND) == "Módulo de Look" and model.text(dialog.COL_KIND) == "Modelo de profundidad"
    assert grain.text(dialog.COL_DEST) == str(modules.modules_dir() / "film_grain")     # «Se instala en»
    assert model.text(dialog.COL_DEST) == str(manager.models_dir)
    assert grain.text(dialog.COL_STATE) == "No instalado" and grain.data(dialog.COL_NAME, Qt.CheckStateRole) is None     # un módulo no lleva casilla
    dialog.kind_filter.set_current_index(2)                                           # solo módulos de Look
    dialog._on_kind_filter(2)
    assert set(rows()) == {"module:film_grain", "module:sharpen", "module:vignette"} and set(dialog._cards) == set(rows())
    dialog.kind_filter.set_current_index(1)                                           # solo modelos
    dialog._on_kind_filter(1)
    assert rows() and not any(k.startswith("module:") for k in rows())
    dialog.kind_filter.set_current_index(0)
    dialog._on_kind_filter(0)

    dialog.tree.setCurrentItem(rows()["module:film_grain"])                           # como un clic en su fila
    assert dialog.selected_id() == "module:film_grain"
    assert dialog.btn_download.isEnabled() and not dialog.btn_activate.isEnabled() and not dialog.btn_delete.isEnabled()
    asked = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: asked.append(a[2]) or QMessageBox.No))
    dialog.download_selected()                                                        # «No» al aviso del código: no se instala
    assert "código Python" in asked[0] and not manager.is_installed("module:film_grain")
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog.download_selected()
    assert wait(lambda: not dialog._busy and manager.is_installed("module:film_grain"))
    assert wait(lambda: "module.film_grain" in KINDS)                                 # aprobado y dado de alta como capa
    assert modules.is_approved(modules.installed_info("film_grain"))
    dialog.refresh("module:film_grain")
    assert rows()["module:film_grain"].text(dialog.COL_STATE).startswith("Instalado") and dialog.btn_delete.isEnabled()
    assert dialog._cards["module:film_grain"].uninstall_button is not None and dialog._cards["module:film_grain"].use_button is None

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog.delete_selected()
    assert not manager.is_installed("module:film_grain") and "module.film_grain" not in KINDS
    assert "module:film_grain" in rows()                                              # sigue en la lista para reinstalarlo


def test_un_modulo_con_modelos_pendientes_no_se_instala_y_lo_explica(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    from deepshot.depth import DepthEstimator
    from deepshot.ui.model_manager import ModelManagerDialog

    folder = _make_module(tmp_path / "src", "con_modelo", name="Con modelo", requires_models=["depth-anything-v2-small"])
    shutil.copytree(folder, tmp_path / "catalogo" / "con_modelo")
    manager = m.ModelManager(tmp_path / "models")
    dialog = ModelManagerDialog(manager, DepthEstimator(device="cpu", manager=manager), "marigold-v1-0", lambda: False)
    infos = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a[2])))
    dialog.download("module:con_modelo")
    assert infos and "Depth Anything" in infos[0] and not manager.is_installed("module:con_modelo")


def test_el_panel_de_look_dibuja_los_ajustes_de_un_modulo_y_conserva_las_capas_sin_modulo(loaded):
    from deepshot.project import Project
    from deepshot.ui.look_panel import LookPanel

    panel = LookPanel()
    project = Project()
    project.look = [LookLayer.new("module.film_grain"), LookLayer(kind="module.fantasma", params={"x": 1.0})]
    panel.bind(project)
    assert set(panel.param_sliders) == {"amount", "size"}
    assert all(s.toolTip() for s in panel.param_sliders.values())
    panel.param_sliders["amount"].setValue(0.7, emit=True)
    assert project.look[0].params["amount"] == pytest.approx(0.7)
    panel.list.setCurrentRow(1)                                                       # la capa sin su módulo
    assert panel.missing_label.isVisibleTo(panel) and not panel.common.isVisibleTo(panel)
    assert "no instalado" in panel.list.item(1).text() and panel.param_sliders == {}
    panel.deleteLater()
