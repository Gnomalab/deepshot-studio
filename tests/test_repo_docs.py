"""Public-repository checks: licence, credits and no personal data in the tracked documents."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    return [ROOT / f for f in out if f.endswith((".md", ".cff", ".toml", ".txt", ".bat")) or f == "LICENSE"]


def test_licence_reserves_the_rights_and_asks_derivatives_to_credit_the_author_and_the_reference():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "All rights reserved" in text and "Juanjo Fernández" in text and "Gnomalab Estudio" in text
    assert "Karim Joseph Nassar" in text and "10.5281/zenodo.23103263" in text and "Vathography" in text
    assert "no affiliation" in text and "THIRD_PARTY_LICENSES.md" in text


def test_readme_and_citation_credit_both_people_and_link_to_the_licence():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Licence" in readme and "(LICENSE)" in readme and "(CITATION.cff)" in readme
    assert "Juanjo Fernández" in readme and "Karim Joseph Nassar" in readme and "Vathography" in readme
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^cff-version: 1\.2\.0$", cff, re.M) and "Nassar" in cff and "Fernández" in cff
    assert "10.5281/zenodo.23103263" in cff


def test_no_personal_paths_or_references_folders_are_published():
    for path in tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "C:\\Users\\gnoma" not in text and "/Users/gnomalabmk2" not in text, path.name
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "Link References" not in tracked and "NOTAS_CREDITOS_Y_3D.md" not in tracked
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "Link References/" in ignore
