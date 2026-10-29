"""Ajustes comunes de las pruebas: textos en español y sin pantalla de inicio (no deben depender del equipo)."""

import os
import tempfile

os.environ.setdefault("DEEPSHOT_LANG", "es")
os.environ.setdefault("DEEPSHOT_NO_START", "1")
# Ninguna prueba debe escribir en la carpeta de datos ni de modelos reales del equipo: por defecto, una temporal
# (las pruebas que necesitan la suya la fijan con monkeypatch.setenv("DEEPSHOT_DATA_DIR", ...)).
os.environ.setdefault("DEEPSHOT_DATA_DIR", tempfile.mkdtemp(prefix="deepshot-tests-"))
