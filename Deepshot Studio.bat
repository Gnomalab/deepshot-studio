@echo off
rem Lanzador de Deepshot Studio: doble clic para abrir la app sin consola.
rem %~dp0 es la carpeta de este archivo, así funciona aunque la ruta tenga espacios.
cd /d "%~dp0"
start "" "%~dp0.venv\Scripts\pythonw.exe" -m deepshot
