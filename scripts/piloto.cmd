@echo off
rem Piloto de riesgo: doble clic para abrirlo en el navegador.
rem Cierre esta ventana (o pulse Ctrl+C) para apagarlo.
cd /d "%~dp0.."
rem El entorno del proyecto (el mismo de la tarea programada); si no existe, el python del sistema.
set "PY=%USERPROFILE%\.venvs\cryptoquant\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m cryptoquant piloto
if errorlevel 1 pause
