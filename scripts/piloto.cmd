@echo off
rem Piloto de riesgo: doble clic para abrirlo en el navegador.
rem Cierre esta ventana (o pulse Ctrl+C) para apagarlo.
cd /d "%~dp0.."
python -m cryptoquant piloto
if errorlevel 1 pause
