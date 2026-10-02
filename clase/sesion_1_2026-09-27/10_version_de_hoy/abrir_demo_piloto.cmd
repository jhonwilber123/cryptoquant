@echo off
rem Demo del piloto para la clase: doble clic para abrirlo en el navegador.
rem Usa la cartera INVENTADA de piloto_demo\ (0,05 BTC, 1,2 ETH, 10 SOL, 500 USDT),
rem nunca la de data\piloto\. Cierre esta ventana (o pulse Ctrl+C) para apagarlo.
set "CRYPTOQUANT_PILOTO=%~dp0piloto_demo"
cd /d "%~dp0..\..\.."
python -m cryptoquant piloto
if errorlevel 1 pause
