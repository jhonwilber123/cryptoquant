# Prueba de la instalación: Python y VS Code.
# Pega todo en una celda de un cuaderno de VS Code y ejecútala. La primera vez tarda unos minutos.
import glob, importlib, os, platform, shutil, subprocess, sys

PAQUETES = {"pandas": "pandas", "numpy": "numpy", "matplotlib": "matplotlib", "scipy": "scipy",
            "scikit-learn": "sklearn", "statsmodels": "statsmodels", "requests": "requests"}
print("Instalando los paquetes del curso (la primera vez tarda unos minutos)...")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *PAQUETES])

fallos = []


def ok(texto):
    print("[OK]", texto)


def revisar(texto):
    fallos.append(texto)
    print("[REVISAR]", texto)


def buscar(*rutas):
    for ruta in rutas:
        hallados = glob.glob(os.path.expanduser(os.path.expandvars(ruta)))
        if hallados:
            return hallados[0]


if sys.version_info >= (3, 10):
    ok(f"Python {platform.python_version()} en {platform.system().replace('Darwin', 'macOS')}")
else:
    revisar(f"Python {platform.python_version()} es muy antiguo: instala el de python.org (paso 1).")
if "google.colab" in sys.modules:
    revisar("Esto es Google Colab. La prueba se hace en VS Code, en tu computadora.")

for nombre, modulo in PAQUETES.items():
    try:
        ok(f"{nombre} {importlib.import_module(modulo).__version__}")
    except Exception:
        revisar(f"No se pudo instalar {nombre}. Copia el error de arriba y pégaselo a Claude.")

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    velas = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "1d", "limit": 90}, timeout=20).json()
    cierre = pd.Series([float(v[4]) for v in velas], index=pd.to_datetime([v[0] for v in velas], unit="ms"))
    ok(f"Internet: último cierre de BTC en Binance, {cierre.iloc[-1]:,.2f} USDT")
    cierre.plot(title="BTC, últimos 90 días (Python)", figsize=(8, 3))
    plt.show()
except Exception as e:
    revisar(f"No se pudo descargar de Binance ({type(e).__name__}). ¿Hay internet? ¿Lo bloquea un antivirus?")

r = shutil.which("Rscript") or buscar(r"C:\Program Files\R\R-*\bin\Rscript.exe",
                                      r"%LOCALAPPDATA%\Programs\R\R-*\bin\Rscript.exe",
                                      "/Library/Frameworks/R.framework/Resources/bin/Rscript")
rstudio = buscar(r"C:\Program Files\RStudio\rstudio.exe", r"%LOCALAPPDATA%\Programs\RStudio\rstudio.exe",
                 "/Applications/RStudio.app", "~/Applications/RStudio.app")
if r:
    ok("R instalado")
else:
    revisar("No encuentro R. Instálalo (paso 3) antes que RStudio.")
if rstudio:
    ok("RStudio instalado")
else:
    revisar("No encuentro RStudio (paso 4).")

print()
if fallos:
    print(f"Faltan {len(fallos)} cosas: mira los [REVISAR] y la tabla «Si algo falla» de la guía.")
else:
    print("TODO LISTO. Escribe en el chat de la clase:  Python y VS Code OK")
