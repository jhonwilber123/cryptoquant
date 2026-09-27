"""Genera alumno.ipynb y docente.ipynb de la primera clase desde una sola fuente.

Los dos cuadernos comparten texto, prompts y verificadores; el del docente
trae ademas la solucion en cada celda de "pega aqui". Editar este script y
volver a ejecutarlo:

    python clase/01_primera_clase/generar_cuadernos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Celda de preparacion: los verificadores. En Colab se ve como un formulario
# plegado (#@title): el alumno la ejecuta sin necesidad de leerla.
# ---------------------------------------------------------------------------
PREPARACION = r'''#@title Preparación: ejecuta esta celda una vez (botón ▶) y no la cambies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)


class _Verificador:
    """Comprueba el trabajo de cada bloque y explica qué revisar si algo falla."""

    # Cierre real de BTC/USDT en Binance el 01-01-2023 (vela diaria UTC).
    CONTROL_FECHA, CONTROL_CIERRE = "2023-01-01", 16616.75

    @staticmethod
    def _ok(texto):
        print("[OK] " + texto)

    @staticmethod
    def _mal(texto):
        print("[REVISAR] " + texto)

    # --- Bloque 1 ------------------------------------------------------------
    def cadena(self, valida_antes, valida_despues):
        if valida_antes is True and valida_despues is False:
            self._ok("La cadena era válida y, al cambiar un bloque, dejó de serlo.")
            print("     Por eso una blockchain es difícil de falsificar: cada bloque guarda")
            print("     la huella del anterior, y cambiar uno rompe todos los que vienen detrás.")
        elif valida_antes is not True:
            self._mal("La cadena original debería ser válida (True). Revisa la función que verifica.")
        else:
            self._mal("Tras cambiar un bloque, la verificación debería dar False. ¿Cambiaste los datos")
            print("     de un bloque SIN recalcular su hash? Si Claude recalculó los hashes, pídele que no lo haga.")

    # --- Bloque 2 ------------------------------------------------------------
    def velas(self, df):
        d = df.copy()
        if "fecha" not in d.columns and isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={d.index.name or "index": "fecha"})
        faltan = [c for c in ["fecha", "open", "high", "low", "close", "volume"] if c not in d.columns]
        if faltan:
            return self._mal(f"Faltan columnas: {faltan}. Pídele a Claude exactamente esos nombres.")
        fechas = pd.to_datetime(d["fecha"], utc=True)
        precios = d[["open", "high", "low", "close", "volume"]]
        if not all(pd.api.types.is_numeric_dtype(precios[c]) for c in precios):
            return self._mal("Los precios deben ser números, no texto. Pide convertirlos con astype(float).")
        if not fechas.is_monotonic_increasing or fechas.duplicated().any():
            return self._mal("Las fechas deben ir de la más antigua a la más reciente, sin repetirse.")
        if fechas.dt.year.max() > 2100 or fechas.dt.year.min() < 2009:
            return self._mal("Hay fechas imposibles. ¿Se leyó la marca de tiempo con la unidad equivocada?"
                             " Binance usa milisegundos (unit='ms').")
        if (fechas.dt.strftime("%H:%M") != "00:00").any():
            return self._mal("Las velas diarias deben empezar a las 00:00 UTC. Si ves 23:59, se usó la hora"
                             " de CIERRE como fecha: pide usar la hora de apertura (primer campo de Binance).")
        control = d.loc[fechas.dt.strftime("%Y-%m-%d") == self.CONTROL_FECHA, "close"]
        if control.empty:
            return self._mal(f"No encuentro la vela del {self.CONTROL_FECHA}. Pide los datos desde esa fecha.")
        if abs(float(control.iloc[0]) - self.CONTROL_CIERRE) > 0.01:
            return self._mal(f"El cierre del {self.CONTROL_FECHA} debería ser {self.CONTROL_CIERRE} y es "
                             f"{float(control.iloc[0])}. ¿Se tomó la columna equivocada? Binance devuelve 12"
                             " campos por vela y el cierre es el quinto; o las fechas están desplazadas un día.")
        malas = ((d["high"] < d[["open", "close"]].max(axis=1)) | (d["low"] > d[["open", "close"]].min(axis=1))).sum()
        if malas:
            return self._mal(f"{malas} velas tienen el máximo por debajo o el mínimo por encima del cuerpo.")
        hoy = pd.Timestamp.now(tz="UTC").normalize()
        if fechas.iloc[-1] >= hoy:
            return self._mal("La última vela es la de hoy, que aún no ha cerrado: su 'cierre' cambiará."
                             " Pide eliminar la vela en curso.")
        huecos = int((fechas.diff().dt.days > 1).sum())
        self._ok(f"{len(d)} velas diarias, del {fechas.iloc[0]:%d-%m-%Y} al {fechas.iloc[-1]:%d-%m-%Y}."
                 f" El cierre de control coincide con Binance.")
        if huecos:
            print(f"     Aviso: {huecos} saltos de más de un día entre velas.")

    def csv(self, ruta="BTCUSDT_1d.csv"):
        if not os.path.exists(ruta):
            return self._mal(f"No existe {ruta}. Pide guardarlo con ese nombre exacto.")
        leido = pd.read_csv(ruta)
        faltan = [c for c in ["fecha", "open", "high", "low", "close", "volume"] if c not in leido.columns]
        if faltan:
            return self._mal(f"Al CSV le faltan columnas: {faltan}. Guárdalo con index=False.")
        self._ok(f"{ruta}: {len(leido)} filas. Es el puente hacia R: descárgalo desde el panel de archivos.")

    # --- Bloque 3 ------------------------------------------------------------
    def riesgo(self, df, vol_anual, peor_dia, dias_mas_de_4sd):
        c = df["close"] if "close" in df else df.iloc[:, 0]
        r = np.log(c.astype(float)).diff().dropna()
        esperado = {"vol_anual": r.std() * np.sqrt(365),
                    "peor_dia": c.astype(float).pct_change().min(),
                    "dias_mas_de_4sd": int((((r - r.mean()) / r.std()).abs() > 4).sum())}
        dado = {"vol_anual": vol_anual, "peor_dia": peor_dia, "dias_mas_de_4sd": dias_mas_de_4sd}
        todo_bien = True
        for k, e in esperado.items():
            x = float(dado[k])
            if np.isclose(x, e, rtol=0.02, atol=1e-9):
                continue
            todo_bien = False
            if k != "dias_mas_de_4sd" and np.isclose(x, e * 100, rtol=0.02):
                self._mal(f"{k} = {x}: parece estar en porcentaje. Usa fracciones (0.45, no 45).")
            elif k == "vol_anual" and np.isclose(x, e / np.sqrt(365), rtol=0.02):
                self._mal("vol_anual parece la volatilidad DIARIA: multiplícala por la raíz de 365.")
            elif k == "vol_anual" and np.isclose(x, e / np.sqrt(365) * np.sqrt(252), rtol=0.02):
                self._mal("Usaste 252 días: eso es para bolsa. Las criptos cotizan los 365 días.")
            else:
                self._mal(f"{k} = {x:.4f}, pero con estos datos sale {e:.4f}.")
        if todo_bien:
            from math import erfc, sqrt
            normal = erfc(4 / sqrt(2)) * len(r)
            self._ok(f"Volatilidad anual {esperado['vol_anual']:.0%}; peor día {esperado['peor_dia']:.1%}.")
            print(f"     Días de más de 4 desviaciones: {esperado['dias_mas_de_4sd']}. Si los rendimientos fueran")
            print(f"     normales, esperaríamos {normal:.2f}. Las caídas extremas son mucho más frecuentes")
            print("     de lo que promete la campana de Gauss: eso es riesgo que un modelo ingenuo no ve.")

    # --- Bloque 5 ------------------------------------------------------------
    def prediccion(self, exactitud_modelo, exactitud_base):
        m, b = float(exactitud_modelo), float(exactitud_base)
        if m > 1 or b > 1:
            m, b = m / 100, b / 100
        if not (0 <= m <= 1 and 0 <= b <= 1):
            return self._mal("Las exactitudes deben estar entre 0 y 1.")
        print(f"Modelo: {m:.1%} de aciertos | Siempre decir 'sube': {b:.1%}")
        if m > 0.60:
            self._mal("Más del 60% es sospechoso. ¿Se mezclaron fechas del futuro en el entrenamiento"
                      " (división aleatoria) o una variable calculada con el precio de mañana?")
        elif m - b > 0.03:
            self._ok("El modelo supera a la base por más de 3 puntos. Antes de creerlo: ¿se repite con otro"
                     " periodo y con otra moneda? Un solo resultado puede ser suerte.")
        else:
            self._ok("El modelo no le gana a la estrategia tonta. No es un fallo del código: es la lección.")
            print("     La dirección del precio casi no se puede predecir. Su volatilidad sí,")
            print("     y por eso medir y controlar el riesgo funciona donde predecir no.")


verificar = _Verificador()

# Lo que los verificadores esperan encontrar, con el nombre que piden los prompts.
_NOMBRES = {"valida_antes", "valida_despues", "df", "vol_anual", "peor_dia",
            "dias_mas_de_4sd", "exactitud_modelo", "exactitud_base"}


def _falta_un_nombre(shell, etype, evalue, tb, tb_offset=None):
    """Un NameError explicado para quien no programa, en lugar de la traza completa."""
    nombre = getattr(evalue, "name", None) or "?"
    print(f"[REVISAR] Python no conoce «{nombre}».")
    if nombre in _NOMBRES:
        print("     ¿Ejecutaste antes la celda con el código de Claude? Si ya lo hiciste, Claude usó")
        print(f"     otro nombre: pídele «guarda el resultado en una variable llamada {nombre}».")
    else:
        print("     Suele faltar un import o una celda sin ejecutar. Copia este mensaje y pégaselo a Claude.")
    return [f"{etype.__name__}: {evalue}"]


try:
    get_ipython().set_custom_exc((NameError,), _falta_un_nombre)
except NameError:  # fuera de Colab o Jupyter
    pass
print("Listo. Ya puedes seguir con el Bloque 1.")
'''

# ---------------------------------------------------------------------------
# Soluciones (solo en el cuaderno del docente)
# ---------------------------------------------------------------------------
SOL_BLOCKCHAIN = '''import hashlib
import json
from datetime import datetime, timezone


def hash_bloque(bloque):
    """Huella SHA-256 de TODO el bloque, incluido el hash del bloque anterior."""
    texto = json.dumps(bloque, sort_keys=True)
    return hashlib.sha256(texto.encode()).hexdigest()


def nuevo_bloque(datos, anterior):
    return {
        "indice": anterior["indice"] + 1 if anterior else 0,
        "fecha": datetime.now(timezone.utc).isoformat(),
        "datos": datos,
        "hash_anterior": hash_bloque(anterior) if anterior else "0" * 64,
    }


def verificar_cadena(cadena):
    for i in range(1, len(cadena)):
        if cadena[i]["hash_anterior"] != hash_bloque(cadena[i - 1]):
            print(f"Cadena rota: el bloque {i} no reconoce al bloque {i - 1}")
            return False
    return True


cadena = [nuevo_bloque("Bloque génesis", None)]
for t in ["Ana paga 2 BTC a Luis", "Luis paga 1 BTC a Marta", "Marta paga 0.5 BTC a Ana"]:
    cadena.append(nuevo_bloque(t, cadena[-1]))
for b in cadena:
    print(b["indice"], b["datos"].ljust(26), "hash anterior:", b["hash_anterior"][:16], "...")

valida_antes = verificar_cadena(cadena)
cadena[1]["datos"] = "Ana paga 200 BTC a Luis"   # falsificación, sin recalcular nada
valida_despues = verificar_cadena(cadena)
print("Antes:", valida_antes, "| Después de falsificar:", valida_despues)
'''

SOL_DESCARGA = '''import requests

URL = "https://data-api.binance.vision/api/v3/klines"   # api.binance.com bloquea Colab
inicio = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
filas = []
while True:
    r = requests.get(URL, params={"symbol": "BTCUSDT", "interval": "1d",
                                  "startTime": inicio, "limit": 1000}, timeout=30)
    r.raise_for_status()
    lote = r.json()
    if not lote:
        break
    filas += lote
    if len(lote) < 1000:
        break
    inicio = lote[-1][0] + 1          # la página siguiente empieza tras la última vela

columnas = ["apertura_ms", "open", "high", "low", "close", "volume",
            "cierre_ms", "q", "n", "tb", "tq", "x"]
crudo = pd.DataFrame(filas, columns=columnas)
ahora_ms = pd.Timestamp.now(tz="UTC").timestamp() * 1000
crudo = crudo[crudo["cierre_ms"] < ahora_ms]          # solo velas ya cerradas

df = pd.DataFrame({"fecha": pd.to_datetime(crudo["apertura_ms"], unit="ms", utc=True)})
for c in ["open", "high", "low", "close", "volume"]:
    df[c] = crudo[c].astype(float)
df = df.drop_duplicates("fecha").reset_index(drop=True)
df.tail()
'''

SOL_GRAFICO = '''ult = df.tail(90)
color = ["green" if c >= o else "red" for o, c in zip(ult["open"], ult["close"])]
fig, ax = plt.subplots(figsize=(12, 5))
ax.vlines(ult["fecha"], ult["low"], ult["high"], color=color, linewidth=1)      # mechas
ax.bar(ult["fecha"], (ult["close"] - ult["open"]).abs(),                        # cuerpos
       bottom=ult[["open", "close"]].min(axis=1), color=color, width=0.7)
ax.set_title("BTC/USDT, últimos 90 días (velas diarias)")
ax.grid(alpha=0.3)
plt.show()

df.to_csv("BTCUSDT_1d.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
print("Guardado BTCUSDT_1d.csv")
'''

SOL_RIESGO = '''from scipy.stats import norm

r = np.log(df["close"]).diff().dropna()           # rendimientos logarítmicos diarios
vol_diaria = r.std()
vol_anual = vol_diaria * np.sqrt(365)             # cripto cotiza los 365 días
peor_dia = df["close"].pct_change().min()         # peor rendimiento simple de un día

z = (r - r.mean()) / r.std()
dias_mas_de_3sd = int((z.abs() > 3).sum())
dias_mas_de_4sd = int((z.abs() > 4).sum())
esperados_3 = 2 * norm.sf(3) * len(r)
esperados_4 = 2 * norm.sf(4) * len(r)

print(f"Volatilidad anual: {vol_anual:.1%}   Peor día: {peor_dia:.1%}")
print(f"Más de 3 desviaciones: {dias_mas_de_3sd} días (una normal esperaría {esperados_3:.1f})")
print(f"Más de 4 desviaciones: {dias_mas_de_4sd} días (una normal esperaría {esperados_4:.2f})")

x = np.linspace(-8, 8, 400)
plt.figure(figsize=(10, 4))
plt.hist(z, bins=100, density=True, alpha=0.6, label="BTC")
plt.plot(x, norm.pdf(x), label="normal")
plt.yscale("log"); plt.ylim(1e-5, 1); plt.legend()
plt.title("Rendimientos diarios estandarizados (eje vertical logarítmico)")
plt.show()
'''

SOL_PREDICCION = '''from sklearn.ensemble import RandomForestClassifier

d = df.copy()
d["r"] = np.log(d["close"]).diff()
for k in range(1, 6):
    d[f"r_hace_{k - 1}"] = d["r"].shift(k - 1)        # hoy, ayer, ... hace 4 días
d["vol20"] = d["r"].rolling(20).std()
delta = d["close"].diff()
ganancia = delta.clip(lower=0).rolling(14).mean()
perdida = (-delta.clip(upper=0)).rolling(14).mean()
d["rsi"] = 100 - 100 / (1 + ganancia / perdida)
d["sube_manana"] = (d["close"].shift(-1) > d["close"]).astype(int)
d = d.iloc[:-1].dropna()                            # la última fila no tiene "mañana"

variables = [f"r_hace_{k}" for k in range(5)] + ["vol20", "rsi"]
corte = int(len(d) * 0.7)                           # pasado para entrenar, futuro para probar
entreno, prueba = d.iloc[:corte], d.iloc[corte:]

modelo = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)
modelo.fit(entreno[variables], entreno["sube_manana"])
exactitud_modelo = (modelo.predict(prueba[variables]) == prueba["sube_manana"]).mean()
exactitud_base = (prueba["sube_manana"] == 1).mean()  # decir siempre "sube"
print(f"Modelo: {exactitud_modelo:.1%}   Siempre 'sube': {exactitud_base:.1%}")
'''

PLAN_B = '''# PLAN B: si la descarga falla, usa el CSV de respaldo que compartió el docente.
# 1) Abre el panel de archivos (icono de carpeta, a la izquierda) y arrastra BTCUSDT_1d.csv.
# 2) Quita el # de las tres líneas de abajo y ejecuta la celda.
# df = pd.read_csv("BTCUSDT_1d.csv", parse_dates=["fecha"])
# df = df[df["fecha"] >= "2023-01-01"].reset_index(drop=True)
# df.tail()
'''

PEGA = "# Pega aquí el código que te dio Claude y ejecuta la celda (Shift + Enter)\n"

# ---------------------------------------------------------------------------
# Guion: (tipo, contenido, solucion)
# ---------------------------------------------------------------------------
GUION = [
    ("md", """# Clase 1 · Blockchain y criptoactivos con IA

En esta clase no vas a escribir código a mano: se lo vas a pedir a **Claude**, lo vas a pegar aquí y vas a
comprobar si funciona. Esa última parte es la importante: la IA se equivoca, y un **verificador** te dirá qué revisar.

**Cómo trabajar**
1. Abre [claude.ai](https://claude.ai) en otra pestaña.
2. Copia el *prompt* de cada bloque (el texto del recuadro gris), pégalo en Claude y envíalo.
3. Copia el código que te devuelva, pégalo en la celda que dice *Pega aquí* y ejecútala con **Shift + Enter**.
4. Ejecuta la celda **verificar** que viene debajo.
5. Si aparece un error, cópialo entero y pégaselo a Claude: *«Me salió este error en Google Colab: …»*.

Nada de esto es consejo de inversión: aprendemos a **medir**, no a apostar."""),
    ("code", PREPARACION, None),

    ("md", """## Bloque 1 · ¿Qué es una blockchain? (20 min)

Una blockchain es una lista de bloques donde **cada bloque guarda la huella (hash) del anterior**. Vamos a
construir una y a intentar falsificarla.

**Prompt para Claude:**
```
Escribe en Python, para Google Colab y usando solo la librería estándar, una mini-blockchain:
- cada bloque es un diccionario con indice, fecha, datos (un texto) y hash_anterior;
- una función hash_bloque que calcule el SHA-256 de todo el bloque;
- una función verificar_cadena que diga si cada bloque guarda el hash correcto del anterior.
Crea un bloque génesis y 3 bloques con transacciones como "Ana paga 2 BTC a Luis" e imprime la cadena.
Guarda en valida_antes el resultado de verificar la cadena. Luego cambia el texto del bloque 1 por
"Ana paga 200 BTC a Luis" SIN recalcular ningún hash y guarda en valida_despues el resultado de verificar
otra vez. Explica cada línea con comentarios en español para alguien que nunca programó.
```"""),
    ("code", PEGA, SOL_BLOCKCHAIN),
    ("code", "verificar.cadena(valida_antes, valida_despues)", None),
    ("md", """**Para pensar:** ¿por qué cambiar el bloque 1 hace que el *bloque 2* deje de cuadrar? ¿Qué tendría que
hacer un falsificador para que la cadena volviera a ser válida, y por qué en Bitcoin eso es carísimo?"""),

    ("md", """## Bloque 2 · Precios reales de Binance (25 min)

Binance publica sus precios gratis en `data-api.binance.vision`. Cada fila es una **vela**: precio de apertura,
máximo, mínimo, cierre y volumen de un día.

**Prompt para Claude:**
```
Escribe código Python para Google Colab que descargue las velas diarias de BTCUSDT de Binance
usando la URL https://data-api.binance.vision/api/v3/klines (no uses api.binance.com: está bloqueada
en Colab). Pide los datos desde el 1 de enero de 2023 hasta hoy, en páginas de 1000 velas.
Guarda el resultado en un DataFrame de pandas llamado df con las columnas fecha (la hora de APERTURA,
en UTC), open, high, low, close y volume, con los precios como números. Elimina la vela de hoy si
todavía no ha cerrado. Usa solo requests y pandas. Comenta cada paso en español.
```"""),
    ("code", PEGA, SOL_DESCARGA),
    ("code", PLAN_B, None),
    ("code", "verificar.velas(df)", None),
    ("md", """**Ahora el gráfico y el CSV.** Prompt para Claude:
```
Con el DataFrame df anterior, dibuja con matplotlib un gráfico de velas japonesas de los últimos
90 días (verde si cerró por encima de la apertura, rojo si no), sin instalar librerías nuevas.
Después guarda df en un archivo llamado BTCUSDT_1d.csv con index=False.
```"""),
    ("code", PEGA, SOL_GRAFICO),
    ("code", 'verificar.csv("BTCUSDT_1d.csv")', None),

    ("md", """## Bloque 3 · ¿Cuánto se puede perder? (25 min)

El **rendimiento** es cuánto sube o baja el precio de un día al siguiente. La **volatilidad** mide cuánto se
mueve. La pregunta de este bloque: ¿las caídas fuertes son raras, como promete la campana de Gauss?

**Prompt para Claude:**
```
Con el DataFrame df (velas diarias de BTC con la columna close):
1. Calcula el rendimiento logarítmico diario con np.log(df["close"]).diff().
2. Calcula la volatilidad anual como la desviación estándar de esos rendimientos por la raíz de 365
   (las criptos cotizan todos los días) y guárdala en vol_anual, como fracción (0.45, no 45).
3. Guarda en peor_dia el peor rendimiento simple de un día (df["close"].pct_change().min()).
4. Estandariza los rendimientos (resta la media y divide por la desviación) y guarda en
   dias_mas_de_4sd cuántos días se alejaron más de 4 desviaciones. Compáralo con lo que esperaría
   una distribución normal para el mismo número de días (scipy.stats.norm).
5. Dibuja un histograma de los rendimientos estandarizados con la curva normal encima y el eje
   vertical en escala logarítmica. Comenta todo en español.
```"""),
    ("code", PEGA, SOL_RIESGO),
    ("code", "verificar.riesgo(df, vol_anual, peor_dia, dias_mas_de_4sd)", None),

    ("md", """## Bloque 4 · El CSV como puente a R (10 min, lo muestra el docente)

Descarga `BTCUSDT_1d.csv` desde el panel de archivos (clic derecho → Descargar). En Posit Cloud, súbelo al
proyecto de la clase y ejecuta `clase01.R`: el mismo archivo, ahora con `dplyr` y `ggplot2`."""),

    ("md", """## Bloque 5 · ¿Puede la IA predecir el precio? (15 min)

**Prompt para Claude:**
```
Con el DataFrame df (velas diarias de BTC), crea en Python un modelo de random forest (scikit-learn)
que prediga si MAÑANA el cierre será mayor que hoy. Usa como variables los rendimientos de los
últimos 5 días, la volatilidad de 20 días y el RSI de 14 días. Entrena con el 70 % más antiguo de los
datos y prueba con el 30 % más reciente, SIN mezclar el orden de las fechas. Guarda la exactitud en
la prueba en exactitud_modelo, y en exactitud_base la de una estrategia tonta que siempre dice "sube".
Comenta cada paso en español.
```"""),
    ("code", PEGA, SOL_PREDICCION),
    ("code", "verificar.prediccion(exactitud_modelo, exactitud_base)", None),

    ("md", """## Cierre

1. ¿Qué parte del código de Claude tuviste que corregir? ¿Cómo te diste cuenta?
2. Si nadie puede predecir si BTC sube mañana, ¿qué **sí** se puede hacer con estos datos?
   (Pista: el Bloque 3 midió algo que cambia despacio y se puede prever.)

**Tarea:** repite los bloques 2 y 3 con ETHUSDT y compara su volatilidad y su peor día con los de BTC."""),
]

NOTA_DOCENTE = """> **Cuaderno del DOCENTE.** Cada celda *Pega aquí* trae la solución de referencia. Sirve para
> ensayar la clase, para proyectarla si un alumno se atasca y como respaldo si Claude no está disponible.
> Ejecutado entero, todos los verificadores deben decir [OK]."""


def construir(docente: bool) -> nbf.NotebookNode:
    celdas = []
    for i, (tipo, contenido, *resto) in enumerate(GUION):
        solucion = resto[0] if resto else None
        if tipo == "md":
            texto = contenido + ("\n\n" + NOTA_DOCENTE if docente and i == 0 else "")
            celdas.append(nbf.v4.new_markdown_cell(texto))
            continue
        codigo = solucion if (docente and solucion) else contenido
        celda = nbf.v4.new_code_cell(codigo)
        if contenido is PREPARACION:
            celda.metadata["cellView"] = "form"  # Colab la muestra plegada
        celdas.append(celda)
    nb = nbf.v4.new_notebook(cells=celdas)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    return nb


if __name__ == "__main__":
    salida = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI
    for nombre, docente in (("alumno.ipynb", False), ("docente.ipynb", True)):
        nbf.write(construir(docente), salida / nombre)
        print("escrito", salida / nombre)
