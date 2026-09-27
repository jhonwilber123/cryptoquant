"""Velas de Binance: descarga, CSV y grafico.

El CSV es el punto de entrega entre Python y R. Lleva la fecha en UTC y en
formato ISO, sin indice implicito, para que `readr::read_csv` lo lea tal cual.

El laboratorio guarda sus velas aparte (`reports/laboratorio/`) y nunca
escribe en `data/cache/`: esa cache alimenta al sistema y al forward test, y
un ejercicio de clase no debe poder alterarla.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfgmod
from ..data import sources
from ..data.sources import _TF_MS

TEMPORALIDADES = tuple(_TF_MS)
COLUMNAS = ["open", "high", "low", "close", "volume"]
# El simbolo acaba formando parte de una ruta: nada de separadores, puntos
# ni unidades. Solo tickers como BTC o BTC/USDT.
_TICKER = re.compile(r"[A-Z0-9]{2,15}")


def carpeta() -> Path:
    # Se resuelve en cada llamada para que los tests puedan redirigir REPORTS_DIR.
    d = cfgmod.REPORTS_DIR / "laboratorio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def validar_ticker(t: str) -> str:
    t = t.strip().upper()
    if not _TICKER.fullmatch(t):
        raise ValueError(f"simbolo no valido: {t!r} (use letras y cifras, p. ej. BTC)")
    return t


def par(simbolo: str, quote: str = "USDT") -> str:
    """'btc' -> 'BTC/USDT'. Rechaza todo lo que no sea un ticker."""
    base, _, cotiza = simbolo.partition("/")
    return f"{validar_ticker(base)}/{validar_ticker(cotiza or quote)}"


def ruta_csv(simbolo: str, temporalidad: str) -> Path:
    if temporalidad not in _TF_MS:
        raise ValueError(f"temporalidad no soportada: {temporalidad} (use {TEMPORALIDADES})")
    return carpeta() / f"velas_{par(simbolo).replace('/', '-')}_{temporalidad}.csv"


# Por orden. api.binance.com rechaza (HTTP 451) las IP de EE. UU., que es donde
# suelen estar Colab y Posit Cloud; data-api.binance.vision sirve los mismos
# datos publicos de mercado sin esa restriccion y sin clave.
FUENTES_API = (("data-api.binance.vision", "https://data-api.binance.vision/api/v3"),
               ("api.binance.com", None))
ARCHIVOS = "https://data.binance.vision/data/spot"


def descargar_velas(simbolo: str = "BTC/USDT", temporalidad: str = "1h",
                    desde: str = "2022-01-01", exchange: str = "binance") -> pd.DataFrame:
    """Descarga OHLCV desde `desde` hasta la ultima vela cerrada.

    Prueba las fuentes en orden y se queda con la primera que responde: la API
    de datos publicos, la API principal y, por ultimo, los archivos mensuales
    y diarios de data.binance.vision. La fuente usada queda en
    `df.attrs["fuente"]`. La vela en curso se descarta: su "cierre" es el
    precio de este instante y cambiaria al volver a descargar.
    """
    if temporalidad not in _TF_MS:
        raise ValueError(f"temporalidad no soportada: {temporalidad} (use {TEMPORALIDADES})")
    p = par(simbolo)
    inicio = pd.Timestamp(desde, tz="UTC")
    barras = int((pd.Timestamp.now(tz="UTC") - inicio) / pd.Timedelta(milliseconds=_TF_MS[temporalidad])) + 1
    fuentes = FUENTES_API if exchange == "binance" else ((exchange, None),)
    fallos = []
    for nombre, api in fuentes:
        try:
            df = sources._fetch_ccxt(exchange, p, temporalidad, int(inicio.timestamp() * 1000),
                                     barras, public_api=api)
            df.attrs["fuente"] = nombre
            return df
        except Exception as exc:  # noqa: BLE001 - se prueba la siguiente fuente
            fallos.append(f"{nombre}: {type(exc).__name__}: {str(exc)[:80]}")
    if exchange == "binance":
        try:
            df = descargar_archivos(p, temporalidad, desde)
            df.attrs["fuente"] = "data.binance.vision (archivos)"
            return df
        except Exception as exc:  # noqa: BLE001
            fallos.append(f"archivos: {type(exc).__name__}: {str(exc)[:80]}")
    raise ValueError(f"no se pudieron descargar velas de {p}. " + " | ".join(fallos))


def _bajar(url: str) -> bytes | None:
    """Contenido de `url`, o None si no existe (404). Aislado para los tests."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def leer_archivo_klines(contenido: bytes) -> pd.DataFrame:
    """Un zip de klines de data.binance.vision -> OHLCV con indice UTC.

    Dos trampas: los archivos no traen cabecera (algunos antiguos si), y desde
    2025 las marcas de tiempo de spot vienen en MICROsegundos (16 cifras) en
    lugar de milisegundos (13). Leidas como ms, las fechas de 2025 caen en el
    ano 57 000.
    """
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        crudo = z.read(z.namelist()[0]).decode("utf-8")
    filas = [f.split(",") for f in crudo.splitlines() if f and f[0].isdigit()]
    ts = np.array([int(f[0]) for f in filas], dtype="int64")
    unidad = np.where(ts > 10**14, 1_000, 1)  # microsegundos -> milisegundos
    idx = pd.DatetimeIndex(pd.to_datetime(ts // unidad, unit="ms", utc=True), name="date")
    return pd.DataFrame([[float(v) for v in f[1:6]] for f in filas], index=idx, columns=COLUMNAS)


def descargar_archivos(simbolo: str, temporalidad: str, desde: str) -> pd.DataFrame:
    """Historia larga desde los archivos publicos: mensuales y, el mes en curso, diarios."""
    sym = par(simbolo).replace("/", "")
    hoy = pd.Timestamp.now(tz="UTC").normalize()
    partes = []
    ultimo_mes = (hoy - pd.offsets.MonthBegin(1)).tz_localize(None).to_period("M")
    for mes in pd.period_range(pd.Timestamp(desde).to_period("M"), ultimo_mes, freq="M"):
        b = _bajar(f"{ARCHIVOS}/monthly/klines/{sym}/{temporalidad}/{sym}-{temporalidad}-{mes}.zip")
        if b:
            partes.append(leer_archivo_klines(b))
    for dia in pd.date_range(hoy.replace(day=1), hoy - pd.Timedelta(days=1), freq="D"):
        b = _bajar(f"{ARCHIVOS}/daily/klines/{sym}/{temporalidad}/{sym}-{temporalidad}-{dia:%Y-%m-%d}.zip")
        if b:
            partes.append(leer_archivo_klines(b))
    if not partes:
        raise ValueError(f"data.binance.vision no tiene archivos de {sym} {temporalidad}")
    df = pd.concat(partes).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return sources.drop_incomplete_bars(df.loc[pd.Timestamp(desde, tz="UTC"):], temporalidad)


def exportar_csv(df: pd.DataFrame, ruta: Path) -> Path:
    out = df[COLUMNAS].copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "fecha"
    out.to_csv(ruta)
    return ruta


def leer_csv(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    faltan = [c for c in ["fecha", *COLUMNAS] if c not in df.columns]
    if faltan:
        raise ValueError(f"{ruta.name} no es un CSV de velas: faltan {faltan}. "
                         "Borrelo o use --descargar.")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("fecha"), utc=True), name="date")
    return df[COLUMNAS].astype(float)


MIN_DIAS = 400  # arranque de indicadores (100) + random forest (200) + margen


def cargar_velas(simbolo: str = "BTC", temporalidad: str = "1h", desde: str = "2022-01-01",
                 descargar: bool = False, min_dias: int = MIN_DIAS) -> tuple[pd.DataFrame, str]:
    """Devuelve (velas, origen). Reutiliza el CSV si existe, salvo `descargar`.

    Una descarga con historia insuficiente se rechaza ANTES de escribir: si
    no, un intento fallido sustituiria el CSV bueno (del que depende la
    version en R) por uno inservible.
    """
    ruta = ruta_csv(par(simbolo), temporalidad)
    if ruta.exists() and not descargar:
        return leer_csv(ruta), f"csv ({ruta.name})"
    df = descargar_velas(par(simbolo), temporalidad, desde)
    dias = len(a_diario(df)) if temporalidad != "1d" else len(df)
    if dias < min_dias:
        raise ValueError(f"solo {dias} dias de historia desde {desde}; hacen falta {min_dias}. "
                         f"No se ha tocado {ruta.name}.")
    exportar_csv(df, ruta)
    return df, df.attrs.get("fuente", "binance")


def a_diario(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega velas intradia a velas diarias UTC, como las publica Binance.

    El ultimo dia se descarta si esta incompleto. Los huecos intermedios (el
    exchange para por mantenimiento) no: ese dia existio y cotizo.
    """
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()
    if len(df) >= 2:
        paso = df.index[-1] - df.index[-2]
        if df.index[-1] + paso < d.index[-1] + pd.Timedelta(days=1):
            d = d.iloc[:-1]
    return d


def grafico_velas(df: pd.DataFrame, ax=None, ultimas: int = 90, medias: tuple[int, ...] = (20, 50),
                  titulo: str = ""):
    """Grafico de velas japonesas con matplotlib, sin dependencias extra.

    Las medias se calculan sobre toda la serie y luego se recortan: calcularlas
    sobre el tramo visible dejaria vacio el comienzo del grafico.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    medias_s = {w: df["close"].rolling(w).mean().iloc[-ultimas:] for w in medias}
    v = df.iloc[-ultimas:]
    x = mdates.date2num(v.index.tz_localize(None).to_pydatetime())
    ancho = (x[1] - x[0]) * 0.7 if len(x) > 1 else 0.5
    sube = (v["close"] >= v["open"]).to_numpy()
    color = np.where(sube, "#2e9e6b", "#d1495b")

    ax.vlines(x, v["low"], v["high"], color=color, linewidth=0.8)
    cuerpo = (v["close"] - v["open"]).abs().to_numpy()
    cuerpo = np.where(cuerpo == 0, v["close"].to_numpy() * 1e-4, cuerpo)
    ax.bar(x, cuerpo, bottom=np.minimum(v["open"], v["close"]), width=ancho,
           color=color, edgecolor=color, linewidth=0.5)
    for w, s in medias_s.items():
        ax.plot(x, s.to_numpy(), linewidth=1.1, label=f"media {w}")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(titulo)
    ax.grid(alpha=0.25)
    if medias:
        ax.legend(loc="upper left", frameon=False)
    return ax
