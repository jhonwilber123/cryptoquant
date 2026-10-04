"""Control de volatilidad (paso 9) y fuentes de datos del laboratorio.

El control debe ser causal y cumplir lo que promete: acercar la volatilidad
al objetivo sin apalancarse. Las fuentes deben degradar con orden cuando
Binance bloquea la IP (Colab, Posit Cloud) y leer bien los archivos masivos,
que desde 2025 fechan en microsegundos.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from cryptoquant.data import sources
from cryptoquant.data.sources import _synthetic
from cryptoquant.laboratorio import control, datos


def _ret(n=1500, seed=3):
    """Retornos con agrupamiento de volatilidad (GARCH), como los de cripto."""
    rng = np.random.default_rng(seed)
    h, r = 0.03 ** 2, np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(h) * rng.standard_t(5) / np.sqrt(5 / 3)
        h = 0.03 ** 2 * 0.05 + 0.10 * r[i] ** 2 + 0.85 * h
    return pd.Series(r, index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"))


# --------------------------------------------------------------------------
# Control de volatilidad
# --------------------------------------------------------------------------
def test_control_es_causal():
    r = _ret()
    completo = control.control_volatilidad(r)
    corto = control.control_volatilidad(r.iloc[:1000])
    comun = corto.index
    pd.testing.assert_series_equal(completo.loc[comun, "exposicion"], corto["exposicion"])


def test_control_se_acerca_al_objetivo_sin_apalancarse():
    t = control.control_volatilidad(_ret(), objetivo_anual=0.15)
    res = control.resumen(t)
    assert t["exposicion"].between(0, 1).all()
    vol_ctl, vol_pas = res.loc["estrategia", "volatilidad"], res.loc["pasiva", "volatilidad"]
    assert abs(vol_ctl - 0.15) < abs(vol_pas - 0.15)
    assert abs(vol_ctl - 0.15) < 0.05
    assert res.loc["pasiva_igual_vol", "volatilidad"] == pytest.approx(vol_ctl, rel=1e-9)


def test_control_cobra_costes():
    r = _ret()
    con = control.resumen(control.control_volatilidad(r, coste=0.01))
    sin = control.resumen(control.control_volatilidad(r, coste=0.0))
    assert con.loc["estrategia", "capital_final"] < sin.loc["estrategia", "capital_final"]


@pytest.mark.parametrize("kw", [{"objetivo_anual": 0}, {"objetivo_anual": -0.1},
                                {"max_exposicion": 1.5}, {"lam": 1.0}])
def test_control_rechaza_parametros_imposibles(kw):
    with pytest.raises(ValueError):
        control.control_volatilidad(_ret(200), **kw)


# --------------------------------------------------------------------------
# Fuentes de datos
# --------------------------------------------------------------------------
def _zip(lineas: list[str]) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.csv", "\n".join(lineas) + "\n")
    return b.getvalue()


def test_a_principios_de_mes_los_dias_del_mes_pasado_salen_de_los_diarios(monkeypatch):
    """El 1 de octubre aun no hay archivo mensual de septiembre: sus dias, de los diarios."""
    fila = "{t},100,110,90,105,7,0,0,0,0,0,0"

    def dias(desde, hasta):
        return _zip([fila.format(t=d.value // 10**6)
                     for d in pd.date_range(desde, hasta, freq="D", tz="UTC")])

    pedidos = []

    def bajar(url):
        pedidos.append(url)
        nombre = url.rsplit("/", 1)[1].removesuffix(".zip")       # BTCUSDT-1d-2026-08(-01)
        fecha = nombre.split("-1d-")[1]
        if "/monthly/" in url:
            mes = pd.Period(fecha, "M")
            return dias(mes.start_time, mes.end_time.normalize()) if mes <= pd.Period("2026-08", "M") else None
        return dias(fecha, fecha)

    monkeypatch.setattr(datos, "_bajar", bajar)
    df = datos.descargar_archivos("BTC", "1d", "2026-07-01", hoy=pd.Timestamp("2026-10-01 05:00", tz="UTC"))
    assert df.index[-1] == pd.Timestamp("2026-09-30", tz="UTC")
    assert len(df) == 31 + 31 + 30 and df.index.is_unique
    assert sum("/daily/" in u for u in pedidos) == 30                # solo los de septiembre

    pedidos.clear()   # a mitad de mes: el mensual del mes pasado ya existe; diarios, solo los de este
    df = datos.descargar_archivos("BTC", "1d", "2026-07-01", hoy=pd.Timestamp("2026-09-15 05:00", tz="UTC"))
    assert df.index[-1] == pd.Timestamp("2026-09-14", tz="UTC")
    assert sum("/daily/" in u for u in pedidos) == 14


def test_archivos_en_milisegundos_y_microsegundos():
    fila = "{t},100,110,90,105,7,0,0,0,0,0,0"
    ms = pd.Timestamp("2024-12-31", tz="UTC").value // 10**6
    us = pd.Timestamp("2025-01-01", tz="UTC").value // 10**3
    df = datos.leer_archivo_klines(_zip(["open_time,open,high,low,close,volume,a,b,c,d,e,f",
                                         fila.format(t=ms), fila.format(t=us)]))
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-12-31", "2025-01-01"]
    assert df.iloc[0].tolist() == [100.0, 110.0, 90.0, 105.0, 7.0]


def test_se_prueba_el_dominio_de_datos_publicos_primero(monkeypatch):
    usados = []
    bueno = _synthetic("BTC/USDT", "1d", "2024-01-01", 50, seed=1)

    def falso(ex, sym, tf, since, n, public_api=None):
        usados.append(public_api)
        return bueno.copy()
    monkeypatch.setattr(sources, "_fetch_ccxt", falso)
    df = datos.descargar_velas("BTC", "1d", "2024-01-01")
    assert usados == ["https://data-api.binance.vision/api/v3"]
    assert df.attrs["fuente"] == "data-api.binance.vision"


def test_bloqueo_por_ubicacion_cae_a_la_siguiente_fuente(monkeypatch):
    bueno = _synthetic("BTC/USDT", "1d", "2024-01-01", 50, seed=1)

    def solo_principal(ex, sym, tf, since, n, public_api=None):
        if public_api:
            raise ConnectionError("451 restricted location")
        return bueno.copy()
    monkeypatch.setattr(sources, "_fetch_ccxt", solo_principal)
    assert datos.descargar_velas("BTC", "1d", "2024-01-01").attrs["fuente"] == "api.binance.com"


def test_sin_api_quedan_los_archivos(monkeypatch):
    def caida(*a, **k):
        raise ConnectionError("451")
    monkeypatch.setattr(sources, "_fetch_ccxt", caida)
    monkeypatch.setattr(datos, "descargar_archivos",
                        lambda *a: _synthetic("BTC/USDT", "1d", "2024-01-01", 50, seed=1))
    df = datos.descargar_velas("BTC", "1d", "2024-01-01")
    assert df.attrs["fuente"].startswith("data.binance.vision")


def test_todo_caido_da_un_error_claro(monkeypatch):
    def caida(*a, **k):
        raise ConnectionError("451")
    monkeypatch.setattr(sources, "_fetch_ccxt", caida)
    monkeypatch.setattr(datos, "_bajar", lambda url: None)  # ningun archivo
    with pytest.raises(ValueError, match="no se pudieron descargar"):
        datos.descargar_velas("BTC", "1d", "2024-01-01")


def test_el_sistema_sigue_usando_su_dominio(monkeypatch):
    """public_api es opcional: sin el, _fetch_ccxt no cambia el dominio."""
    import ccxt

    creado = {}

    class Falso:
        def __init__(self, cfg):
            self.urls = {"api": {"public": "https://api.binance.com/api/v3"}}
            self.rateLimit = 0
            creado["ex"] = self

        def fetch_ohlcv(self, *a, **k):
            return []
    monkeypatch.setattr(ccxt, "binance", Falso)
    with pytest.raises(RuntimeError):
        sources._fetch_ccxt("binance", "BTC/USDT", "1d", 0, 10)
    assert creado["ex"].urls["api"]["public"] == "https://api.binance.com/api/v3"


def test_cli_objetivo_fuera_de_rango():
    from cryptoquant.cli import main

    for malo in ("0", "-1", "nan"):
        with pytest.raises(SystemExit):
            main(["laboratorio", "--objetivo", malo])
