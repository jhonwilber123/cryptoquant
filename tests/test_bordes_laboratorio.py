"""Casos borde y entradas hostiles del laboratorio.

Cada test corresponde a un fallo real encontrado al explorar entradas
degeneradas: rutas que escapaban de su carpeta, parametros imposibles que
producian capital negativo, y datos insuficientes que acababan en un volcado
de pila en vez de en un mensaje.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant import config as cfgmod
from cryptoquant.data import sources
from cryptoquant.data.sources import _synthetic
from cryptoquant.laboratorio import aprendizaje, datos, montecarlo, variables, volatilidad


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "REPORTS_DIR", tmp_path / "reports")

    def no_network(*_a, **_k):
        raise AssertionError("no debe tocar la red")

    monkeypatch.setattr(sources, "_fetch_ccxt", no_network)
    return tmp_path


@pytest.fixture
def velas():
    return _synthetic("LAB/USDT", "1d", "2020-01-01", 900, seed=7)


@pytest.mark.parametrize("malo", ["..\\..\\evil", "../x", "C:\\Windows\\x", "BTC/../../x",
                                  "", "B", "BTC USDT", "btc;rm", "BTC/USDT/X", "ÑU"])
def test_simbolo_no_escapa_de_su_carpeta(sandbox, malo):
    with pytest.raises(ValueError):
        datos.ruta_csv(malo, "1h")


def test_simbolo_valido_se_normaliza(sandbox):
    ruta = datos.ruta_csv("eth", "1h")
    assert ruta.parent == datos.carpeta() and ruta.name == "velas_ETH-USDT_1h.csv"
    with pytest.raises(ValueError):
        datos.ruta_csv("BTC", "7m")


def test_csv_ajeno_da_error_claro(sandbox):
    ruta = datos.carpeta() / "x.csv"
    ruta.write_text("hola,adios\n1,2\n")
    with pytest.raises(ValueError, match="no es un CSV de velas"):
        datos.leer_csv(ruta)


@pytest.mark.parametrize("kw", [{"sigma": 0.0}, {"sigma": -0.02}, {"precio": 0.0},
                                {"capital": 0.0}, {"riesgo_por_operacion": 1.5},
                                {"riesgo_por_operacion": 0.0}])
def test_plan_rechaza_parametros_imposibles(kw):
    a = {"precio": 100.0, "sigma": 0.02, "capital": 1000.0, **kw}
    with pytest.raises(ValueError):
        volatilidad.plan_operacion(a.pop("precio"), a.pop("sigma"), a.pop("capital"), **a)


@pytest.mark.parametrize("acierto,rb,riesgo", [(0.5, 2, 1.0), (0.5, 2, 1.5), (1.2, 2, 0.01),
                                               (-0.1, 2, 0.01), (0.5, 0, 0.01), (0.5, -1, 0.01)])
def test_cuenta_rechaza_parametros_imposibles(acierto, rb, riesgo):
    with pytest.raises(ValueError):
        montecarlo.simular_cuenta(acierto, rb, riesgo, n_sim=10)


def test_trayectorias_con_historia_corta():
    with pytest.raises(ValueError, match="al menos"):
        montecarlo.trayectorias_precio(pd.Series([0.01, -0.01, 0.02]), 100.0, bloque=5)


def test_bosque_con_pocos_datos_o_una_clase(velas):
    with pytest.raises(ValueError, match="filas completas"):
        aprendizaje.entrenar_bosque(variables.preparar_variables(velas.iloc[:150]))
    sube = velas.assign(close=100 * 1.01 ** np.arange(len(velas)))
    with pytest.raises(ValueError, match="una sola clase"):
        aprendizaje.entrenar_bosque(variables.preparar_variables(sube), n_arboles=10)


def test_cli_laboratorio_errores_sin_traceback(sandbox, capsys):
    from cryptoquant.cli import main

    assert main(["laboratorio", "--simbolo", "..\\..\\evil", "--no-plots"]) == 2
    assert "simbolo no valido" in capsys.readouterr().out
    for malo in (["--riesgo", "2"], ["--riesgo", "nan"], ["--acierto", "1.5"], ["--rb", "0"],
                 ["--capital", "inf"]):
        with pytest.raises(SystemExit) as e:
            main(["laboratorio", *malo])
        assert e.value.code == 2


def test_descarga_corta_no_pisa_el_csv_bueno(sandbox, monkeypatch):
    bueno = _synthetic("BTC/USDT", "1d", "2020-01-01", 900, seed=1)
    ruta = datos.exportar_csv(bueno, datos.ruta_csv("BTC", "1d"))
    antes = ruta.read_bytes()
    monkeypatch.setattr(datos, "descargar_velas", lambda *a, **k: bueno.iloc[-60:])
    with pytest.raises(ValueError, match="No se ha tocado"):
        datos.cargar_velas("BTC", "1d", "2026-01-01", descargar=True)
    assert ruta.read_bytes() == antes


def test_el_origen_dice_que_dominio_respondio(sandbox, monkeypatch):
    # En Colab lo que importa es saber si contesto data-api o hubo que ir a los archivos.
    bueno = _synthetic("BTC/USDT", "1d", "2020-01-01", 900, seed=1)
    bueno.attrs["fuente"] = "data.binance.vision (archivos)"
    monkeypatch.setattr(datos, "descargar_velas", lambda *a, **k: bueno)
    _, origen = datos.cargar_velas("BTC", "1d", "2020-01-01", descargar=True)
    assert origen == "data.binance.vision (archivos)"
