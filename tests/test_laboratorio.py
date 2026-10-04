"""Tests del laboratorio.

Como en el nucleo, lo esencial es que ninguna variable explicativa mire al
futuro: si lo hace, el random forest "acierta" y la leccion es falsa. El resto
comprueba que cada pieza calcula lo que dice sobre casos con respuesta
conocida, y que el laboratorio jamas toca la cache del sistema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant import config as cfgmod
from cryptoquant.data import sources
from cryptoquant.data.sources import _synthetic
from cryptoquant.laboratorio import (
    aprendizaje,
    cuantitativo,
    datos,
    informe,
    montecarlo,
    series,
    variables,
    volatilidad,
)


@pytest.fixture
def velas():
    return _synthetic("LAB/USDT", "1d", "2020-01-01", 900, seed=7)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Informes y cache en un directorio temporal, y la red prohibida."""
    monkeypatch.setattr(cfgmod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()

    def no_network(*_a, **_k):
        raise AssertionError("no debe tocar la red")

    monkeypatch.setattr(sources, "_fetch_ccxt", no_network)
    return tmp_path


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------
def test_variables_are_causal(velas):
    """Truncar el futuro no puede cambiar ninguna variable explicativa."""
    k = 600
    full = variables.preparar_variables(velas)
    cut = variables.preparar_variables(velas.iloc[:k])
    cols = [c for c in full.columns if c not in ("retorno_futuro", "sube")]
    pd.testing.assert_frame_equal(full[cols].iloc[:k], cut[cols], check_exact=False, atol=1e-10)


def test_objetivo_mira_a_la_vela_siguiente(velas):
    v = variables.preparar_variables(velas)
    esperado = (velas["close"].shift(-1) > velas["close"]).astype(float)
    pd.testing.assert_series_equal(v["sube"].iloc[:-1], esperado.iloc[:-1], check_names=False)
    assert np.isnan(v["sube"].iloc[-1])
    assert not {"sube", "retorno_futuro"} & set(variables.VARIABLES_MODELO)


def test_indicadores_en_casos_conocidos(velas):
    v = variables.preparar_variables(velas).dropna()
    assert ((v["rsi_14"] >= 0) & (v["rsi_14"] <= 100)).all()
    assert (v["bb_sup"] > v["bb_media"]).all() and (v["bb_inf"] < v["bb_media"]).all()
    np.testing.assert_allclose(v["macd_hist"], v["macd"] - v["macd_senal"])
    np.testing.assert_allclose(v["retorno_l1"].iloc[1:], v["retorno"].iloc[:-1])


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def test_csv_ida_y_vuelta(sandbox, velas):
    ruta = datos.exportar_csv(velas, datos.ruta_csv("LAB/USDT", "1d"))
    assert ruta.read_text().splitlines()[0] == "fecha,open,high,low,close,volume"
    pd.testing.assert_frame_equal(datos.leer_csv(ruta), velas, check_freq=False,
                                  check_index_type=False, check_names=False)


def test_a_diario_descarta_el_dia_incompleto():
    idx = pd.date_range("2024-01-01", periods=24 * 3 + 5, freq="h", tz="UTC")
    h = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    d = datos.a_diario(h)
    assert len(d) == 3
    assert (d["volume"] == 24).all()


# --------------------------------------------------------------------------
# Cuantitativo y series
# --------------------------------------------------------------------------
def test_colas_normales_no_se_inflan():
    r = pd.Series(np.random.default_rng(0).standard_normal(200_000))
    colas = cuantitativo.tabla_colas(r)
    assert colas.loc[2, "veces_mas"] == pytest.approx(1.0, rel=0.05)
    assert cuantitativo.distribucion_retornos(r)["curtosis_exceso"] == pytest.approx(0, abs=0.05)


def test_senal_de_compra_en_su_vela(velas):
    v = variables.preparar_variables(velas)
    ev = cuantitativo.senales(v)
    dorado = ev.index[ev["cruce_dorado_20_50"] == 1]
    assert len(dorado)
    for t in dorado:
        i = v.index.get_loc(t)
        assert v["sma_20"].iloc[i] > v["sma_50"].iloc[i]
        assert v["sma_20"].iloc[i - 1] <= v["sma_50"].iloc[i - 1]
    assert set(np.unique(ev.to_numpy())) <= {-1, 0, 1}


def test_precio_no_estacionario_retorno_si(velas):
    t = series.precio_frente_a_retorno(velas["close"])
    assert t.loc["retorno_log", "adf_p"] < 0.01


def test_arima_no_mejora_ruido_blanco():
    r = pd.Series(np.random.default_rng(1).standard_normal(1500) * 0.02)
    a = series.arima_basico(r)
    assert a["mejora_sobre_cero"] < 0.02


# --------------------------------------------------------------------------
# Volatilidad
# --------------------------------------------------------------------------
def test_garch_recupera_parametros():
    rng = np.random.default_rng(3)
    omega, alpha, beta = 0.05, 0.10, 0.85
    n, h, r = 4000, omega / (1 - alpha - beta), []
    for _ in range(n):
        x = np.sqrt(h) * rng.standard_normal()
        r.append(x)
        h = omega + alpha * x * x + beta * h
    g = volatilidad.pronostico_garch(pd.Series(r) / 100, dist="normal")
    assert g["alpha"] == pytest.approx(alpha, abs=0.04)
    assert g["beta"] == pytest.approx(beta, abs=0.06)


def test_plan_arriesga_lo_pactado():
    p = volatilidad.plan_operacion(100.0, 0.03, 10_000, riesgo_por_operacion=0.01)
    assert p["perdida_si_stop"] == pytest.approx(100.0)
    assert p["objetivo"] - p["precio"] == pytest.approx(2 * (p["precio"] - p["stop"]))
    # Mas volatilidad: stop mas lejos y posicion mas pequena, misma perdida.
    q = volatilidad.plan_operacion(100.0, 0.06, 10_000, riesgo_por_operacion=0.01)
    assert q["stop"] < p["stop"] and q["nominal"] < p["nominal"]
    assert q["perdida_si_stop"] == pytest.approx(100.0)


def test_plan_sin_apalancamiento():
    p = volatilidad.plan_operacion(100.0, 0.001, 10_000, riesgo_por_operacion=0.05)
    assert p["limitado_por_capital"] and p["nominal"] == pytest.approx(10_000)


# --------------------------------------------------------------------------
# Aprendizaje
# --------------------------------------------------------------------------
def test_bosque_prueba_con_el_futuro(velas):
    v = variables.preparar_variables(velas)
    rf = aprendizaje.entrenar_bosque(v, n_arboles=50)
    X, _ = variables.matriz_modelo(v)
    n_tr = int(len(X) * 0.7)
    assert rf.prueba.index.min() > X.index[n_tr - 1]
    assert rf.prueba.index.min() == X.index[n_tr + 1]  # embargo de una vela
    assert {"random_forest", "persistencia"} <= set(rf.metricas.index)


def test_bosque_no_aprende_de_ruido():
    """Si las variables no dicen nada del futuro, no puede batir a la base."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=3000, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(3000) * 0.02))
    velas = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                          "close": close, "volume": 1.0}, index=idx)
    rf = aprendizaje.entrenar_bosque(variables.preparar_variables(velas), n_arboles=100)
    assert rf.metricas.loc["random_forest", "auc"] < 0.56


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def test_cuenta_con_esperanza_nula_no_crece():
    # acierto 1/3 con R:B 1:2 -> esperanza 0 R por operacion.
    c = montecarlo.simular_cuenta(1 / 3, 2.0, 0.01, n_operaciones=300, n_sim=4000)
    r = montecarlo.resumen_cuenta(c, 1 / 3, 2.0, 0.01)
    assert r["esperanza_por_operacion_R"] == pytest.approx(0.0)
    # Con fraccion fija, esperanza 0 hace que la mediana final quede por debajo del inicio.
    assert r["capital_mediano_final"] < 10_000


def test_trayectorias_arrancan_en_el_precio(velas):
    ret = np.log(velas["close"]).diff()
    c = montecarlo.trayectorias_precio(ret, 50.0, dias=30, n=200, bloque=5)
    assert c.shape == (200, 31)
    assert (c[:, 0] == 50.0).all()


# --------------------------------------------------------------------------
# Informe completo
# --------------------------------------------------------------------------
def test_informe_no_toca_la_cache_del_sistema(sandbox):
    idx = pd.date_range("2022-01-01", periods=24 * 800, freq="h", tz="UTC")
    rng = np.random.default_rng(9)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(len(idx)) * 0.004))
    h = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                      "close": close, "volume": 1.0}, index=idx)
    datos.exportar_csv(h, datos.ruta_csv("BTC/USDT", "1h"))

    inf = informe.ejecutar("BTC", "1h", graficos=False)
    assert list((sandbox / "cache").iterdir()) == []
    assert all(p.parent == sandbox / "reports" / "laboratorio" for p in inf.archivos)
    assert "hora" in inf.resultados["calendario"]
