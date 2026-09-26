"""Tests de la medicion del riesgo.

Una medida de riesgo equivocada es peor que ninguna: da una seguridad que no
existe. Aqui se comprueba que cada pieza da la respuesta conocida en casos
donde se conoce, que ningun pronostico mira al futuro, que las posiciones del
modelo se reconstruyen exactamente, y que el filtro de evidencia rechaza lo
que debe rechazar sin ser imposible de pasar.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cryptoquant import config as cfgmod
from cryptoquant.backtest.engine import WalkForwardBacktest
from cryptoquant.config import Config
from cryptoquant.data.sources import _synthetic
from cryptoquant.riesgo import cartera, evidencia, futuro, informe, pruebas, var


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(cfgmod, "DATA_DIR", tmp_path / "data")
    return tmp_path


def _retornos(n=1500, gl=4, seed=0, activos=("A", "B")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    x = rng.standard_t(gl, size=(n, len(activos))) * 0.02 / np.sqrt(gl / (gl - 2))
    return pd.DataFrame(x, index=idx, columns=list(activos))


# --------------------------------------------------------------------------
# Pruebas estadisticas
# --------------------------------------------------------------------------
def test_kupiec_valor_conocido():
    exc = np.zeros(250, dtype=bool)
    exc[:5] = True  # 5 excepciones en 250 dias al 99%: el doble de lo esperado
    k = pruebas.kupiec(exc, 0.99)
    assert k["lr_kupiec"] == pytest.approx(1.9568, abs=1e-3)
    assert k["p_kupiec"] == pytest.approx(stats.chi2.sf(1.9568, 1), abs=1e-3)


def test_christoffersen_detecta_rachas():
    # Independientes de verdad: al azar. (Cada 20 dias exactos tampoco lo es:
    # nunca van seguidas, y Christoffersen lo detecta con razon.)
    sueltas = np.random.default_rng(0).random(1000) < 0.05
    rachas = np.zeros(1000, dtype=bool)
    for i in range(0, 1000, 200):
        rachas[i:i + 10] = True
    assert pruebas.christoffersen(sueltas)["p_independencia"] > 0.05
    assert pruebas.christoffersen(rachas)["p_independencia"] < 0.001


# --------------------------------------------------------------------------
# VaR
# --------------------------------------------------------------------------
def test_var_normal_analitico():
    x = np.random.default_rng(1).normal(0.001, 0.02, 5000)
    v, es = var.var_es(x, "normal", 0.95)
    mu, sd = x.mean(), x.std(ddof=1)
    assert v == pytest.approx(-(mu + sd * stats.norm.ppf(0.05)))
    assert es > v


def test_t_con_muchos_grados_es_normal():
    q, es = var.cuantil_t(1e6, 0.05)
    assert q == pytest.approx(stats.norm.ppf(0.05), abs=1e-4)
    assert es == pytest.approx(-stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05, abs=1e-4)


def test_pronostico_causal():
    R = _retornos(800)
    w = pd.Series({"A": 0.6, "B": 0.4})
    completo = var.pronosticar(R, w, metodos=("historico", "normal", "t"))
    corto = var.pronosticar(R.iloc[:600], w, metodos=("historico", "normal", "t"))
    pd.testing.assert_frame_equal(completo.loc[corto.index], corto)


def test_en_efectivo_el_var_es_cero():
    R = _retornos(400)
    pos = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    t = var.pronosticar(R, pos, metodos=("historico", "garch_t"))
    assert (t[["var_historico", "var_garch_t"]] == 0).all().all()
    assert pruebas.evaluar(t, 0.95, metodos=("historico",))["n"].iloc[0] == 0


def test_var_calibrado_en_datos_conocidos():
    """Sobre t(4) i.i.d., el VaR t acierta la tasa y el normal al 99% se queda corto."""
    R = _retornos(3000, gl=4, seed=3, activos=("A",))
    t = var.pronosticar(R, pd.Series({"A": 1.0}), alpha=0.99, metodos=("normal", "t"))
    ev = pruebas.evaluar(t, 0.99, metodos=("normal", "t"))
    assert ev.loc["t", "p_kupiec"] > 0.05
    assert ev.loc["normal", "tasa"] > ev.loc["t", "tasa"]


def test_var_siguiente_escala_con_la_exposicion():
    R = _retornos(600)
    lleno = var.var_siguiente(R, pd.Series({"A": 1.0, "B": 0.0}), metodos=("historico", "normal"))
    mitad = var.var_siguiente(R, pd.Series({"A": 0.5, "B": 0.0}), metodos=("historico", "normal"))
    np.testing.assert_allclose(mitad["var"], lleno["var"] / 2)


# --------------------------------------------------------------------------
# Posiciones del modelo
# --------------------------------------------------------------------------
def test_posiciones_reproducen_los_retornos_del_backtest():
    cfg = Config()
    cfg.backtest.warmup = 260
    ohlcv, closes = {}, {}
    for i in range(3):
        df = _synthetic(f"A{i}/USDT", "1d", "2020-01-01", 600, seed=100 + i)
        ohlcv[f"A{i}"], closes[f"A{i}"] = df, df["close"]
    closes = pd.DataFrame(closes)
    res = WalkForwardBacktest(cfg, "hrp", use_ml=False).run(ohlcv, closes)
    R = closes.pct_change().fillna(0.0)
    pos = informe.posiciones_modelo(res.targets, R.index, res.returns.index[-1]).iloc[:-1]
    implicito = (R.shift(-1).loc[pos.index] * pos).sum(axis=1).to_numpy()
    np.testing.assert_allclose(implicito - res.costs.to_numpy(), res.returns.to_numpy(), atol=1e-12)


# --------------------------------------------------------------------------
# Futuro y cartera
# --------------------------------------------------------------------------
def test_riesgo_futuro_mas_volatil_mas_caidas():
    r = _retornos(1000)["A"]
    t = futuro.riesgo_futuro({"baja": r * 0.2, "alta": r}, horizontes=(63,), n=2000)
    t = t.set_index("serie")
    assert t.loc["alta", "prob_caida_10"] > t.loc["baja", "prob_caida_10"]


def test_parsear_cartera():
    assert cartera.parsear("btc=0.5; ETH=2, btc=0.5") == {"BTC": 1.0, "ETH": 2.0}
    with pytest.raises(ValueError):
        cartera.parsear("BTC 0.5")
    with pytest.raises(ValueError):
        cartera.parsear("BTC=-1")


def test_cartera_suma_y_efectivo():
    closes = 100 * np.exp(_retornos(700).cumsum())
    r = cartera.analizar({"A": 10, "B": 5, "USDT": 1000}, closes, n_sim=500)
    p = r.posiciones
    assert r.total == pytest.approx(p["valor"].sum())
    assert p["peso"].sum() == pytest.approx(1.0)
    assert p.loc["efectivo", "contribucion_riesgo"] == 0
    assert p["contribucion_riesgo"].sum() == pytest.approx(1.0)
    # Mas efectivo, menos VaR en proporcion al total.
    r2 = cartera.analizar({"A": 10, "B": 5, "USDT": 100_000}, closes, n_sim=500)
    assert (r2.var_manana["var_pct"] < r.var_manana["var_pct"]).all()


# --------------------------------------------------------------------------
# Filtro de evidencia
# --------------------------------------------------------------------------
def _velas_con_memoria(n=2000, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * 0.02
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = rho * r[i - 1] + e[i]
    c = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)


def _momentum(df):
    return np.sign(np.log(df["close"]).diff()).fillna(0.0)


def test_evidencia_rechaza_lo_que_mira_al_futuro(sandbox):
    velas = {"X": _velas_con_memoria()}
    trampa = lambda df: np.sign(df["close"].shift(-1) - df["close"]).fillna(0.0)  # noqa: E731
    v = evidencia.evaluar_senal("trampa", trampa, velas)
    assert not v.causal and not v.pasa


def test_evidencia_rechaza_el_azar(sandbox):
    velas = {k: _velas_con_memoria(seed=s) for k, s in (("X", 1), ("Y", 2))}
    v = evidencia.evaluar_senal("momentum", _momentum, velas, horizonte=1)
    assert v.causal and not v.pasa


def test_evidencia_no_es_imposible(sandbox):
    """Con memoria real en los retornos, el momentum debe pasar."""
    velas = {k: _velas_con_memoria(rho=0.3, seed=s) for k, s in (("X", 1), ("Y", 2))}
    v = evidencia.evaluar_senal("momentum", _momentum, velas, horizonte=1, coste_ida_vuelta=0.0)
    assert v.pasa, v.motivos


def test_evidencia_cada_idea_endurece_el_umbral(sandbox):
    velas = {"X": _velas_con_memoria()}
    a = evidencia.evaluar_senal("a", _momentum, velas)
    b = evidencia.evaluar_senal("b", _momentum, velas)
    evidencia.evaluar_senal("b", _momentum, velas)  # repetir la misma no cuenta como otra
    assert (a.ideas_evaluadas, b.ideas_evaluadas, evidencia.ideas_evaluadas()) == (1, 2, 2)
    assert b.umbral == pytest.approx(0.025)
    lineas = (sandbox / "data" / "evidencia" / "registro.jsonl").read_text().splitlines()
    assert len(lineas) == 3 and json.loads(lineas[0])["pasa"] is False
