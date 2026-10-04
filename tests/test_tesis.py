"""Tests de los contrastes retrospectivos de la tesis.

No comprueban que el modelo gane: comprueban que los contrastes digan lo que
dicen medir. Que el corte deje fuera el futuro, que el bootstrap sea pareado,
que la comparacion a igual volatilidad sea exacta y que un modelo sin ventaja
no la encuentre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant import tesis
from cryptoquant.backtest.engine import BacktestResult

ANN = 365.0


def _serie(n=800, vol=0.03, seed=1):
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(0.0005, vol, n), index=idx)


def test_recortar_incluye_el_corte_y_nada_despues():
    idx = pd.date_range("2026-09-01", "2026-09-20", freq="D", tz="UTC")
    closes = pd.DataFrame({"BTC": np.arange(len(idx), dtype=float)}, index=idx)
    ohlcv = {"BTC": closes.rename(columns={"BTC": "close"})}
    o, c = tesis.recortar(ohlcv, closes, "2026-09-10")
    assert c.index[-1] == pd.Timestamp("2026-09-10", tz="UTC")
    assert o["BTC"].index[-1] == c.index[-1]


def test_recortar_rechaza_un_corte_posterior_a_los_datos():
    idx = pd.date_range("2026-09-01", "2026-09-05", freq="D", tz="UTC")
    closes = pd.DataFrame({"BTC": 1.0}, index=idx)
    with pytest.raises(ValueError):
        tesis.recortar({"BTC": closes}, closes, "2026-09-10")


def test_caida_maxima_cuenta_desde_el_capital_inicial():
    # Pierde un 50% el primer dia: la caida es -50% aunque la curva empiece abajo.
    assert tesis.caida_maxima(np.array([-0.5, 0.2])) == pytest.approx(-0.5)
    assert tesis.caida_maxima(np.array([0.1, -0.1, 0.05])) == pytest.approx(-0.1)


def test_bloques_contiguos_y_de_la_longitud_pedida():
    idx = tesis.indices_bloques(100, 21, np.random.default_rng(0))
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100
    assert np.all(np.diff(idx[:21]) == 1)


def test_bootstrap_pareado_da_diferencia_nula_con_series_identicas():
    r = _serie().to_numpy()
    dist = tesis.bootstrap(np.column_stack([r, r]),
                           lambda m: tesis.vol_anual(m[:, 0], ANN) - tesis.vol_anual(m[:, 1], ANN), 200)
    assert np.allclose(dist, 0.0)


def test_pasiva_igual_vol_iguala_la_volatilidad():
    modelo, pasiva = _serie(vol=0.01, seed=2), _serie(vol=0.04, seed=3)
    k, igual = tesis.pasiva_igual_vol(modelo, pasiva)
    assert igual.std(ddof=1) == pytest.approx(modelo.std(ddof=1))
    assert k == pytest.approx(0.25, rel=0.1)


def test_he4_no_encuentra_ventaja_en_una_copia_escalada_de_la_pasiva():
    pasiva = _serie(seed=4)
    contrastes, extra = tesis.contraste_he4(0.3 * pasiva, pasiva, ANN, 200)
    assert extra["exceso_anual"] == pytest.approx(0.0, abs=1e-12)
    assert contrastes[0].resultado != "se cumple"


def test_he3_detecta_menos_riesgo_en_una_version_reducida():
    pasiva = _serie(seed=5)
    he3 = tesis.contraste_he3((0.2 * pasiva).to_numpy(), pasiva.to_numpy(), ANN, 300)
    assert [c.resultado for c in he3] == ["se cumple", "se cumple"]


def test_he1_usa_solo_los_dias_invertido():
    r = _serie(n=720, vol=0.15 / np.sqrt(ANN), seed=6)
    expo = pd.Series(1.0, index=r.index)
    expo.iloc[::2] = 0.0
    r[expo == 0.0] = 0.0          # en efectivo no se mueve
    res = BacktestResult(equity=(1 + r).cumprod(), returns=r, weights=pd.DataFrame(),
                         gross_exposure=expo, benchmark_returns=r, diagnostics=pd.DataFrame(),
                         turnover=r * 0, costs=r * 0)
    c, ventanas = tesis.contraste_he1(res, 0.15, ANN, 200)
    assert c.estimacion == pytest.approx(0.15, rel=0.15)
    assert c.resultado == "se cumple"
    assert len(ventanas) == 360 // tesis.VENTANA_H1


def test_huella_cambia_si_cambia_un_retorno():
    r = _serie(n=50)
    otra = r.copy()
    otra.iloc[10] += 1e-6
    assert tesis.huella(r) == tesis.huella(r.copy())
    assert tesis.huella(r) != tesis.huella(otra)


# --------------------------------------------------------------------------
# Ampliacion: HE5 a HE7
# --------------------------------------------------------------------------
def test_var_cvar_como_perdidas_positivas():
    r = np.linspace(-0.10, 0.09, 20)          # 20 retornos; el 5% peor es -0,10
    v, c = tesis.var_cvar(r, 0.95)
    assert v > 0 and c >= v
    assert c == pytest.approx(0.10, abs=1e-9)


def test_he5_frente_a_igual_volatilidad_no_premia_invertir_menos():
    # Una copia escalada de la pasiva tiene menos cola que la pasiva, pero
    # exactamente la misma que la pasiva a su misma volatilidad.
    pasiva = _serie(seed=6)
    modelo = 0.25 * pasiva
    _, igual = tesis.pasiva_igual_vol(modelo, pasiva)
    contrastes, tabla = tesis.contraste_he5(modelo.to_numpy(), pasiva.to_numpy(), igual.to_numpy(), 200)
    assert contrastes[0].resultado == "se cumple"
    assert contrastes[1].estimacion == pytest.approx(0.0, abs=1e-12)
    assert contrastes[1].resultado == "no concluyente"
    assert set(tabla.index) == {"VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"}


def test_volatilidad_fuera_de_muestra_prefiere_garch_cuando_hay_agrupamiento():
    # GARCH(1,1) simulado: la varianza de hoy depende de ayer, asi que la
    # prevision condicional debe mejorar a la varianza constante.
    rng = np.random.default_rng(7)
    n, w, a, b = 1500, 0.02, 0.12, 0.85
    s2, r = w / (1 - a - b), np.empty(n)
    for i in range(n):
        r[i] = np.sqrt(s2) * rng.standard_normal()
        s2 = w + a * r[i] ** 2 + b * s2
    ret = pd.Series(r / 100, index=pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"))
    info, d = tesis.volatilidad_fuera_de_muestra(ret)
    assert len(d) == info["n_prueba"] == n - int(n * 0.7)
    assert info["qlike_garch"] < info["qlike_constante"]
    assert info["dm_p"] < 0.05


def test_arima_no_encuentra_direccion_en_ruido_blanco():
    ret = _serie(n=600, seed=8)
    info = tesis.direccion_arima(ret)
    assert info["r2_fuera_muestra"] < 0.02
    assert info["dm_p"] > 0.05


def test_he7_aplica_bonferroni_sobre_toda_la_familia():
    pred = pd.DataFrame({"arima_dm_p": [0.5, 0.004], "rf_p": [0.3, 0.9]}, index=["A", "B"])
    senales = pd.DataFrame({"p_valor": [0.2, 0.7]})
    conjunta = pd.Series(np.r_[np.full(50, -0.2), np.full(50, -0.1)] + np.random.default_rng(9).normal(0, 0.05, 100))
    a, b = tesis.contraste_he7(pred, conjunta, senales)
    assert a.resultado == "se cumple"
    # 0,004 queda por debajo de 0,05/6 = 0,0083: hay direccion detectable y
    # la parte (b) no se cumple.
    assert b.resultado == "no se cumple" and "0,05/6" in b.criterio


# --------------------------------------------------------------------------
# Historia larga
# --------------------------------------------------------------------------
def test_tramos_halving_cubren_el_periodo_sin_solaparse():
    idx = pd.date_range("2011-08-18", "2026-09-10", freq="D", tz="UTC")
    tramos = tesis.tramos_halving(idx)
    assert [n for n, _, _ in tramos] == ["2011–2012", "2012–2016", "2016–2020", "2020–2024", "2024–2026"]
    assert tramos[0][1] == idx[0] and tramos[-1][2] == idx[-1] + pd.Timedelta(days=1)
    assert all(fin == ini for (_, _, fin), (_, ini, _) in zip(tramos, tramos[1:]))


def test_cargar_historia_arrastra_el_cierre_sin_inventar_volumen(tmp_path, monkeypatch):
    # Un dia sin negociacion en Bitstamp: precio del dia anterior y volumen cero.
    monkeypatch.setattr(tesis, "CARPETA_HISTORIA", tmp_path)
    idx = pd.to_datetime(["2011-08-18", "2011-08-19", "2011-08-21"], utc=True)
    velas = pd.DataFrame({c: [10.0, 11.0, 12.0] for c in ("open", "high", "low", "close")}
                         | {"volume": [1.0, 2.0, 3.0]}, index=idx)
    velas.to_parquet(tesis.ruta_historia("bitstamp", "BTC/USD"))
    velas.to_parquet(tesis.ruta_historia("binance", "BTC/USDT"))
    btc, binance = tesis.cargar_historia(["BTC"], "2011-08-21")
    hueco = btc.loc[pd.Timestamp("2011-08-20", tz="UTC")]
    assert len(btc) == 4 and hueco["close"] == 11.0 and hueco["open"] == 11.0 and hueco["volume"] == 0.0
    assert list(binance) == ["BTC"]
    # Sin la historia de un activo no se analiza nada a medias.
    assert tesis.cargar_historia(["ETH"], "2011-08-21") is None


# --------------------------------------------------------------------------
# Sensibilidad a los dos errores de implementacion (el modelo no cambia)
# --------------------------------------------------------------------------
def _panel_sintetico(n_bars=900, n_assets=3):
    from cryptoquant.config import Config
    from cryptoquant.data.sources import _synthetic

    ohlcv, closes = {}, {}
    for i in range(n_assets):
        df = _synthetic(f"S{i}/USDT", "1d", "2020-01-01", n_bars, seed=300 + i)
        ohlcv[f"S{i}"], closes[f"S{i}"] = df, df["close"]
    cfg = Config()
    bt = tesis.WalkForwardBacktest(cfg)
    closes = pd.DataFrame(closes)
    feats, labels, _ = bt.prepare(ohlcv, closes)
    return cfg, bt, feats, labels, closes.index


def test_modelo_corregido_sin_correcciones_replica_el_original():
    """Si la replica no fuera exacta, la tabla de sensibilidad mediria otra cosa."""
    cfg, bt, feats, labels, idx = _panel_sintetico()
    copia = tesis.ModeloCorregido(cfg, pesos_relativos=False, calibracion_temporal=False)
    for upto in (400, len(idx) - 1):                  # ventana fija y ventana deslizada
        a, b = bt._pooled_training_set(feats, labels, upto, idx), copia._pooled_training_set(feats, labels, upto, idx)
        for x, z in zip(a, b):
            pd.testing.assert_frame_equal(pd.DataFrame(x), pd.DataFrame(z))


def test_los_pesos_solo_fallan_cuando_la_ventana_se_desliza():
    cfg, bt, feats, labels, idx = _panel_sintetico()
    corregido = tesis.ModeloCorregido(cfg, pesos_relativos=True)
    _, _, w_fija = bt._pooled_training_set(feats, labels, 400, idx)          # empieza en la barra 0
    _, _, w_fija_c = corregido._pooled_training_set(feats, labels, 400, idx)
    np.testing.assert_allclose(w_fija, w_fija_c)
    _, _, w = bt._pooled_training_set(feats, labels, len(idx) - 1, idx)       # ventana deslizada
    _, _, w_c = corregido._pooled_training_set(feats, labels, len(idx) - 1, idx)
    assert np.corrcoef(w, w_c)[0, 1] < 0.9


def test_la_calibracion_temporal_reserva_las_fechas_mas_recientes():
    cfg, _, feats, labels, idx = _panel_sintetico()
    apilado, _, _ = tesis.ModeloCorregido(cfg, calibracion_temporal=False).panel(feats, labels, len(idx) - 1, idx)
    ordenado, _, _ = tesis.ModeloCorregido(cfg, calibracion_temporal=True).panel(feats, labels, len(idx) - 1, idx)
    corte = int(len(apilado) * 0.8)
    # Apilado por activo, el 20 % reservado son fechas que el ajuste tambien ve en otros activos.
    assert apilado.index[corte:].min() < apilado.index[:corte].max()
    assert ordenado.index.is_monotonic_increasing
    assert ordenado.index[corte:].min() >= ordenado.index[:corte].max()


def test_comprar_y_mantener_no_rebalancea():
    """Uno se duplica y otro no se mueve: sin rebalanceo se acaba en 1,5."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    closes = pd.DataFrame({"A": [1.0, 1.5, 2.0], "B": [1.0, 1.0, 1.0]}, index=idx)
    r = tesis.comprar_y_mantener(closes, idx[0])
    assert float((1 + r).prod()) == pytest.approx(1.5)
    # La pasiva de la tesis (media diaria de rendimientos) acaba en otro sitio.
    pasiva = closes.pct_change().dropna().mean(axis=1)
    assert float((1 + pasiva).prod()) != pytest.approx(1.5)
