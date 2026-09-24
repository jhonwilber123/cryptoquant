"""Tests del nucleo del sistema.

El test mas importante de todos es `test_features_are_causal`: si una feature
usa informacion futura, todas las metricas del backtest son ficcion y el resto
de tests da igual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptoquant.backtest.engine import WalkForwardBacktest
from cryptoquant.config import Config
from cryptoquant.data.sources import _synthetic
from cryptoquant.econometrics.risk import (
    cvar_historical,
    max_drawdown,
    var_cornish_fisher,
    var_historical,
)
from cryptoquant.econometrics.volatility import ewma_forecast, ledoit_wolf_cov
from cryptoquant.features.technical import build_features, hurst_exponent
from cryptoquant.models.labeling import triple_barrier_labels
from cryptoquant.models.validation import purged_walk_forward
from cryptoquant.portfolio.optimizer import apply_constraints, build_weights
from cryptoquant.portfolio.sizing import (
    drawdown_scalar,
    kelly_fraction,
    volatility_scalar,
)


def _copy_cfg(cfg: Config) -> Config:
    import copy

    return copy.deepcopy(cfg)


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def ohlcv():
    return _synthetic("TEST/USDT", "1d", "2020-01-01", 900, seed=7)


# --------------------------------------------------------------------------
# Causalidad: el test que sostiene la validez de todo lo demas
# --------------------------------------------------------------------------
def test_features_are_causal(ohlcv, cfg):
    """Truncar el futuro no puede cambiar las features del pasado.

    Se calculan las features sobre la serie completa y sobre la serie cortada
    en la barra k. Los valores hasta k deben coincidir exactamente. Si difieren,
    alguna feature esta mirando hacia delante.
    """
    full = build_features(ohlcv, 365.0, cfg.signal)
    k = 700
    truncated = build_features(ohlcv.iloc[:k], 365.0, cfg.signal)

    # `hurst` se calcula cada 5 barras y se propaga, asi que su ultimo valor
    # depende de donde cae el corte; se compara con tolerancia de rejilla.
    cols = [c for c in full.columns if c != "hurst"]
    a = full[cols].iloc[:k]
    b = truncated[cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-10)


def test_no_future_leak_in_regime(ohlcv, cfg):
    feats = build_features(ohlcv, 365.0, cfg.signal)
    ma = ohlcv["close"].rolling(cfg.signal.regime_ma,
                               min_periods=cfg.signal.regime_ma // 2).mean()
    expected = (ohlcv["close"] > ma).astype(float)
    pd.testing.assert_series_equal(feats["regime_up"], expected, check_names=False)


# --------------------------------------------------------------------------
# Etiquetado
# --------------------------------------------------------------------------
def test_triple_barrier_hits_upper():
    """Precio que sube en linea recta debe etiquetarse como beneficio."""
    close = pd.Series(np.linspace(100, 130, 60), index=pd.date_range("2021-01-01", periods=60))
    sigma = pd.Series(0.02, index=close.index)
    lab = triple_barrier_labels(close, sigma, horizon=10, profit_mult=2.0, stop_mult=1.5)
    assert lab["label"].iloc[0] == 1.0
    assert lab["ret"].iloc[0] > 0


def test_triple_barrier_hits_lower():
    close = pd.Series(np.linspace(100, 70, 60), index=pd.date_range("2021-01-01", periods=60))
    sigma = pd.Series(0.02, index=close.index)
    lab = triple_barrier_labels(close, sigma, horizon=10, profit_mult=2.0, stop_mult=1.5)
    assert lab["label"].iloc[0] == -1.0
    assert lab["ret"].iloc[0] < 0


def test_triple_barrier_exit_within_horizon():
    """La salida nunca puede caer mas alla de la barrera vertical."""
    px = _synthetic("X/USDT", "1d", "2020-01-01", 300, seed=3)["close"]
    sigma = np.log(px).diff().rolling(20, min_periods=10).std()
    lab = triple_barrier_labels(px, sigma, horizon=10)
    valid = lab["exit_bar"].dropna()
    starts = np.arange(len(lab))[lab["exit_bar"].notna().to_numpy()]
    assert (valid.to_numpy() - starts <= 10).all()
    assert (valid.to_numpy() > starts).all()


# --------------------------------------------------------------------------
# Validacion temporal
# --------------------------------------------------------------------------
def test_purged_splits_do_not_overlap():
    idx = pd.date_range("2019-01-01", periods=1200)
    horizon = 10
    for sp in purged_walk_forward(idx, n_splits=4, horizon=horizon, embargo=0.01):
        assert sp.train_idx.max() < sp.test_idx.min()
        # La purga debe dejar al menos `horizon` barras de separacion.
        assert sp.test_idx.min() - sp.train_idx.max() > horizon
        assert len(np.intersect1d(sp.train_idx, sp.test_idx)) == 0


def test_purged_splits_are_chronological():
    idx = pd.date_range("2019-01-01", periods=1200)
    splits = list(purged_walk_forward(idx, n_splits=4, horizon=10))
    assert len(splits) >= 3
    for a, b in zip(splits, splits[1:]):
        assert b.test_start > a.test_start


# --------------------------------------------------------------------------
# Riesgo
# --------------------------------------------------------------------------
def test_cvar_is_worse_than_var():
    """El CVaR mide la cola mas alla del VaR, asi que nunca es menor."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_t(df=4, size=3000) * 0.02)
    assert cvar_historical(r, 0.95) >= var_historical(r, 0.95)


def test_cornish_fisher_detects_fat_left_tail():
    """Con asimetria negativa moderada (dentro del dominio de validez de la
    expansion), el VaR de Cornish-Fisher debe superar al gaussiano: esa
    diferencia es exactamente el riesgo que el supuesto normal ignora."""
    from cryptoquant.econometrics.risk import _cf_is_monotonic, var_parametric
    from scipy import stats as _st

    rng = np.random.default_rng(1)
    n = 6000
    base = rng.normal(0, 0.01, n)
    crashes = rng.choice([0, 1], size=n, p=[0.95, 0.05]) * rng.normal(-0.03, 0.006, n)
    r = pd.Series(base + crashes)  # skew ~ -0.75, exceso de curtosis ~ 1.8

    z = _st.norm.ppf(0.01)
    assert _cf_is_monotonic(z, float(_st.skew(r)), float(_st.kurtosis(r))), \
        "el caso de prueba debe caer DENTRO del dominio de C-F"
    assert var_cornish_fisher(r, 0.99) > var_parametric(r, 0.99)
    # Y debe acercarse al historico mucho mas que el gaussiano.
    hist = var_historical(r, 0.99)
    assert abs(var_cornish_fisher(r, 0.99) - hist) < abs(var_parametric(r, 0.99) - hist)


def test_cornish_fisher_never_returns_impossible_var():
    """Regresion: con curtosis extrema la expansion producia un VaR negativo
    ("perder es imposible"). Debe detectarse el dominio y usar el historico.

    Los parametros reproducen el perfil de momentos de BNB en el historico
    real, que es donde aparecio el fallo.
    """
    rng = np.random.default_rng(12)
    base = rng.normal(0.001, 0.02, 2000)
    jumps = rng.choice([0, 1], size=2000, p=[0.997, 0.003]) * rng.normal(0.45, 0.10, 2000)
    r = pd.Series(base + jumps)  # asimetria positiva y curtosis enormes

    var = var_cornish_fisher(r, 0.95)
    assert var > 0, "un VaR no positivo con retornos negativos presentes es imposible"
    assert var == pytest.approx(var_historical(r, 0.95)), "deberia haber usado el respaldo"
    # Sin respaldo debe declararlo, no inventar un numero.
    assert np.isnan(var_cornish_fisher(r, 0.95, fallback=False))


def test_cornish_fisher_used_when_moments_are_mild():
    """Con momentos moderados si debe aplicarse la formula, no el respaldo."""
    rng = np.random.default_rng(13)
    r = pd.Series(rng.normal(0, 0.01, 3000))
    assert var_cornish_fisher(r, 0.95) != pytest.approx(var_historical(r, 0.95), abs=1e-12)


def test_max_drawdown_sign_and_magnitude():
    eq = pd.Series([100, 120, 60, 90, 130])
    assert max_drawdown(eq) == pytest.approx(-0.5)


# --------------------------------------------------------------------------
# Dimensionamiento
# --------------------------------------------------------------------------
def test_kelly_zero_without_edge():
    """Sin ventaja (p=0.5, payoff 1:1) Kelly debe ser cero: no apostar."""
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)
    assert kelly_fraction(0.4, 1.0) == 0.0  # ventaja negativa -> nunca negativo
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)


def test_volatility_scalar_reduces_when_market_is_wild():
    # Vol prevista 60%, objetivo 15% -> invertir el 25%.
    assert volatility_scalar(0.60, 0.15) == pytest.approx(0.25)
    # Vol prevista baja -> se topa en el cap, nunca crece sin limite.
    assert volatility_scalar(0.01, 0.15, cap=3.0) == 3.0
    assert volatility_scalar(0.0, 0.15) == 0.0


def test_conviction_normalises_kelly_to_unit_interval():
    from cryptoquant.portfolio.sizing import conviction_from_kelly

    # Al alcanzar el nivel de tamano pleno, conviccion = 1 (no sigue creciendo).
    assert conviction_from_kelly(0.25, 0.25) == pytest.approx(1.0)
    assert conviction_from_kelly(0.80, 0.25) == pytest.approx(1.0)
    # Ventaja a medias -> medio tamano.
    assert conviction_from_kelly(0.125, 0.25) == pytest.approx(0.5)
    assert conviction_from_kelly(0.0, 0.25) == 0.0


def test_exposure_is_not_double_discounted(cfg):
    """Regresion del fallo principal: Kelly y el objetivo de volatilidad se
    multiplicaban entre si, y ambos recortan por el mismo riesgo. El resultado
    era un 1% de exposicion y una vol realizada 35 veces menor que la pedida.

    Con una ventaja decente y la vol del mercado al cuadruple del objetivo, la
    exposicion debe rondar el escalar de volatilidad (25%), no una fraccion
    minuscula de el.
    """
    from cryptoquant.portfolio.sizing import decide_exposure

    cfg.risk.target_volatility = 0.15
    d = decide_exposure(prob_win=0.60, payoff_ratio=2.0 / 1.5, forecast_vol=0.60,
                        current_drawdown=0.0, regime_ok=True, risk_cfg=cfg.risk)
    assert d.vol_scalar == pytest.approx(0.25)
    assert d.gross_exposure > 0.15, f"exposicion colapsada: {d.gross_exposure:.4f}"
    assert d.gross_exposure <= 0.25 + 1e-9


def test_realised_volatility_tracks_the_target(cfg):
    """La promesa central del sistema: tu fijas el riesgo.

    Se comprueba que subir el objetivo de volatilidad suba de verdad la
    volatilidad realizada, y que el nivel guarde una relacion sensata con lo
    pedido en vez de quedarse en una fraccion despreciable.
    """
    cfg.backtest.warmup = 260
    ohlcv, closes = _small_universe(n_assets=4, n_bars=900)

    realised = {}
    for target in (0.10, 0.30):
        c = _copy_cfg(cfg)
        c.risk.target_volatility = target
        res = WalkForwardBacktest(c, "hrp", use_ml=False).run(ohlcv, closes)
        realised[target] = float(res.returns.std(ddof=1) * np.sqrt(365))

    assert realised[0.30] > realised[0.10], "el objetivo no controla el riesgo"
    # No se exige clavarlo (la vol prevista tiene error), pero si estar en el
    # mismo orden de magnitud: nada de un 0.4% cuando se pide un 30%.
    assert realised[0.30] > 0.30 * 0.25, f"vol realizada colapsada: {realised[0.30]:.3%}"


def test_drawdown_scalar_shuts_down_at_stop():
    assert drawdown_scalar(-0.05, 0.10, 0.25) == 1.0
    assert drawdown_scalar(-0.175, 0.10, 0.25) == pytest.approx(0.5)
    assert drawdown_scalar(-0.30, 0.10, 0.25) == 0.0


# --------------------------------------------------------------------------
# Cartera
# --------------------------------------------------------------------------
def test_weights_sum_to_one_and_respect_cap():
    rng = np.random.default_rng(5)
    data = rng.normal(0, 0.02, (400, 6))
    data[:, 0] *= 0.2  # un activo mucho menos volatil que el resto
    cov = np.cov(data, rowvar=False)
    labels = [f"A{i}" for i in range(6)]
    for method in ("hrp", "risk_parity", "min_variance", "equal"):
        w = build_weights(cov, labels, method, max_weight=0.30)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all()
        assert w.max() <= 0.30 + 1e-6, f"{method} incumple el tope por activo"


def test_hrp_favours_the_calm_asset():
    """HRP debe asignar mas peso al activo de menor varianza."""
    rng = np.random.default_rng(11)
    data = rng.normal(0, 0.02, (600, 4))
    data[:, 0] *= 0.15
    cov = np.cov(data, rowvar=False)
    w = build_weights(cov, ["calmo", "b", "c", "d"], "hrp", max_weight=0.9)
    assert w["calmo"] == w.max()


def test_apply_constraints_redistributes():
    """Con el tope factible (5 activos x 0.30 = 1.5 > 1) debe cumplirse."""
    w = pd.Series({"a": 0.60, "b": 0.20, "c": 0.10, "d": 0.06, "e": 0.04})
    out = apply_constraints(w, 0.30)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.30 + 1e-9
    # El orden relativo se conserva entre los que no tocan el tope.
    assert out["b"] > out["c"] > out["d"] > out["e"]


def test_apply_constraints_infeasible_falls_back_to_equal():
    """3 activos con tope 0.30 no pueden sumar 1: se reparte por igual y el
    limite real lo impone despues el tope sobre el peso absoluto."""
    w = pd.Series({"a": 0.8, "b": 0.15, "c": 0.05})
    out = apply_constraints(w, 0.30)
    assert out.sum() == pytest.approx(1.0)
    assert out.nunique() == 1


def test_absolute_concentration_cap_is_enforced(cfg):
    """El tope de `max_weight_per_asset` debe cumplirse sobre el capital total,
    incluso con muy pocos activos elegibles."""
    cfg.backtest.warmup = 260
    cfg.risk.max_weight_per_asset = 0.20
    ohlcv, closes = _small_universe(n_assets=3)
    res = WalkForwardBacktest(cfg, "hrp", use_ml=False).run(ohlcv, closes)
    assert res.diagnostics["gross_target"].max() <= 3 * 0.20 + 1e-6


def test_ledoit_wolf_is_positive_semidefinite():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(0, 0.02, (100, 12)))  # n < p*10: caso ruidoso
    cov = ledoit_wolf_cov(df)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-12)


# --------------------------------------------------------------------------
# Otros
# --------------------------------------------------------------------------
def _integrated_ar1(phi: float, n: int, seed: int) -> pd.Series:
    """Serie cuyos incrementos siguen un AR(1). phi>0 da memoria persistente,
    phi<0 antipersistente. Es el proceso que el exponente de Hurst mide, a
    diferencia de una deriva determinista."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return pd.Series(np.cumsum(x))


def test_hurst_separates_persistence_from_mean_reversion():
    persistent = _integrated_ar1(0.7, 3000, 4)
    random_walk = pd.Series(np.cumsum(np.random.default_rng(4).normal(0, 0.01, 3000)))
    antipersistent = _integrated_ar1(-0.7, 3000, 4)

    h_p = hurst_exponent(persistent)
    h_r = hurst_exponent(random_walk)
    h_a = hurst_exponent(antipersistent)

    assert h_p > h_r > h_a, f"orden incorrecto: {h_p:.3f} {h_r:.3f} {h_a:.3f}"
    assert 0.35 < h_r < 0.65, "el paseo aleatorio debe salir cerca de 0.5"
    assert h_a < 0.5


def test_hurst_ignores_deterministic_drift():
    """Una recta no es memoria larga: sus diferencias a lag fijo son
    constantes, asi que H se queda en el nivel del ruido subyacente."""
    rng = np.random.default_rng(9)
    drifted = pd.Series(np.cumsum(np.full(2000, 0.01) + rng.normal(0, 0.002, 2000)))
    assert 0.35 < hurst_exponent(drifted) < 0.65


def test_ewma_forecast_positive():
    r = pd.Series(np.random.default_rng(6).normal(0, 0.02, 500))
    fc = ewma_forecast(r, 365.0)
    assert fc.converged and 0 < fc.sigma_ann < 5


# --------------------------------------------------------------------------
# Integracion
# --------------------------------------------------------------------------
def _small_universe(n_assets=4, n_bars=700):
    ohlcv, closes = {}, {}
    for i in range(n_assets):
        sym = f"A{i}"
        df = _synthetic(f"{sym}/USDT", "1d", "2020-01-01", n_bars, seed=100 + i)
        ohlcv[sym] = df
        closes[sym] = df["close"]
    return ohlcv, pd.DataFrame(closes)


def test_backtest_runs_end_to_end(cfg):
    cfg.backtest.warmup = 260
    cfg.backtest.retrain_every = 120
    ohlcv, closes = _small_universe()
    res = WalkForwardBacktest(cfg, "hrp", use_ml=True).run(ohlcv, closes)

    assert len(res.equity) > 100
    assert res.equity.notna().all()
    assert (res.equity > 0).all(), "el capital nunca puede volverse negativo"
    assert len(res.returns) == len(res.equity) - 1
    # La equity debe reconstruirse exactamente desde los retornos.
    rebuilt = cfg.backtest.initial_capital * (1 + res.returns).cumprod()
    np.testing.assert_allclose(rebuilt.to_numpy(), res.equity.iloc[1:].to_numpy(), rtol=1e-9)


def test_backtest_never_exceeds_max_exposure(cfg):
    cfg.backtest.warmup = 260
    cfg.risk.max_gross_exposure = 0.60
    ohlcv, closes = _small_universe()
    res = WalkForwardBacktest(cfg, "hrp", use_ml=False).run(ohlcv, closes)
    # Se permite un margen por la deriva de precios entre rebalanceos.
    assert res.diagnostics["gross_target"].max() <= 0.60 + 1e-6


def test_zero_target_volatility_means_no_position(cfg):
    """Con objetivo de vol ~0 el sistema debe quedarse en efectivo."""
    cfg.backtest.warmup = 260
    cfg.risk.target_volatility = 1e-6
    ohlcv, closes = _small_universe()
    res = WalkForwardBacktest(cfg, "hrp", use_ml=False).run(ohlcv, closes)
    assert res.diagnostics["gross_target"].max() < 0.01


def test_costs_reduce_returns(cfg):
    """Subir las comisiones no puede mejorar el resultado."""
    cfg.backtest.warmup = 260
    ohlcv, closes = _small_universe()
    cheap = WalkForwardBacktest(cfg, "hrp", use_ml=False).run(ohlcv, closes)

    import copy

    expensive_cfg = copy.deepcopy(cfg)
    expensive_cfg.backtest.fee = 0.01
    expensive_cfg.backtest.slippage = 0.005
    expensive = WalkForwardBacktest(expensive_cfg, "hrp", use_ml=False).run(ohlcv, closes)

    assert expensive.equity.iloc[-1] <= cheap.equity.iloc[-1]


# --------------------------------------------------------------------------
# Significancia estadistica
# --------------------------------------------------------------------------
def test_sharpe_ratios_respect_the_frequency_scale():
    """Regresion: se pasaba el Sharpe ANUALIZADO a una formula que exige el
    Sharpe por periodo. La diferencia es un factor sqrt(365) y hundia el
    resultado a 0% para cualquier estrategia, por buena que fuese.
    """
    from cryptoquant.models.validation import (
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
    )

    ann, n = 365.0, 1927
    dsr = deflated_sharpe_ratio(0.74, n, 12, skew=0.11, kurtosis=18.18,
                                periods_per_year=ann)
    psr = probabilistic_sharpe_ratio(0.74, n, 0.0, skew=0.11, kurtosis=18.18,
                                     periods_per_year=ann)
    # Un Sharpe de 0.74 sobre 1927 observaciones no es ni 0% ni 100% de certeza.
    assert 0.3 < dsr < 0.9, f"DSR fuera de rango plausible: {dsr}"
    assert 0.85 < psr < 0.99, f"PSR fuera de rango plausible: {psr}"
    # Pasar el mismo Sharpe ya en escala por periodo debe dar lo mismo.
    assert deflated_sharpe_ratio(0.74 / np.sqrt(ann), n, 12, 0.11, 18.18, 1.0) \
        == pytest.approx(dsr)


def test_deflation_penalises_more_trials():
    """Cuantas mas configuraciones se prueban, mas exigente debe ser el umbral."""
    from cryptoquant.models.validation import deflated_sharpe_ratio

    few = deflated_sharpe_ratio(0.9, 2000, 3, periods_per_year=365.0)
    many = deflated_sharpe_ratio(0.9, 2000, 500, periods_per_year=365.0)
    assert few > many


def test_psr_increases_with_sharpe():
    from cryptoquant.models.validation import probabilistic_sharpe_ratio

    low = probabilistic_sharpe_ratio(0.2, 2000, periods_per_year=365.0)
    high = probabilistic_sharpe_ratio(1.5, 2000, periods_per_year=365.0)
    assert high > low


# --------------------------------------------------------------------------
# Barras sin cerrar
# --------------------------------------------------------------------------
def test_incomplete_bar_is_dropped():
    """Regresion: el exchange devuelve la barra en curso, cuyo 'cierre' es el
    precio de este instante. Registrar una decision sobre ella y evaluarla
    despues contra el cierre definitivo mide el reloj, no la estrategia.
    """
    from cryptoquant.data.sources import drop_incomplete_bars

    now = pd.Timestamp("2026-09-09 20:40:00", tz="UTC")
    idx = pd.date_range("2026-09-06", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=idx)

    out = drop_incomplete_bars(df, "1d", now=now)
    # La barra del 09 cierra el 10 a las 00:00: aun esta abierta.
    assert out.index[-1] == pd.Timestamp("2026-09-08", tz="UTC")
    assert len(out) == 3


def test_bar_needs_a_grace_period_after_closing():
    """Justo en el instante del cierre la barra NO se acepta todavia: el
    exchange tarda unos segundos en consolidarla, y una tarea automatica puede
    dispararse exactamente en ese momento."""
    from cryptoquant.data.sources import drop_incomplete_bars

    idx = pd.date_range("2026-09-06", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    close_instant = pd.Timestamp("2026-09-09", tz="UTC")

    assert len(drop_incomplete_bars(df, "1d", now=close_instant)) == 2
    assert len(drop_incomplete_bars(
        df, "1d", now=close_instant + pd.Timedelta(minutes=5))) == 3
    # Con margen cero vuelve al criterio estricto de cierre.
    assert len(drop_incomplete_bars(
        df, "1d", now=close_instant, grace_seconds=0)) == 3


def test_hourly_bars_use_their_own_period():
    from cryptoquant.data.sources import drop_incomplete_bars

    idx = pd.date_range("2026-09-09 10:00", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    out = drop_incomplete_bars(df, "1h", now=pd.Timestamp("2026-09-09 12:30", tz="UTC"))
    assert len(out) == 2  # la de las 12:00 cierra a las 13:00


def test_empty_frame_is_handled():
    from cryptoquant.data.sources import drop_incomplete_bars

    assert drop_incomplete_bars(pd.DataFrame(), "1d").empty


def test_cache_is_detected_as_stale_when_a_bar_has_closed():
    """Regresion critica para la automatizacion: sin esta deteccion la cache
    se sirve para siempre, la tarea diaria recalcula la misma fecha una y otra
    vez, el diario la rechaza por duplicada y el forward test no avanza nunca
    -- todo ello sin un solo mensaje de error."""
    from cryptoquant.data.sources import cache_is_stale

    idx = pd.date_range("2026-09-06", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)  # ultima: 09-08

    # La barra del 09-09 cierra el 09-10 a las 00:00. Antes de eso no falta nada.
    assert not cache_is_stale(df, "1d", now=pd.Timestamp("2026-09-09 20:00", tz="UTC"))
    # Ya cerrada (mas el margen), la cache esta incompleta.
    assert cache_is_stale(df, "1d", now=pd.Timestamp("2026-09-10 00:05", tz="UTC"))
    # Varios dias despues, obviamente.
    assert cache_is_stale(df, "1d", now=pd.Timestamp("2026-09-15", tz="UTC"))


def test_empty_cache_counts_as_stale():
    from cryptoquant.data.sources import cache_is_stale

    assert cache_is_stale(pd.DataFrame(), "1d")
