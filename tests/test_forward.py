"""Tests del forward test.

Lo que se verifica aqui no es que el sistema gane dinero, sino que el
experimento sea honesto: que el pasado no se pueda retocar, que la aritmetica
del resultado realizado sea correcta, y que el analisis de potencia diga la
verdad sobre lo que el experimento puede y no puede concluir.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cryptoquant.backtest.engine import WalkForwardBacktest, effective_retrain_every
from cryptoquant.config import Config
from cryptoquant.data.sources import _synthetic
from cryptoquant.forward import journal as jn
from cryptoquant.forward.evaluate import (
    annualised_volatility,
    block_bootstrap_ci,
    newey_west_tstat,
    observations_required,
    portfolio_state_at,
    realised_returns,
    volatility_precision,
)


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    j = tmp_path / "journal.jsonl"
    p = tmp_path / "preregistration.json"
    monkeypatch.setattr(jn, "JOURNAL_PATH", j)
    monkeypatch.setattr(jn, "PREREG_PATH", p)
    monkeypatch.setattr(jn, "AMENDMENTS_PATH", tmp_path / "amendments.json")
    return j, p


def _add(date, prices, weights, cfg, **kw):
    return jn.append_record(
        decision_date=date, prices=prices, weights=weights,
        gross_exposure=sum(weights.values()), diagnostics={"reason": "test"},
        cfg=cfg, code_version="test", **kw,
    )


# --------------------------------------------------------------------------
# Integridad: lo que hace que el experimento valga algo
# --------------------------------------------------------------------------
def test_chain_is_intact_when_untouched(paths, cfg):
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    _add("2026-01-02", {"BTC": 110.0}, {"BTC": 0.4}, cfg)
    _add("2026-01-03", {"BTC": 105.0}, {"BTC": 0.2}, cfg)

    status = jn.verify_chain()
    assert status.ok
    assert status.n_records == 3
    assert status.problems == []


def test_tampering_with_a_past_record_is_detected(paths, cfg):
    """El escenario que este diseno existe para impedir: alguien mejora una
    decision pasada despues de ver como salio."""
    j, _ = paths
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    _add("2026-01-02", {"BTC": 110.0}, {"BTC": 0.4}, cfg)
    _add("2026-01-03", {"BTC": 105.0}, {"BTC": 0.2}, cfg)
    assert jn.verify_chain().ok

    lines = j.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["weights"]["BTC"] = 0.95           # "yo habia apostado fuerte, en realidad"
    rec["gross_exposure"] = 0.95
    lines[0] = json.dumps(rec, ensure_ascii=False)
    j.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = jn.verify_chain()
    assert not status.ok
    assert status.first_break == 1
    assert any("modificado" in p for p in status.problems)


def test_tampering_that_recomputes_the_hash_still_breaks_the_chain(paths, cfg):
    """Un manipulador algo mas listo recalcula el hash del registro que edita.
    La cadena sigue rompiendose, porque el registro SIGUIENTE apunta al hash
    antiguo."""
    j, _ = paths
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    _add("2026-01-02", {"BTC": 110.0}, {"BTC": 0.4}, cfg)

    lines = j.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["weights"]["BTC"] = 0.95
    rec["hash"] = jn._digest(rec)          # rehace su propio hash
    lines[0] = json.dumps(rec, ensure_ascii=False)
    j.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = jn.verify_chain()
    assert not status.ok
    assert any("no enlaza" in p for p in status.problems)


def test_cannot_record_the_same_date_twice(paths, cfg):
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    with pytest.raises(ValueError, match="ya hay una anotacion"):
        _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.9}, cfg)


def test_preregistration_cannot_be_silently_replaced(paths, cfg):
    jn.write_preregistration(cfg, ["BTC", "ETH"], "test")
    with pytest.raises(FileExistsError):
        jn.write_preregistration(cfg, ["BTC", "ETH"], "test")
    assert jn.preregistration_is_intact()


def test_edited_preregistration_is_detected(paths, cfg):
    _, p = paths
    jn.write_preregistration(cfg, ["BTC"], "test")
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["hypotheses"][0]["success"] = "lo que sea que haya salido"
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert not jn.preregistration_is_intact()


def test_records_carry_the_inputs_needed_to_reproduce(paths, cfg):
    """Sin el estado de entrada, recomputar da otro resultado y H2 fallaria
    por un motivo falso."""
    rec = _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg,
               state_in={"prev_weights": {"BTC": 0.1}, "drawdown": -0.05})
    assert rec["state_in"]["prev_weights"] == {"BTC": 0.1}
    assert rec["state_in"]["drawdown"] == -0.05


# --------------------------------------------------------------------------
# Aritmetica del resultado realizado
# --------------------------------------------------------------------------
def test_realised_return_matches_hand_calculation(paths, cfg):
    """Un tramo con numeros redondos, calculado a mano."""
    _add("2026-01-01", {"BTC": 100.0, "ETH": 50.0}, {"BTC": 0.5, "ETH": 0.5}, cfg)
    _add("2026-01-02", {"BTC": 110.0, "ETH": 45.0}, {"BTC": 0.5, "ETH": 0.5}, cfg)

    res = realised_returns(jn.read_journal(), pd.DataFrame(), fee=0.0, slippage=0.0)
    # BTC +10%, ETH -10%, al 50/50 -> 0%. Rotacion 1.0 en el primer tramo pero
    # con coste cero no resta nada.
    assert float(res.returns.iloc[0]) == pytest.approx(0.0, abs=1e-12)
    # La referencia equiponderada sobre los mismos activos es identica aqui.
    assert float(res.benchmark.iloc[0]) == pytest.approx(0.0, abs=1e-12)


def test_costs_are_charged_on_turnover(paths, cfg):
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.5}, cfg)
    _add("2026-01-02", {"BTC": 100.0}, {"BTC": 0.5}, cfg)

    free = realised_returns(jn.read_journal(), pd.DataFrame(), 0.0, 0.0)
    paid = realised_returns(jn.read_journal(), pd.DataFrame(), 0.001, 0.0005)
    # Precio plano: la unica diferencia posible es el coste de montar la posicion.
    assert float(paid.returns.iloc[0]) < float(free.returns.iloc[0])
    assert float(paid.costs.iloc[0]) == pytest.approx(0.5 * 0.0015)


def test_missing_price_is_reported_not_silently_ignored(paths, cfg):
    _add("2026-01-01", {"BTC": 100.0, "ETH": 50.0}, {"BTC": 0.3, "ETH": 0.3}, cfg)
    _add("2026-01-02", {"BTC": 110.0}, {"BTC": 0.3, "ETH": 0.3}, cfg)  # falta ETH
    res = realised_returns(jn.read_journal(), pd.DataFrame(), 0.0, 0.0)
    assert any("ETH" in n for n in res.notes)


def test_single_record_yields_no_measurable_span(paths, cfg):
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    res = realised_returns(jn.read_journal(), pd.DataFrame(), 0.0, 0.0)
    assert res.n_gaps == 0
    assert res.notes


# --------------------------------------------------------------------------
# Honestidad estadistica
# --------------------------------------------------------------------------
def test_power_analysis_reports_decades_for_a_weak_edge():
    """El numero que define si el experimento tiene sentido.

    Ventaja de 2.5% anual con tracking error de 8.8%: information ratio 0.29,
    que exige decadas. Si esta funcion devolviera algo optimista, el forward
    test prometeria lo que no puede dar.
    """
    req = observations_required(0.0251, 0.0875, periods_per_year=365.0)
    assert req["information_ratio"] == pytest.approx(0.287, abs=0.01)
    assert 40 < req["years_significance"] < 55
    assert 85 < req["years_powered"] < 105


def test_power_analysis_rewards_a_strong_edge():
    weak = observations_required(0.02, 0.10)
    strong = observations_required(0.20, 0.10)
    assert strong["years_significance"] < weak["years_significance"]
    assert strong["years_significance"] < 2


def test_no_edge_means_nothing_to_prove():
    req = observations_required(-0.01, 0.10)
    assert not np.isfinite(req["years_significance"])


def test_volatility_converges_faster_than_the_mean():
    """La razon por la que H1 es contrastable en meses y H3 no lo es nunca."""
    assert volatility_precision(180) < 0.06
    assert volatility_precision(365) < 0.04
    assert volatility_precision(30) > volatility_precision(365)


def test_newey_west_is_more_conservative_under_autocorrelation():
    """Con datos autocorrelacionados el t simple exagera la significancia."""
    rng = np.random.default_rng(3)
    n = 1500
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + e[i]      # fuerte dependencia serial
    x = x + 0.0008

    t_nw, p_nw, lag = newey_west_tstat(x)
    t_naive = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    assert lag > 0
    assert abs(t_nw) < abs(t_naive), "Newey-West debe corregir a la baja"


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(5)
    x = rng.normal(0.001, 0.01, 2000)
    lo, hi, pos = block_bootstrap_ci(x, n_boot=2000)
    assert lo < x.mean() < hi
    assert 0.0 <= pos <= 1.0


# --------------------------------------------------------------------------
# Equivalencia entre el modo en vivo y el backtest
# --------------------------------------------------------------------------
def _universe(n_assets=4, n_bars=800):
    ohlcv, closes = {}, {}
    for i in range(n_assets):
        sym = f"A{i}"
        df = _synthetic(f"{sym}/USDT", "1d", "2020-01-01", n_bars, seed=200 + i)
        ohlcv[sym] = df
        closes[sym] = df["close"]
    return ohlcv, pd.DataFrame(closes)


def test_decide_at_is_deterministic(cfg):
    """Dos llamadas con el mismo estado deben dar exactamente lo mismo. Sin
    esto, H2 no significa nada."""
    cfg.backtest.warmup = 260
    ohlcv, closes = _universe()
    bt = WalkForwardBacktest(cfg, "hrp", use_ml=True)
    w1, _ = bt.decide_at(ohlcv, closes)
    w2, _ = bt.decide_at(ohlcv, closes)
    pd.testing.assert_series_equal(w1, w2)


def test_decide_at_is_causal(cfg):
    """La decision para una fecha pasada no puede cambiar porque hoy haya mas
    datos disponibles. Es la version en produccion del test de causalidad."""
    cfg.backtest.warmup = 260
    ohlcv, closes = _universe(n_bars=800)
    target = closes.index[600]

    bt = WalkForwardBacktest(cfg, "hrp", use_ml=True)
    w_then, _ = bt.decide_at(
        {s: d.iloc[:601] for s, d in ohlcv.items()}, closes.iloc[:601], when=target)
    w_now, _ = bt.decide_at(ohlcv, closes, when=target)

    pd.testing.assert_series_equal(w_then, w_now, atol=1e-10)


def test_decide_at_refuses_without_enough_history(cfg):
    cfg.backtest.warmup = 260
    ohlcv, closes = _universe(n_bars=800)
    with pytest.raises(ValueError, match="historico insuficiente"):
        WalkForwardBacktest(cfg, "hrp").decide_at(ohlcv, closes, when=closes.index[50])


def test_decide_at_respects_the_exposure_cap(cfg):
    cfg.backtest.warmup = 260
    cfg.risk.max_gross_exposure = 0.40
    ohlcv, closes = _universe()
    w, _ = WalkForwardBacktest(cfg, "hrp", use_ml=False).decide_at(ohlcv, closes)
    assert float(w.sum()) <= 0.40 + 1e-9


def test_span_counts_real_days_even_with_a_single_gap(paths, cfg):
    """Regresion: con un solo tramo el informe decia '0 dias' porque media
    desde el final del primer tramo, no desde la primera decision."""
    _add("2026-09-08", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    _add("2026-09-10", {"BTC": 95.0}, {"BTC": 0.3}, cfg)  # falto el dia 9
    res = realised_returns(jn.read_journal(), pd.DataFrame(), 0.0, 0.0)
    assert res.n_gaps == 1
    assert res.span_days == 2


def test_missed_day_means_position_was_held_across_the_gap(paths, cfg):
    """Un dia sin anotar no se rellena: la posicion se mantuvo durante todo el
    hueco, que es lo que habria pasado de verdad."""
    _add("2026-09-08", {"BTC": 100.0}, {"BTC": 0.5}, cfg)
    _add("2026-09-10", {"BTC": 90.0}, {"BTC": 0.5}, cfg)
    res = realised_returns(jn.read_journal(), pd.DataFrame(), 0.0, 0.0)
    # -10% en BTC al 50% durante los dos dias: -5%, sin inventar el dia 9.
    assert float(res.returns.iloc[0]) == pytest.approx(-0.05)


# --------------------------------------------------------------------------
# Calendario: el modo en vivo debe ser el backtest, no una variante
# --------------------------------------------------------------------------
def test_live_schedule_matches_the_backtest(cfg):
    """El test que habria detectado el fallo de la fase 1.

    Se ejecuta el backtest y despues el modo en vivo barra a barra, dandole el
    mismo estado que tenia el backtest (pesos derivados y drawdown). Tienen que
    coincidir que dias se rebalancea y con que pesos. Recalcular una fecha
    pasada (lo que hacia H2) no lo comprueba: el modo en vivo antiguo era
    perfectamente reproducible y aun asi reentrenaba y rebalanceaba a diario.
    """
    cfg.backtest.warmup = 260
    ohlcv, closes = _universe(n_bars=400)
    bt = WalkForwardBacktest(cfg, "hrp", use_ml=True)
    res = bt.run(ohlcv, closes)
    index = closes.index

    # El ancla es el primer reentrenamiento real del backtest, que no tiene por
    # que ser su primera barra: si aun no hay datos para entrenar, lo aplaza.
    anchor = index.get_loc(res.model_reports[0]["date"])
    span = effective_retrain_every(cfg.backtest) + 10      # cubre dos reentrenamientos
    assert anchor + span < len(index) - 1
    eq = res.equity
    last_reb, prev_target, live_rebalances = None, None, []
    for i in range(anchor, anchor + span):
        t = index[i]
        dd = float(eq.loc[t] / eq.loc[:t].max() - 1.0)
        w, diag = bt.decide_at(
            ohlcv, closes, when=t, current_drawdown=dd,
            current_weights=res.weights.loc[t], anchor=index[anchor],
            last_rebalance=last_reb, previous_target=prev_target)
        if diag["action"] == "mantener":
            pd.testing.assert_series_equal(w, prev_target)
            continue
        assert diag["action"] == "rebalanceo"
        pd.testing.assert_series_equal(w, res.targets.loc[t], check_names=False, atol=1e-10)
        live_rebalances.append(t)
        last_reb, prev_target = t, w

    expected = [t for t in res.targets.index if index[anchor] <= t < index[anchor + span]]
    assert live_rebalances == expected
    assert len(expected) == span // cfg.backtest.rebalance_every
    # Que no pase comparando solo ceros: tiene que haber habido posiciones.
    assert float(res.targets.loc[expected].to_numpy().sum()) > 0


def test_missed_rebalance_is_done_late_with_the_scheduled_model(cfg):
    """Si el dia del rebalanceo el equipo estaba apagado, se rebalancea en la
    siguiente anotacion, con el modelo del calendario. No se opera el pasado."""
    cfg.backtest.warmup = 260
    ohlcv, closes = _universe(n_bars=360)
    bt = WalkForwardBacktest(cfg, "hrp", use_ml=True)
    idx = closes.index
    a = 300
    held = pd.Series(0.1, index=closes.columns)

    w, d = bt.decide_at(ohlcv, closes, when=idx[a + 3], anchor=idx[a],
                        last_rebalance=idx[a], previous_target=held)
    assert d["action"] == "mantener"
    pd.testing.assert_series_equal(w, held)

    _, d = bt.decide_at(ohlcv, closes, when=idx[a + 6], anchor=idx[a],
                        last_rebalance=idx[a], previous_target=held)
    assert d["action"] == "rebalanceo tardio"
    assert d["scheduled_rebalance"] == str(idx[a + 5].date())
    assert d["model_upto"] == str(idx[a].date())
    assert d["next_rebalance"] == str(idx[a + 10].date())


def test_retraining_happens_where_the_backtest_does(cfg):
    # run() solo reentrena en barras de rebalanceo: 63 con rebalanceo cada 5 es 65.
    assert effective_retrain_every(cfg.backtest) == 65
    cfg.backtest.retrain_every = 60
    assert effective_retrain_every(cfg.backtest) == 60


# --------------------------------------------------------------------------
# Volatilidad con tramos de distinta duracion (H1)
# --------------------------------------------------------------------------
def test_volatility_reduces_to_plain_std_with_daily_segments():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.02, 50))
    assert annualised_volatility(r, pd.Series(1.0, index=r.index)) == pytest.approx(
        float(r.std(ddof=1) * np.sqrt(365)))


def test_multi_day_segments_do_not_inflate_the_volatility():
    """Con el patron real (1-1-1-1-3 dias por semana), tratar cada tramo como
    un dia sobreestima la volatilidad; normalizando por duracion no."""
    rng = np.random.default_rng(7)
    days = np.tile([1, 1, 1, 1, 3], 2000).astype(float)
    r = rng.normal(0, 0.02 * np.sqrt(days))
    true = 0.02 * np.sqrt(365)
    naive = float(np.std(r, ddof=1) * np.sqrt(365))
    fixed = annualised_volatility(pd.Series(r), pd.Series(days))
    assert fixed == pytest.approx(true, rel=0.03)
    assert naive > true * 1.15


# --------------------------------------------------------------------------
# Costes y estado de la cartera
# --------------------------------------------------------------------------
def test_holding_the_target_still_pays_for_the_drift(paths, cfg):
    """Volver a los pesos objetivo despues de que el precio los mueva cuesta,
    igual que en el backtest."""
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.5}, cfg)
    _add("2026-01-02", {"BTC": 110.0}, {"BTC": 0.5}, cfg)
    _add("2026-01-03", {"BTC": 110.0}, {"BTC": 0.5}, cfg)
    res = realised_returns(jn.read_journal(), pd.DataFrame(), 0.001, 0.0005)
    drifted = 0.5 * 1.1 / 1.05
    assert float(res.costs.iloc[1]) == pytest.approx((drifted - 0.5) * 0.0015)


def test_state_includes_today_and_counts_a_first_loss_as_drawdown(paths, cfg):
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.5}, cfg)
    w, dd = portfolio_state_at(jn.read_journal(), "2026-01-02", {"BTC": 80.0}, 0.0, 0.0)
    # -20% al 50%: la cartera cae un 10% y el peso deriva a 0.4/0.9.
    assert dd == pytest.approx(-0.10)
    assert w["BTC"] == pytest.approx(0.4 / 0.9)


# --------------------------------------------------------------------------
# Enmiendas y escritura robusta
# --------------------------------------------------------------------------
def test_amendment_governs_only_later_records(paths, cfg):
    jn.write_preregistration(cfg, ["BTC"], "test")
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    _add("2026-01-02", {"BTC": 101.0}, {"BTC": 0.3}, cfg)
    with pytest.raises(ValueError):
        jn.write_amendment(jn.AMENDMENT_E1, "2026-01-02", cfg, "test")   # retroactiva

    a = jn.write_amendment(jn.AMENDMENT_E1, "2026-01-03", cfg, "test")
    assert a["journal_head"]["seq"] == 2
    rec = _add("2026-01-03", {"BTC": 102.0}, {"BTC": 0.3}, cfg)
    assert rec["amendment_hash"] == a["hash"]
    assert [r["seq"] for r in jn.active_records(jn.read_journal(), a)] == [3]
    assert jn.verify_chain().ok
    with pytest.raises(FileExistsError):
        jn.write_amendment(jn.AMENDMENT_E1, "2026-01-04", cfg, "test")


def test_edited_amendment_is_detected(paths, cfg):
    jn.write_preregistration(cfg, ["BTC"], "test")
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    jn.write_amendment(jn.AMENDMENT_E1, "2026-01-02", cfg, "test")
    data = json.loads(jn.AMENDMENTS_PATH.read_text(encoding="utf-8"))
    data[0]["anchor"] = "2026-01-05"
    jn.AMENDMENTS_PATH.write_text(json.dumps(data), encoding="utf-8")
    assert not jn.verify_chain().ok


def test_torn_last_line_is_reported_instead_of_crashing(paths, cfg):
    j, _ = paths
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    with j.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "decision_da')          # apagado a mitad de escritura
    status = jn.verify_chain()
    assert not status.ok
    assert "no es legible" in status.problems[0]
    with pytest.raises(jn.JournalCorrupt):
        jn.read_journal()


def test_append_leaves_no_temp_file(paths, cfg):
    j, _ = paths
    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)
    _add("2026-01-02", {"BTC": 101.0}, {"BTC": 0.3}, cfg)
    assert not list(j.parent.glob("*.tmp"))
    assert len(j.read_text(encoding="utf-8").splitlines()) == 2


def test_working_copy_cannot_write_the_experiment(tmp_path, monkeypatch, cfg):
    """La copia del USB lee el diario, pero no puede escribirlo: dos diarios
    avanzando a la vez bifurcarian el experimento."""
    data = tmp_path / "data"
    marker = tmp_path / "COPIA_DE_TRABAJO.txt"
    monkeypatch.setattr(jn, "DATA_DIR", data)
    monkeypatch.setattr(jn, "WORKING_COPY_MARKER", marker)
    monkeypatch.setattr(jn, "JOURNAL_PATH", data / "forward" / "journal.jsonl")
    monkeypatch.setattr(jn, "PREREG_PATH", data / "forward" / "preregistration.json")
    monkeypatch.setattr(jn, "AMENDMENTS_PATH", data / "forward" / "amendments.json")

    _add("2026-01-01", {"BTC": 100.0}, {"BTC": 0.3}, cfg)       # sin marcador: recolector
    marker.write_text("copia de trabajo", encoding="utf-8")
    with pytest.raises(jn.WorkingCopyError):
        _add("2026-01-02", {"BTC": 101.0}, {"BTC": 0.3}, cfg)
    with pytest.raises(jn.WorkingCopyError):
        jn.write_preregistration(cfg, ["BTC"], "test")
    with pytest.raises(jn.WorkingCopyError):
        jn.write_amendment(jn.AMENDMENT_E1, "2026-01-05", cfg, "test")
    assert len(jn.read_journal()) == 1                          # leer, si puede
    assert jn.verify_chain().ok
