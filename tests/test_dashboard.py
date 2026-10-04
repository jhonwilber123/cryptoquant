"""Tests del panel.

Lo que se comprueba no es el aspecto sino que el panel sea honesto y robusto:
que no invente datos cuando faltan, que su lectura no toque la red, que un
texto malicioso en los datos no pueda romperlo, y que cada decision que puede
tomar el sistema tenga su explicacion en lenguaje llano.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from cryptoquant import config as cfgmod
from cryptoquant.config import Config
from cryptoquant.dashboard import state as st
from cryptoquant.dashboard.render import TEMPLATE, build_dashboard, render_html, to_json
from cryptoquant.data import sources
from cryptoquant.data.sources import _synthetic, load_symbol, load_universe
from cryptoquant.forward import journal as jn
from cryptoquant.portfolio.sizing import decide_exposure


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Cache, informes y diario en un directorio temporal, y la red prohibida."""
    dirs = {name: tmp_path / name for name in ("cache", "reports", "forward")}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setattr(sources, "CACHE_DIR", dirs["cache"])
    monkeypatch.setattr(cfgmod, "REPORTS_DIR", dirs["reports"])
    monkeypatch.setattr(jn, "FORWARD_DIR", dirs["forward"])
    monkeypatch.setattr(jn, "JOURNAL_PATH", dirs["forward"] / "journal.jsonl")
    monkeypatch.setattr(jn, "PREREG_PATH", dirs["forward"] / "preregistration.json")
    monkeypatch.setattr(jn, "AMENDMENTS_PATH", dirs["forward"] / "amendments.json")

    def no_network(*_a, **_k):
        raise AssertionError("el panel no debe tocar la red")

    monkeypatch.setattr(sources, "_fetch_ccxt", no_network)
    return dirs


def _cfg(symbols=("A0", "A1", "A2")) -> Config:
    cfg = Config()
    cfg.data.symbols = list(symbols)
    return cfg


def _write_cache(cfg: Config, n: int = 400) -> None:
    for i, sym in enumerate(cfg.data.symbols):
        df = _synthetic(f"{sym}/USDT", "1d", "2024-01-01", n, seed=300 + i)
        path = sources._cache_path(cfg.data.exchange, f"{sym}/{cfg.data.quote}", cfg.data.timeframe)
        df.to_parquet(path)


def _record(cfg, date, weights, prices, reason="condiciones normales", pv=0.12):
    return jn.append_record(
        decision_date=date, prices=prices, weights=weights,
        gross_exposure=sum(weights.values()),
        diagnostics={"reason": reason, "port_vol_forecast": pv, "n_eligible": 2.0},
        cfg=cfg, code_version="test",
        state_in={"prev_weights": {}, "drawdown": -0.02},
    )


# --------------------------------------------------------------------------
# Degradacion honesta
# --------------------------------------------------------------------------
def test_nothing_available_still_renders(sandbox):
    """Sin cache, sin diario y sin backtest: el panel se genera y dice que no
    hay datos, en vez de fallar o inventar cifras."""
    state = st.collect_state(_cfg())
    assert state["today"] is None
    assert state["forward"] is None
    assert state["backtest"] is None
    assert state["market"] is None
    assert state["status"]["data"] is None
    html = render_html(state)
    assert "__STATE__" not in html


def test_viewer_never_touches_the_network(sandbox):
    """La cache esta atrasada a proposito. Aun asi el panel no descarga nada
    (el fixture hace fallar cualquier intento) y marca los datos como atrasados."""
    cfg = _cfg()
    _write_cache(cfg)
    state = st.collect_state(cfg)
    assert state["status"]["data"]["stale"] is True
    assert len(state["market"]) == 3


def test_offline_load_without_cache_refuses_instead_of_inventing(sandbox):
    with pytest.raises(FileNotFoundError):
        load_symbol("NOPE", _cfg().data, offline=True)


def test_offline_universe_skips_symbols_without_cache(sandbox):
    cfg = _cfg(("A0", "A1", "SIN_CACHE"))
    _write_cache(_cfg(("A0", "A1")))
    _, closes, origins = load_universe(cfg, offline=True)
    assert set(closes.columns) == {"A0", "A1"}
    assert set(origins.values()) == {"cache-stale"}


# --------------------------------------------------------------------------
# Contenido
# --------------------------------------------------------------------------
def test_today_reflects_the_last_journal_record(sandbox):
    cfg = _cfg()
    jn.write_preregistration(cfg, list(cfg.data.symbols), "test")
    _record(cfg, "2026-01-01", {"A0": 0.10, "A1": 0.05}, {"A0": 10.0, "A1": 20.0})
    _record(cfg, "2026-01-02", {"A0": 0.08, "A1": 0.20, "A2": 0.0},
            {"A0": 11.0, "A1": 19.0, "A2": 5.0}, reason="vol prevista por encima del objetivo")

    today = st.collect_state(cfg)["today"]
    assert today["decision_date"] == "2026-01-02"
    # Ordenado de mayor a menor peso y sin los activos a cero.
    assert [w["asset"] for w in today["weights"]] == ["A1", "A0"]
    assert today["gross"] == pytest.approx(0.28)
    assert today["cash"] == pytest.approx(0.72)
    assert today["explanation"]["template"] == st.EXPLANATIONS["vol prevista por encima del objetivo"]
    assert today["explanation"]["dd"] == pytest.approx(0.02)


def test_forward_section_counts_and_preregistered_years(sandbox):
    cfg = _cfg()
    jn.write_preregistration(cfg, list(cfg.data.symbols), "test")
    for i, px in enumerate((100.0, 102.0, 99.0)):
        _record(cfg, f"2026-01-0{i + 1}", {"A0": 0.5}, {"A0": px})

    fwd = st.collect_state(cfg)["forward"]
    assert fwd["n_records"] == 3
    assert fwd["curve"]["strategy"][0] == 0.0
    assert len(fwd["curve"]["x"]) == len(fwd["curve"]["strategy"]) == 3
    h3 = next(h for h in fwd["hypotheses"] if h["id"] == "H3")
    # Sin redondeo intermedio: redondear a 46,5 y luego a entero daba 47.
    assert h3["years"] == pytest.approx(16969 / 365)
    assert not h3["testable"]


def test_backtest_section_reads_the_saved_csvs(sandbox):
    cfg = _cfg()
    _write_cache(cfg)
    _, closes, _ = load_universe(cfg, offline=True)
    bench = closes.pct_change().fillna(0.0).mean(axis=1)
    rets = (bench * 0.3).iloc[100:]
    equity = pd.concat([pd.Series([10_000.0], index=[closes.index[99]]),
                        10_000.0 * (1 + rets).cumprod()])
    rets.rename("return").rename_axis("date").to_frame().to_csv(sandbox["reports"] / "backtest_returns.csv")
    equity.rename("equity").rename_axis("date").to_frame().to_csv(sandbox["reports"] / "backtest_equity.csv")

    bt = st.collect_state(cfg)["backtest"]
    assert len(bt["curve"]["x"]) == len(rets) + 1
    assert max(bt["drawdown"]["strategy"]) <= 0.0
    # La estrategia es el indice escalado al 30 %: la fraccion equivalente debe
    # salir ~0.3 y su volatilidad la misma que la de la estrategia.
    assert bt["matched"]["fraction"] == pytest.approx(0.3, rel=1e-6)
    assert bt["matched"]["metrics"]["volatility"] == pytest.approx(
        bt["metrics"]["strategy"]["volatility"], rel=1e-9)


# --------------------------------------------------------------------------
# Seguridad y formato
# --------------------------------------------------------------------------
def test_json_is_strict_and_cannot_close_the_script_tag():
    state = {"texto": "</script><script>alert(1)</script>", "nan": float("nan"),
             "inf": float("inf"), "np": np.float64(1.5), "ok": True}
    text = to_json(state)
    assert "</script" not in text.lower()
    parsed = json.loads(text)                 # JSON estricto: NaN no es valido
    assert parsed["nan"] is None and parsed["inf"] is None
    assert parsed["np"] == 1.5 and parsed["ok"] is True
    assert parsed["texto"] == "</script><script>alert(1)</script>"


def test_template_has_exactly_one_placeholder():
    assert TEMPLATE.read_text(encoding="utf-8").count("__STATE__") == 1


def test_build_is_atomic_and_leaves_no_temp_file(sandbox, tmp_path):
    out = tmp_path / "panel.html"
    assert build_dashboard(_cfg(), out) == out
    assert out.exists()
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------
# Explicaciones
# --------------------------------------------------------------------------
def test_every_sizing_decision_has_a_plain_language_explanation():
    """Recorre todas las ramas reales de decide_exposure. Si algun dia se anade
    un motivo nuevo sin explicacion, el panel mostraria la clave interna."""
    cfg = Config()
    seen = set()
    for p in (0.30, 0.52, 0.60, 0.90):
        for fv in (0.05, 0.20, 0.50):
            for dd in (0.0, -0.15, -0.30):
                for regime in (True, False):
                    d = decide_exposure(p, 2.0 / 1.5, fv, dd, regime, cfg.risk)
                    seen.add(d.reason)
    seen.add("ningun activo elegible")
    missing = seen - set(st.EXPLANATIONS)
    assert not missing, f"motivos sin explicacion: {missing}"
    assert len(seen) >= 8, "la rejilla deberia cubrir todas las ramas"


def test_explanations_only_use_placeholders_the_page_fills():
    for text in st.EXPLANATIONS.values():
        assert set(re.findall(r"\{(\w+)\}", text)) <= {"pv", "tv", "dd"}


# --------------------------------------------------------------------------
# Estado de la automatizacion
# --------------------------------------------------------------------------
def test_automation_status_ignores_panel_lines_and_detects_failure(tmp_path):
    log = tmp_path / "record.log"
    log.write_text(
        "\ufeff2026-09-11 00:30:00Z [INFO] iniciando registro diario\n"
        "    Cargando datos de mercado...\n"
        "2026-09-11 00:30:09Z [FALLO] el comando devolvio codigo 1\n"
        "2026-09-11 00:30:15Z [PANEL] regenerado\n",
        encoding="utf-8",
    )
    s = st.automation_status(log)
    # La regeneracion del panel no debe tapar que el registro fallo.
    assert s["level"] == "FALLO" and s["ok"] is False


def test_automation_status_reports_success(tmp_path):
    log = tmp_path / "record.log"
    log.write_text("2026-09-11 00:30:00Z [INFO] inicio\n"
                   "2026-09-11 00:30:05Z [OK] anotacion registrada (#3)\n", encoding="utf-8")
    s = st.automation_status(log)
    assert s["ok"] is True
    assert s["last_run_utc"].startswith("2026-09-11T00:30:05")


def test_missing_log_means_unknown_not_ok(tmp_path):
    assert st.automation_status(tmp_path / "no_existe.log") is None


def test_journal_charts_include_phase_one_and_mark_the_amendment(sandbox):
    """Las hipotesis cuentan desde la enmienda, pero los graficos del dia a dia
    muestran toda la cartera simulada y senalan donde empieza el recuento."""
    cfg = _cfg()
    jn.write_preregistration(cfg, list(cfg.data.symbols), "test")
    _record(cfg, "2026-01-01", {"A1": 0.2}, {"A1": 10.0})
    _record(cfg, "2026-01-02", {"A1": 0.2, "A0": 0.1}, {"A1": 11.0, "A0": 5.0})
    jn.write_amendment(jn.AMENDMENT_E1, "2026-01-04", cfg, "test")
    _record(cfg, "2026-01-04", {"A0": 0.3}, {"A1": 12.0, "A0": 5.5})

    fwd = st.collect_state(cfg)["forward"]
    j = fwd["journal"]
    assert fwd["n_records"] == 1 and fwd["protocol"]["archived"] == 2
    assert len(j["x"]) == len(j["curve"]["strategy"]) == 3
    assert j["phase_start"] == j["x"][2]
    # Orden del universo, no de aparicion: el color sigue al activo.
    assert j["assets"] == ["A0", "A1"]
    assert j["weights"]["A0"] == [0.0, 0.1, 0.3]


def test_market_history_carries_enough_bars_for_the_moving_average(sandbox):
    cfg = _cfg()
    _write_cache(cfg, n=1500)
    _, closes, _ = load_universe(cfg, offline=True)   # sin las barras aun no cerradas
    h = st.collect_state(cfg)["history"]
    assert h["ma_window"] == cfg.signal.regime_ma
    assert len(h["x"]) == min(len(closes), 365 * 3 + cfg.signal.regime_ma)
    assert all(len(v) == len(h["x"]) for v in h["prices"].values())
    assert h["x"] == sorted(h["x"]) and h["x"][1] - h["x"][0] == 86_400_000
