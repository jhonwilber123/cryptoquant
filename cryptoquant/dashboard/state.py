"""Estado del sistema para el panel.

Reune en un solo diccionario serializable todo lo que el panel muestra: la
decision vigente, el progreso del experimento, el backtest historico y el
estado del mercado.

Es estrictamente de solo lectura: no descarga datos, no entrena modelos y no
escribe en el diario. Un visor con efectos secundarios podria alterar justo lo
que se supone que solo observa, y ademas tardaria minutos en abrirse.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfgmod
from ..backtest.metrics import summarize
from ..config import Config, annualization_factor
from ..data.sources import load_universe
from ..forward import journal as jn
from ..forward.evaluate import annualised_volatility, observations_required, realised_returns
from ..models.validation import deflated_sharpe_ratio, probabilistic_sharpe_ratio

# Mismo numero de pruebas que usa `backtest` al deflactar el Sharpe, para que
# el panel y la linea de comandos digan exactamente lo mismo.
N_TRIALS = 12


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ms(index: pd.DatetimeIndex) -> list[int]:
    idx = index if index.tz is not None else index.tz_localize("UTC")
    # asi8 va en la unidad del indice: el parquet puede traer ms o us, no ns.
    if hasattr(idx, "as_unit"):
        idx = idx.as_unit("ns")
    return [int(v) for v in idx.asi8 // 1_000_000]


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def clean(o: Any) -> Any:
    """Convierte a tipos JSON validos. NaN e infinito pasan a null: JSON no los
    admite y un NaN en el panel se leeria como un fallo."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    return o


# --------------------------------------------------------------------------
# Explicaciones en lenguaje llano
# --------------------------------------------------------------------------
# Una por cada categoria que emite `sizing.decide_exposure`, mas la del motor
# cuando no hay ningun activo elegible. Un test comprueba que no falte ninguna.
EXPLANATIONS: dict[str, str] = {
    "vol prevista muy por encima del objetivo":
        "El mercado está muy agitado: la volatilidad prevista ({pv}) es más del "
        "doble de tu objetivo ({tv}). Por eso el sistema invierte solo una parte "
        "y deja el resto en efectivo.",
    "vol prevista por encima del objetivo":
        "La volatilidad prevista ({pv}) supera tu objetivo ({tv}), así que el "
        "sistema reduce la exposición para compensar.",
    "condiciones normales":
        "Condiciones normales: la volatilidad prevista ({pv}) cabe dentro de tu "
        "objetivo ({tv}).",
    "conviccion baja":
        "Las señales son débiles, así que el sistema invierte menos de lo que tu "
        "presupuesto de riesgo permitiría.",
    "regimen bajista: exposicion reducida":
        "Más de la mitad del mercado está en tendencia bajista: la exposición se "
        "recorta al 25 % de lo normal.",
    "desapalancado por drawdown":
        "La cartera acumula una caída del {dd} desde su máximo; el freno reduce "
        "la exposición hasta que se recupere.",
    "cortacircuito por drawdown maximo":
        "Se ha alcanzado la caída máxima permitida ({dd}): todo en efectivo hasta "
        "que cambien las condiciones.",
    "sin ventaja estadistica (Kelly <= 0)":
        "Los modelos no ven ventaja estadística ahora mismo: todo en efectivo.",
    "ningun activo elegible":
        "Ningún activo cumple a la vez las condiciones de tendencia y de señal: "
        "todo en efectivo.",
}


def explanation(reason: str, pv: float | None, tv: float | None, dd: float) -> dict:
    """Plantilla + valores, sin formatear.

    Los numeros los formatea el navegador con la configuracion regional del
    usuario, igual que el resto del panel. Formatearlos aqui mezclaba "35,3 %"
    en el texto con "35.3 %" en la ficha de al lado.
    """
    return {
        "template": EXPLANATIONS.get(reason) or reason or "Sin explicación disponible.",
        "pv": pv, "tv": tv, "dd": abs(dd),
    }


# --------------------------------------------------------------------------
# Secciones
# --------------------------------------------------------------------------
_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}Z) \[([A-Z-]+)\] (.*)$")
# Solo estas lineas cierran una ejecucion. INFO marca el inicio y PANEL la
# regeneracion de este mismo panel, que no debe tapar el resultado del registro.
_FINAL_LEVELS = {"OK", "FALLO", "EXCEPCION", "ERROR"}


def automation_status(log_path: Path) -> dict | None:
    """Resultado de la ultima ejecucion de la tarea programada, leido del log."""
    if not log_path.exists():
        return None
    last = None
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.lstrip("﻿")          # PowerShell 5.1 escribe UTF-8 con BOM
        if line.startswith(" "):             # salida del subproceso, indentada
            continue
        m = _LOG_LINE.match(line.strip())
        if m and m.group(2) in _FINAL_LEVELS:
            last = m
    if last is None:
        return None
    ts = datetime.strptime(last.group(1), "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
    return {
        "last_run_utc": ts.isoformat(),
        "level": last.group(2),
        "message": last.group(3),
        "ok": last.group(2) == "OK",
    }


def status_state(origins: dict[str, str], closes: pd.DataFrame | None) -> dict:
    chain = jn.verify_chain()
    data = None
    if closes is not None and not closes.empty:
        data = {
            "last_bar": str(closes.index[-1].date()),
            "stale": any(o != "cache" for o in origins.values()),
        }
    return {
        "journal_ok": chain.ok,
        "journal_records": chain.n_records,
        "journal_problems": chain.problems,
        "prereg_present": jn.read_preregistration() is not None,
        "automation": automation_status(jn.FORWARD_DIR / "record.log"),
        "data": data,
    }


def today_state(records: list[dict], n_universe: int) -> dict | None:
    """La decision vigente: la ultima anotacion del diario."""
    if not records:
        return None
    rec = records[-1]
    diag = rec.get("diagnostics") or {}
    prices = rec.get("prices") or {}
    tv = _num(rec.get("target_volatility"))
    pv = _num(diag.get("port_vol_forecast"))
    dd = _num((rec.get("state_in") or {}).get("drawdown")) or 0.0

    weights = [
        {"asset": a, "weight": float(w), "price": _num(prices.get(a))}
        for a, w in sorted(rec.get("weights", {}).items(), key=lambda kv: -kv[1])
        if w > 1e-4
    ]
    gross = _num(rec.get("gross_exposure"))
    if gross is None:
        gross = sum(w["weight"] for w in weights)
    reason = str(diag.get("reason", ""))
    n_elig = _num(diag.get("n_eligible"))

    return {
        "decision_date": rec.get("decision_date"),
        "seq": rec.get("seq"),
        "weights": weights,
        "gross": gross,
        "cash": max(0.0, 1.0 - gross),
        "port_vol_forecast": pv,
        "target_volatility": tv,
        "n_eligible": int(n_elig) if n_elig is not None else None,
        "n_universe": n_universe,
        "drawdown": dd,
        "reason": reason,
        "explanation": explanation(reason, pv, tv, dd),
        # Solo desde la enmienda E1; las anotaciones anteriores no lo tienen.
        "action": diag.get("action"),
        "next_rebalance": diag.get("next_rebalance"),
    }


def forward_state(all_records: list[dict], prereg, cfg: Config, ann: float,
                  amendment: dict | None = None) -> dict | None:
    if prereg is None and not all_records:
        return None
    # Solo cuentan las anotaciones del protocolo vigente.
    records = jn.active_records(all_records, amendment)
    n = len(records)
    invested = sum(1 for r in records if (_num(r.get("gross_exposure")) or 0) >= 0.01)
    out: dict[str, Any] = {"n_records": n, "n_invested": invested, "curve": None,
                           "cum_return": None, "bench_return": None,
                           "vol_realised": None, "hypotheses": [],
                           "protocol": None,
                           "journal": journal_state(all_records, cfg, amendment)}
    if amendment:
        out["protocol"] = {"id": amendment["id"], "title": amendment.get("title"),
                           "effective_from": amendment["effective_from"],
                           "archived": len(all_records) - n}

    if records:
        first = date.fromisoformat(records[0]["decision_date"])
        last = date.fromisoformat(records[-1]["decision_date"])
        span = (last - first).days + 1
        out.update({
            "first": str(first), "last": str(last),
            "coverage": n / span if span > 0 else None,
            "missing_days": max(span - n, 0),
            "days_since_last": (datetime.now(timezone.utc).date() - last).days,
        })

    if n >= 2:
        res = realised_returns(records, pd.DataFrame(), cfg.backtest.fee, cfg.backtest.slippage)
        if len(res.returns):
            eq = (1 + res.returns).cumprod()
            bench = (1 + res.benchmark).cumprod()
            out["cum_return"] = float(eq.iloc[-1] - 1)
            out["bench_return"] = float(bench.iloc[-1] - 1)
            if len(res.returns) >= 2:
                out["vol_realised"] = annualised_volatility(res.returns, res.days, ann)
            start_ms = _ms(pd.DatetimeIndex([pd.Timestamp(records[0]["decision_date"])]))
            out["curve"] = {
                "x": start_ms + _ms(pd.DatetimeIndex(res.returns.index)),
                "strategy": [0.0] + [round(float(v - 1), 6) for v in eq],
                "benchmark": [0.0] + [round(float(v - 1), 6) for v in bench],
            }

    if prereg is not None:
        for h in prereg.hypotheses:
            need = int(h.get("min_observations", 0))
            have = invested if h["id"] == "H1" else n
            out["hypotheses"].append({
                "id": h["id"], "claim": h["claim"], "testable": bool(h["testable"]),
                "have": have, "need": need,
                # Sin redondear: 46,49 redondeado a 46,5 y luego a entero daba
                # "47 anos", mientras el veredicto del backtest decia "46".
                "years": need / ann if not h["testable"] else None,
            })
    return out


def journal_state(all_records: list[dict], cfg: Config,
                  amendment: dict | None) -> dict | None:
    """Todo el diario, fase 1 incluida, para los graficos del dia a dia.

    Las hipotesis solo cuentan el protocolo vigente, pero la cartera simulada
    es una sola desde el primer dia: se dibuja entera y la fase 1 se marca.
    """
    if not all_records:
        return None
    held = {a for r in all_records for a, w in (r.get("weights") or {}).items() if w > 1e-4}
    universe = list(cfg.data.symbols)
    assets = [s for s in universe if s in held] + sorted(held - set(universe))
    x = _ms(pd.DatetimeIndex(pd.to_datetime([r["decision_date"] for r in all_records])))
    diags = [r.get("diagnostics") or {} for r in all_records]
    out: dict[str, Any] = {
        "x": x,
        "universe": universe,
        "assets": assets,
        "weights": {a: [round(float((r.get("weights") or {}).get(a, 0.0)), 5)
                        for r in all_records] for a in assets},
        "gross": [_num(r.get("gross_exposure")) for r in all_records],
        "action": [d.get("action") for d in diags],
        "vol_forecast": [_num(d.get("port_vol_forecast")) for d in diags],
        "target_vol": _num(all_records[-1].get("target_volatility")),
        "phase_start": (_ms(pd.DatetimeIndex([pd.Timestamp(amendment["effective_from"])]))[0]
                        if amendment else None),
        "curve": None,
    }
    if len(all_records) >= 2:
        res = realised_returns(all_records, pd.DataFrame(), cfg.backtest.fee, cfg.backtest.slippage)
        if len(res.returns) == len(all_records) - 1:
            eq, bench = (1 + res.returns).cumprod(), (1 + res.benchmark).cumprod()
            out["curve"] = {
                "strategy": [0.0] + [round(float(v - 1), 6) for v in eq],
                "benchmark": [0.0] + [round(float(v - 1), 6) for v in bench],
            }
    return out


def market_history(closes: pd.DataFrame, regime_ma: int, years: int = 3) -> dict | None:
    """Cierres recientes por activo para el explorador de mercado.

    Se envian `regime_ma` barras de mas para que la media movil ya este
    formada en el primer dia visible; la media se calcula en la pagina.
    """
    if closes is None or closes.empty:
        return None
    sub = closes.iloc[-(365 * years + regime_ma):]
    return {
        "x": _ms(_utc_index(sub.index)),
        "ma_window": regime_ma,
        "prices": {a: [float(f"{v:.6g}") if np.isfinite(v) else None for v in sub[a].to_numpy(float)]
                   for a in sub.columns},
    }


def backtest_state(closes: pd.DataFrame, cfg: Config, ann: float) -> dict | None:
    """Resultado del ultimo backtest, leido de los CSV que deja `backtest`.

    No se recalcula aqui: con reentrenamientos walk-forward tarda minutos. La
    referencia buy & hold se reconstruye exactamente como en el motor: la media
    de los retornos simples del universo en las mismas fechas.
    """
    rdir = cfgmod.REPORTS_DIR
    rp, ep = rdir / "backtest_returns.csv", rdir / "backtest_equity.csv"
    if not (rp.exists() and ep.exists()):
        return None

    rets = pd.read_csv(rp, index_col=0)["return"]
    rets.index = _utc_index(pd.to_datetime(rets.index, utc=True))
    eq_raw = pd.read_csv(ep, index_col=0)["equity"]
    capital = float(eq_raw.iloc[0])
    start = pd.to_datetime(eq_raw.index[0], utc=True)

    simple = closes.pct_change().fillna(0.0)
    common = rets.index.intersection(simple.index)
    if len(common) < 30:
        return None
    rets = rets.loc[common]
    bench = simple.loc[common].mean(axis=1)

    strat_eq = capital * (1 + rets).cumprod()
    bench_eq = capital * (1 + bench).cumprod()
    x = _ms(pd.DatetimeIndex([start])) + _ms(rets.index)
    s_curve = [capital] + [round(float(v), 2) for v in strat_eq]
    b_curve = [capital] + [round(float(v), 2) for v in bench_eq]
    s_full = pd.Series(s_curve)
    b_full = pd.Series(b_curve)

    ms, mb = summarize(rets, ann), summarize(bench, ann)
    psr = probabilistic_sharpe_ratio(ms["sharpe"], ms["n_periods"], 0.0,
                                     ms["skew"], ms["kurtosis"] + 3, ann)
    dsr = deflated_sharpe_ratio(ms["sharpe"], ms["n_periods"], N_TRIALS,
                                ms["skew"], ms["kurtosis"] + 3, ann)

    # La comparacion que de verdad importa: buy & hold con la MISMA volatilidad.
    frac = float(rets.std(ddof=1) / bench.std(ddof=1)) if bench.std(ddof=1) > 0 else None
    matched = None
    years = None
    if frac:
        naive = bench * frac
        mn = summarize(naive, ann)
        diff = rets - naive
        excess = float(diff.mean() * ann)
        te = float(diff.std(ddof=1) * np.sqrt(ann))
        req = observations_required(excess, te, periods_per_year=ann)
        years = req["years_significance"]
        matched = {"fraction": frac, "metrics": _pick(mn),
                   "final": capital * float((1 + naive).prod()),
                   "excess": excess, "tracking_error": te}

    monthly = pd.DataFrame({"s": strat_eq, "b": bench_eq}).resample("ME").last()

    return {
        "generated": datetime.fromtimestamp(rp.stat().st_mtime, timezone.utc).isoformat(),
        "capital": capital,
        "curve": {"x": x, "strategy": s_curve, "benchmark": b_curve},
        "drawdown": {
            "x": x,
            "strategy": [round(float(v), 5) for v in (s_full / s_full.cummax() - 1)],
            "benchmark": [round(float(v), 5) for v in (b_full / b_full.cummax() - 1)],
        },
        "metrics": {
            "strategy": {**_pick(ms), "final": float(strat_eq.iloc[-1])},
            "benchmark": {**_pick(mb), "final": float(bench_eq.iloc[-1])},
        },
        "matched": matched,
        "significance": {"sharpe": ms["sharpe"], "psr": psr, "dsr": dsr,
                         "n_trials": N_TRIALS, "years_needed": years},
        "monthly": [
            {"month": idx.strftime("%Y-%m"), "strategy": round(float(r.s), 2),
             "benchmark": round(float(r.b), 2)}
            for idx, r in monthly.iterrows()
        ],
    }


def _pick(m: dict) -> dict:
    keys = ("cagr", "volatility", "sharpe", "sortino", "calmar", "max_drawdown", "martin_ratio")
    return {k: m.get(k) for k in keys}


def market_state(closes: pd.DataFrame, regime_ma: int, ann: float) -> list[dict] | None:
    if closes is None or closes.empty:
        return None
    logret = np.log(closes).diff()
    rows = []
    for a in closes.columns:
        c = closes[a].dropna()
        if len(c) < 31:
            continue
        ma = c.rolling(regime_ma, min_periods=regime_ma // 2).mean()
        rows.append({
            "asset": a,
            "price": float(c.iloc[-1]),
            "chg_7d": float(c.iloc[-1] / c.iloc[-8] - 1),
            "chg_30d": float(c.iloc[-1] / c.iloc[-31] - 1),
            "vol_30d": float(logret[a].iloc[-30:].std(ddof=1) * np.sqrt(ann)),
            "trend_up": bool(c.iloc[-1] > ma.iloc[-1]),
            "dd_from_high": float(c.iloc[-1] / c.max() - 1),
        })
    return rows


# --------------------------------------------------------------------------
def leer_riesgo() -> dict | None:
    """Lo que dejo `riesgo`. Como el backtest, se lee y no se recalcula."""
    from ..riesgo.informe import leer_estado

    return leer_estado()


# --------------------------------------------------------------------------
def collect_state(cfg: Config) -> dict:
    ann = annualization_factor(cfg.data.timeframe)
    try:
        _, closes, origins = load_universe(cfg, offline=True)
    except (FileNotFoundError, ValueError, KeyError):
        closes, origins = None, {}

    try:
        records = jn.read_journal()
    except jn.JournalCorrupt:
        # El panel no repara nada: muestra vacio y `forward verify` explica por que.
        records = []
    prereg = jn.read_preregistration()
    amendment = jn.current_amendment()
    n_universe = len(closes.columns) if closes is not None else len(cfg.data.symbols)

    return clean({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "target_volatility": cfg.risk.target_volatility,
            "drawdown_derisk_start": cfg.risk.drawdown_derisk_start,
            "max_drawdown_stop": cfg.risk.max_drawdown_stop,
            "initial_capital": cfg.backtest.initial_capital,
            "symbols": list(cfg.data.symbols),
            "quote": cfg.data.quote,
        },
        "status": status_state(origins, closes),
        "today": today_state(records, n_universe),
        "forward": forward_state(records, prereg, cfg, ann, amendment),
        "backtest": backtest_state(closes, cfg, ann) if closes is not None else None,
        "market": market_state(closes, cfg.signal.regime_ma, ann) if closes is not None else None,
        "history": market_history(closes, cfg.signal.regime_ma) if closes is not None else None,
        "risk": leer_riesgo(),
    })
