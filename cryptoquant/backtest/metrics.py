"""Metricas de rendimiento ajustado por riesgo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..econometrics.risk import drawdown_series, max_drawdown, ulcer_index


def _ann_return(returns: pd.Series, ann: float) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    total = float((1 + r).prod())
    if total <= 0:
        return -1.0
    return float(total ** (ann / len(r)) - 1)


def summarize(returns: pd.Series, ann: float, rf: float = 0.0) -> dict[str, float]:
    """Panel completo de metricas para una serie de retornos periodicos."""
    r = returns.dropna()
    if len(r) < 5:
        return {}

    equity = (1 + r).cumprod()
    ann_ret = _ann_return(r, ann)
    ann_vol = float(r.std(ddof=1) * np.sqrt(ann))
    excess = ann_ret - rf

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(ann)) if len(downside) > 1 else np.nan
    mdd = max_drawdown(equity)

    # Ratio de acierto y factor de beneficio sobre barras, no sobre trades.
    wins, losses = r[r > 0], r[r < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf

    dd = drawdown_series(equity)
    underwater = int((dd < -1e-9).sum())

    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": ann_ret,
        "volatility": ann_vol,
        "sharpe": float(excess / ann_vol) if ann_vol > 0 else np.nan,
        "sortino": float(excess / downside_vol) if downside_vol and downside_vol > 0 else np.nan,
        "calmar": float(ann_ret / abs(mdd)) if mdd and mdd < 0 else np.nan,
        "max_drawdown": mdd,
        "ulcer_index": ulcer_index(equity),
        # Martin ratio: retorno por unidad de "dolor" (drawdown sostenido).
        "martin_ratio": float(ann_ret / ulcer_index(equity)) if ulcer_index(equity) > 0 else np.nan,
        "win_rate": float((r > 0).mean()),
        "profit_factor": profit_factor,
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "pct_underwater": float(underwater / len(r)),
        "n_periods": int(len(r)),
    }


def compare(strategies: dict[str, pd.Series], ann: float) -> pd.DataFrame:
    """Tabla comparativa de varias series de retornos."""
    rows = {name: summarize(ret, ann) for name, ret in strategies.items()}
    df = pd.DataFrame(rows).T
    order = [
        "cagr", "volatility", "sharpe", "sortino", "calmar", "max_drawdown",
        "ulcer_index", "martin_ratio", "win_rate", "profit_factor",
        "worst_period", "skew", "kurtosis", "n_periods",
    ]
    return df[[c for c in order if c in df.columns]]


def rolling_sharpe(returns: pd.Series, window: int, ann: float) -> pd.Series:
    mu = returns.rolling(window, min_periods=window // 2).mean() * ann
    sd = returns.rolling(window, min_periods=window // 2).std(ddof=1) * np.sqrt(ann)
    return mu / sd.replace(0.0, np.nan)


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """Retornos mensuales en formato ano x mes."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    monthly = (1 + r).resample("ME").prod() - 1
    tbl = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.to_numpy()}
    )
    return tbl.pivot_table(index="year", columns="month", values="ret")
