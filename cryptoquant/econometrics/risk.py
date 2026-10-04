"""Medidas de riesgo.

VaR y CVaR con tres estimadores distintos porque en cripto el supuesto normal
subestima sistematicamente el riesgo de cola: la diferencia entre el VaR
parametrico y el historico es, en la practica, la diferencia entre sobrevivir
un crash y no sobrevivirlo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RiskMetrics:
    var_historical: float
    var_parametric: float
    var_cornish_fisher: float
    cf_used_fallback: bool  # True si C-F salio de su dominio y se uso el historico
    cvar_historical: float
    volatility_ann: float
    skew: float
    kurtosis: float
    max_drawdown: float
    ulcer_index: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def var_historical(returns: pd.Series, alpha: float = 0.95) -> float:
    """VaR empirico. Devuelto como perdida positiva."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float(-np.quantile(r, 1 - alpha))


def var_parametric(returns: pd.Series, alpha: float = 0.95) -> float:
    """VaR gaussiano. Se incluye como referencia: casi siempre es el mas
    optimista de los tres y por eso no debe usarse solo."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float(-(r.mean() + stats.norm.ppf(1 - alpha) * r.std(ddof=1)))


def _cf_is_monotonic(z: float, s: float, k: float) -> bool:
    """Comprueba que la expansion de Cornish-Fisher sea valida en este punto.

    La expansion solo es una transformacion monotona del cuantil normal dentro
    de un dominio acotado de (asimetria, curtosis). Fuera de el deja de ser una
    funcion cuantil y devuelve resultados sin sentido: con series muy
    leptocurticas puede dar un VaR negativo, es decir "es imposible perder", que
    es exactamente lo contrario de lo que ocurre. Se verifica la derivada sobre
    el tramo de cola relevante en lugar de confiar en la formula a ciegas.
    """
    grid = np.linspace(min(z, 0.0) - 0.5, 0.0, 40)
    dz = (
        1.0
        + 2 * grid * s / 6
        + (3 * grid**2 - 3) * k / 24
        - (6 * grid**2 - 5) * s**2 / 36
    )
    return bool(np.all(dz > 0))


def var_cornish_fisher(returns: pd.Series, alpha: float = 0.95,
                       fallback: bool = True) -> float:
    """VaR con expansion de Cornish-Fisher: corrige el cuantil normal por
    asimetria y curtosis.

    Si los momentos muestrales caen fuera del dominio de validez de la
    expansion, se devuelve el VaR historico (o NaN si `fallback` es False).
    Devolver el valor de la formula en ese caso seria peor que no calcularlo.
    """
    r = returns.dropna()
    if len(r) < 30:
        return np.nan
    z = stats.norm.ppf(1 - alpha)
    s = float(stats.skew(r))
    k = float(stats.kurtosis(r))  # exceso de curtosis

    if not _cf_is_monotonic(z, s, k):
        return var_historical(r, alpha) if fallback else np.nan

    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    var = float(-(r.mean() + z_cf * r.std(ddof=1)))
    # Red de seguridad: un VaR no positivo con retornos que tienen cola
    # izquierda real siempre es un artefacto numerico.
    if var <= 0 and (r < 0).any():
        return var_historical(r, alpha) if fallback else np.nan
    return var


def cvar_historical(returns: pd.Series, alpha: float = 0.95) -> float:
    """Expected Shortfall: perdida media condicionada a estar en la cola.

    Responde a "si el mal dia ocurre, cuanto pierdo de media", que es la
    pregunta relevante para dimensionar posiciones.
    """
    r = returns.dropna()
    if r.empty:
        return np.nan
    threshold = np.quantile(r, 1 - alpha)
    tail = r[r <= threshold]
    return float(-tail.mean()) if len(tail) else float(-threshold)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else np.nan


def ulcer_index(equity: pd.Series) -> float:
    """Indice de Ulcer: RMS del drawdown. Penaliza caidas profundas Y largas,
    a diferencia del max drawdown que solo mira el peor punto."""
    dd = drawdown_series(equity)
    return float(np.sqrt((dd**2).mean())) if len(dd) else np.nan


def compute_risk(returns: pd.Series, ann: float, alpha: float = 0.95) -> RiskMetrics:
    r = returns.dropna()
    equity = (1 + r).cumprod()
    return RiskMetrics(
        var_historical=var_historical(r, alpha),
        var_parametric=var_parametric(r, alpha),
        var_cornish_fisher=var_cornish_fisher(r, alpha),
        cf_used_fallback=not _cf_is_monotonic(
            stats.norm.ppf(1 - alpha), float(stats.skew(r)), float(stats.kurtosis(r))
        ) if len(r) >= 30 else False,
        cvar_historical=cvar_historical(r, alpha),
        volatility_ann=float(r.std(ddof=1) * np.sqrt(ann)) if len(r) > 1 else np.nan,
        skew=float(stats.skew(r)) if len(r) > 2 else np.nan,
        kurtosis=float(stats.kurtosis(r)) if len(r) > 3 else np.nan,
        max_drawdown=max_drawdown(equity),
        ulcer_index=ulcer_index(equity),
    )


def portfolio_var(weights: np.ndarray, cov: np.ndarray, alpha: float = 0.95) -> float:
    """VaR parametrico de cartera a un periodo."""
    sigma = float(np.sqrt(weights @ cov @ weights))
    return float(-stats.norm.ppf(1 - alpha) * sigma)


def marginal_risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Contribucion de cada activo al riesgo total de la cartera.

    Un peso del 10% no significa un 10% del riesgo: si el activo es volatil y
    esta correlacionado con el resto, puede aportar el 40%. Esta funcion es la
    que revela la concentracion real de riesgo.
    """
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if port_vol <= 0:
        return np.zeros_like(weights)
    mrc = (cov @ weights) / port_vol
    return weights * mrc / port_vol  # normalizado: suma 1
