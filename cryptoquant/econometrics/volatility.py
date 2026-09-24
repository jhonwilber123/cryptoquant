"""Modelizacion de volatilidad.

En cripto la volatilidad no es constante ni independiente: se agrupa (dias
volatiles siguen a dias volatiles) y responde de forma asimetrica a las malas
noticias. Un GJR-GARCH captura ambas cosas; el EWMA es el respaldo barato y
robusto cuando el GARCH no converge.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolForecast:
    sigma_ann: float          # volatilidad anualizada prevista para t+1
    model: str                # 'gjr-garch' | 'ewma'
    persistence: float        # alpha+beta: cercano a 1 => shocks duraderos
    converged: bool


def ewma_forecast(returns: pd.Series, ann: float, lam: float = 0.94) -> VolForecast:
    """EWMA estilo RiskMetrics. Sin optimizacion, no falla nunca."""
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 20:
        return VolForecast(np.nan, "ewma", np.nan, False)
    var = float(np.var(r[: min(30, len(r))]))
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return VolForecast(float(np.sqrt(var * ann)), "ewma", lam, True)


def garch_forecast(returns: pd.Series, ann: float, dist: str = "skewt") -> VolForecast:
    """GJR-GARCH(1,1,1) con innovaciones t asimetrica.

    El termino GJR anade un coeficiente gamma que solo se activa con retornos
    negativos: modela el efecto apalancamiento (las caidas generan mas
    volatilidad futura que las subidas de igual magnitud).
    """
    r = returns.dropna()
    if len(r) < 250:
        return ewma_forecast(r, ann)
    try:
        from arch import arch_model
    except ImportError:
        return ewma_forecast(r, ann)

    # `arch` trabaja mejor con retornos escalados a unidades de porcentaje.
    scaled = r.to_numpy(dtype=float) * 100.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(scaled, vol="GARCH", p=1, o=1, q=1, dist=dist, mean="Constant")
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 300})
            fc = res.forecast(horizon=1, reindex=False)
            var_next = float(fc.variance.iloc[-1, 0])
            sigma_daily = np.sqrt(var_next) / 100.0
            p = res.params
            persistence = float(
                p.get("alpha[1]", 0.0) + p.get("beta[1]", 0.0) + 0.5 * p.get("gamma[1]", 0.0)
            )
        if not np.isfinite(sigma_daily) or sigma_daily <= 0:
            return ewma_forecast(r, ann)
        return VolForecast(float(sigma_daily * np.sqrt(ann)), "gjr-garch", persistence,
                           bool(res.convergence_flag == 0))
    except Exception:
        return ewma_forecast(r, ann)


def forecast_universe(returns: pd.DataFrame, ann: float, use_garch: bool = True
                      ) -> pd.DataFrame:
    """Prevision de volatilidad t+1 para cada activo del universo."""
    rows = {}
    for col in returns.columns:
        fc = garch_forecast(returns[col], ann) if use_garch else ewma_forecast(returns[col], ann)
        rows[col] = {
            "sigma_ann": fc.sigma_ann,
            "model": fc.model,
            "persistence": fc.persistence,
            "converged": fc.converged,
        }
    return pd.DataFrame(rows).T


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Covarianza con shrinkage de Ledoit-Wolf.

    Con 8 activos y 180 observaciones la covarianza muestral ya es ruidosa; el
    optimizador de cartera amplifica ese ruido y produce pesos extremos. El
    shrinkage la contrae hacia una matriz estructurada y estabiliza los pesos.
    """
    from sklearn.covariance import LedoitWolf

    clean = returns.dropna()
    if len(clean) < len(returns.columns) + 5:
        return np.cov(returns.fillna(0.0).to_numpy(dtype=float), rowvar=False)
    return LedoitWolf().fit(clean.to_numpy(dtype=float)).covariance_


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    sd[sd <= 0] = 1e-12
    return cov / np.outer(sd, sd)
