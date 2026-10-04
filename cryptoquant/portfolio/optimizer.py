"""Construccion de cartera.

Markowitz clasico funciona mal en la practica porque los retornos esperados se
estiman con enorme error y el optimizador los amplifica: pequenos cambios en
los inputs producen carteras radicalmente distintas. Por eso el metodo por
defecto aqui es HRP, que no necesita invertir la matriz de covarianzas ni
estimar retornos esperados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from ..econometrics.volatility import correlation_from_cov


def _quasi_diagonal_order(link: np.ndarray, n: int) -> list[int]:
    """Reordena los activos siguiendo el arbol jerarquico, de forma que los
    correlacionados queden adyacentes."""
    link = link.astype(int)
    order = pd.Series([link[-1, 0], link[-1, 1]])
    while order.max() >= n:
        order.index = range(0, order.shape[0] * 2, 2)
        frame = order[order >= n]
        i, j = frame.index, frame.to_numpy() - n
        order[i] = link[j, 0]
        order = pd.concat([order, pd.Series(link[j, 1], index=i + 1)]).sort_index()
        order.index = range(order.shape[0])
    return order.tolist()


def _inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def hierarchical_risk_parity(cov: np.ndarray, labels: list[str]) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Agrupa los activos por correlacion y reparte el riesgo recursivamente entre
    ramas del arbol. Al no invertir la matriz de covarianzas, es estable cuando
    la matriz esta mal condicionada, que es lo habitual en cripto porque casi
    todo cotiza contra BTC y las correlaciones se disparan en los crashes.
    """
    n = len(labels)
    if n == 1:
        return pd.Series([1.0], index=labels)

    corr = correlation_from_cov(cov)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    # Distancia de correlacion de Mantegna.
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")

    order = _quasi_diagonal_order(link, n)
    weights = pd.Series(1.0, index=order, dtype=float)
    clusters = [order]

    while clusters:
        # Biseccion de cada cluster en dos mitades.
        clusters = [
            half
            for c in clusters
            for half in (c[: len(c) // 2], c[len(c) // 2 :])
            if len(c) > 1
        ]
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            v_left = _cluster_variance(cov, left)
            v_right = _cluster_variance(cov, right)
            total = v_left + v_right
            alpha = 1.0 - v_left / total if total > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1 - alpha

    out = pd.Series(weights.to_numpy(), index=[labels[i] for i in weights.index])
    return out.reindex(labels).fillna(0.0)


def risk_parity(cov: np.ndarray, labels: list[str]) -> pd.Series:
    """Paridad de riesgo estricta: cada activo aporta la misma contribucion al
    riesgo total. Se resuelve numericamente."""
    n = len(labels)
    target = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol <= 0:
            return 1e6
        contrib = w * (cov @ w) / port_vol
        return float(np.sum((contrib / contrib.sum() - target) ** 2))

    res = minimize(
        objective,
        target,
        method="SLSQP",
        bounds=[(1e-4, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = res.x if res.success else target
    return pd.Series(w / w.sum(), index=labels)


def min_variance(cov: np.ndarray, labels: list[str], max_weight: float = 0.35) -> pd.Series:
    n = len(labels)
    x0 = np.full(n, 1.0 / n)
    res = minimize(
        lambda w: float(w @ cov @ w),
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500},
    )
    w = res.x if res.success else x0
    return pd.Series(np.clip(w, 0, None) / np.clip(w, 0, None).sum(), index=labels)


def apply_constraints(weights: pd.Series, max_weight: float) -> pd.Series:
    """Aplica el tope por activo redistribuyendo el exceso iterativamente.

    Estos son pesos RELATIVOS: suman 1 y luego se escalan por la exposicion
    bruta. Si hay menos de 1/max_weight activos el tope es infactible en
    terminos relativos (3 activos no pueden sumar 1 con tope 0.30), y la
    funcion devuelve el reparto mas uniforme posible.

    Eso no rompe el limite de concentracion real, porque el motor vuelve a
    aplicar el tope sobre el peso ABSOLUTO (peso relativo x exposicion bruta),
    que es la magnitud a la que se refiere `max_weight_per_asset`: no mas de
    ese porcentaje del capital total en un solo activo.
    """
    w = weights.clip(lower=0.0).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    if len(w) * max_weight <= 1.0 + 1e-9:
        return pd.Series(1.0 / len(w), index=w.index)
    for _ in range(50):
        excess = (w - max_weight).clip(lower=0.0)
        if excess.sum() <= 1e-9:
            break
        w = w - excess
        room = (max_weight - w).clip(lower=0.0)
        if room.sum() <= 1e-9:
            break
        w = w + excess.sum() * room / room.sum()
    return w / w.sum() if w.sum() > 0 else w


def build_weights(cov: np.ndarray, labels: list[str], method: str = "hrp",
                  max_weight: float = 0.30) -> pd.Series:
    if method == "hrp":
        w = hierarchical_risk_parity(cov, labels)
    elif method == "risk_parity":
        w = risk_parity(cov, labels)
    elif method == "min_variance":
        w = min_variance(cov, labels, max_weight)
    elif method == "equal":
        w = pd.Series(1.0 / len(labels), index=labels)
    else:
        raise ValueError(f"metodo de cartera desconocido: {method}")
    return apply_constraints(w, max_weight)
