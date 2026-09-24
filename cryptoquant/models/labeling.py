"""Etiquetado de la variable objetivo.

El error mas comun al aplicar ML a mercados es etiquetar con el retorno a N
barras fijas. Eso ignora que una posicion real tiene stop loss y toma de
beneficios: el retorno a 10 dias es irrelevante si el stop salto el dia 2.

El metodo de triple barrera (Lopez de Prado) etiqueta segun cual de tres
barreras se toca primero: beneficio, perdida o vencimiento. Las barreras se
escalan con la volatilidad, de modo que la etiqueta significa lo mismo en un
mercado tranquilo y en uno agitado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    horizon: int = 10,
    profit_mult: float = 2.0,
    stop_mult: float = 1.5,
) -> pd.DataFrame:
    """Etiqueta cada barra segun la primera barrera tocada.

    Args:
        close: serie de precios de cierre.
        volatility: volatilidad diaria (no anualizada) estimada de forma causal.
        horizon: barrera vertical, en barras.
        profit_mult / stop_mult: barreras horizontales en multiplos de sigma.

    Returns:
        DataFrame con `label` (1 beneficio, -1 stop, 0 vencimiento),
        `ret` (retorno realizado hasta la salida) y `exit_bar`.
    """
    px = close.to_numpy(dtype=float)
    sigma = volatility.to_numpy(dtype=float)
    n = len(px)
    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    exits = np.full(n, np.nan)

    for i in range(n - 1):
        s = sigma[i]
        if not np.isfinite(s) or s <= 0:
            continue
        upper = px[i] * (1 + profit_mult * s)
        lower = px[i] * (1 - stop_mult * s)
        end = min(i + horizon, n - 1)
        path = px[i + 1 : end + 1]
        if len(path) == 0:
            continue

        hit_up = np.argmax(path >= upper) if (path >= upper).any() else np.inf
        hit_dn = np.argmax(path <= lower) if (path <= lower).any() else np.inf

        if hit_up < hit_dn:
            k = int(hit_up)
            labels[i], rets[i] = 1.0, path[k] / px[i] - 1.0
        elif hit_dn < hit_up:
            k = int(hit_dn)
            labels[i], rets[i] = -1.0, path[k] / px[i] - 1.0
        else:
            k = len(path) - 1
            labels[i], rets[i] = 0.0, path[k] / px[i] - 1.0
        exits[i] = i + 1 + k

    return pd.DataFrame(
        {"label": labels, "ret": rets, "exit_bar": exits}, index=close.index
    )


def binary_target(labels: pd.DataFrame) -> pd.Series:
    """Convierte la etiqueta de 3 clases en binaria: 1 si el trade fue rentable.

    Se modela P(retorno > 0) en lugar de la magnitud porque la probabilidad es
    lo que alimenta directamente el dimensionamiento por Kelly.
    """
    return (labels["ret"] > 0).astype(int).where(labels["ret"].notna())


def sample_weights_by_uniqueness(labels: pd.DataFrame) -> pd.Series:
    """Pondera cada muestra por su unicidad temporal.

    Muestras cuyos horizontes se solapan comparten informacion: contarlas como
    independientes infla artificialmente el tamano efectivo de la muestra y
    hace que el modelo parezca mas fiable de lo que es. Se pondera cada muestra
    por el inverso del numero de observaciones concurrentes.
    """
    n = len(labels)
    counts = np.zeros(n)
    starts = np.arange(n)
    ends = labels["exit_bar"].to_numpy(dtype=float)
    for i in range(n):
        e = ends[i]
        if not np.isfinite(e):
            continue
        counts[starts[i] : int(e) + 1] += 1

    weights = np.full(n, np.nan)
    for i in range(n):
        e = ends[i]
        if not np.isfinite(e):
            continue
        window = counts[starts[i] : int(e) + 1]
        window = window[window > 0]
        weights[i] = float(np.mean(1.0 / window)) if len(window) else np.nan

    w = pd.Series(weights, index=labels.index)
    return w / w.mean() if w.notna().any() else w
