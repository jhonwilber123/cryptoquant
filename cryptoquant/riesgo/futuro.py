"""Riesgo a futuro: que puede pasar en los proximos N dias.

La caida maxima del backtest es un unico numero: la peor que ocurrio en la
unica historia que hubo. Remuestreando bloques de esa historia se obtiene la
distribucion de caidas que las mismas reglas podrian haber producido, y con
ella probabilidades: cuantas veces de cada cien se cae un 20% en un ano.

Limitacion honesta: todo sale de la historia observada (2021-2026). Un
escenario peor que cualquier bloque de esa historia no aparece nunca.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..laboratorio.montecarlo import trayectorias_precio


def indices_bloques(n_hist: int, dias: int, n: int, bloque: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Matriz (n, dias) de indices de filas historicas, por bloques contiguos."""
    if n_hist < 2 * bloque or dias < 1 or n < 1:
        raise ValueError(f"se necesitan al menos {2 * bloque} dias de historia (hay {n_hist})")
    n_bloques = -(-dias // bloque)
    inicios = rng.integers(0, n_hist - bloque + 1, size=(n, n_bloques))
    return (inicios[:, :, None] + np.arange(bloque)).reshape(n, -1)[:, :dias]


def resumen_caminos(caminos: np.ndarray, caidas=(0.10, 0.20, 0.50)) -> dict:
    """Caminos de capital que arrancan en 1 -> probabilidades y cuantiles."""
    final = caminos[:, -1] - 1
    dd = (caminos / np.maximum.accumulate(caminos, axis=1) - 1).min(axis=1)
    peor5 = np.percentile(final, 5)
    return {
        "retorno_mediano": float(np.median(final)),
        "retorno_p5": float(peor5),
        "cvar_5": float(final[final <= peor5].mean()),
        "prob_perdida": float((final < 0).mean()),
        "caida_mediana": float(np.median(dd)),
        "caida_p95": float(np.percentile(dd, 5)),
        **{f"prob_caida_{int(c * 100)}": float((dd <= -c).mean()) for c in caidas},
    }


def riesgo_futuro(series: dict[str, pd.Series], horizontes=(21, 63, 252), n: int = 10_000,
                  bloque: int = 10, semilla: int = 42) -> pd.DataFrame:
    """Una fila por (serie, horizonte) a partir de retornos simples diarios."""
    filas = []
    for nombre, r in series.items():
        lr = np.log1p(r.dropna())
        for h in horizontes:
            caminos = trayectorias_precio(lr, 1.0, dias=h, n=n, bloque=bloque, semilla=semilla)
            filas.append({"serie": nombre, "horizonte_dias": h, **resumen_caminos(caminos)})
    return pd.DataFrame(filas)
