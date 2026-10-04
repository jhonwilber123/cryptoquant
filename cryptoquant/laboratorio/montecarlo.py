"""Monte Carlo: trayectorias de precio y simulacion de la cuenta.

Un backtest es una sola trayectoria: la que ocurrio. Monte Carlo pregunta
que otras podrian haber ocurrido con las mismas reglas, y con que
probabilidad. La respuesta casi siempre es mas incomoda que el backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trayectorias_precio(ret: pd.Series, precio_inicial: float, dias: int = 90,
                        n: int = 5000, bloque: int = 5, semilla: int = 42) -> np.ndarray:
    """Remuestreo por bloques de retornos historicos -> matriz (n, dias + 1).

    Se remuestrean bloques de `bloque` dias consecutivos, no dias sueltos:
    asi se conserva parte del agrupamiento de la volatilidad. Con bloque=1 las
    rachas malas se diluyen y las colas de la simulacion salen demasiado
    finas.
    """
    r = ret.dropna().to_numpy(dtype=float)
    if len(r) < 2 * bloque or dias < 1 or n < 1:
        raise ValueError(f"se necesitan al menos {2 * bloque} retornos (hay {len(r)}), "
                         "dias >= 1 y n >= 1")
    rng = np.random.default_rng(semilla)
    n_bloques = -(-dias // bloque)
    inicios = rng.integers(0, len(r) - bloque + 1, size=(n, n_bloques))
    idx = (inicios[:, :, None] + np.arange(bloque)).reshape(n, -1)[:, :dias]
    caminos = np.cumsum(r[idx], axis=1)
    return precio_inicial * np.exp(np.hstack([np.zeros((n, 1)), caminos]))


def resumen_trayectorias(caminos: np.ndarray, caida: float = 0.20) -> dict:
    p0, final = caminos[:, 0], caminos[:, -1]
    minimo = caminos.min(axis=1)
    q = np.percentile(final, [5, 25, 50, 75, 95])
    return {
        **{f"precio_p{k}": v for k, v in zip((5, 25, 50, 75, 95), q)},
        "prob_termina_abajo": float((final < p0).mean()),
        f"prob_toca_menos_{int(caida * 100)}": float((minimo <= p0 * (1 - caida)).mean()),
    }


def simular_cuenta(acierto: float, riesgo_beneficio: float, riesgo: float,
                   n_operaciones: int = 200, n_sim: int = 10_000, capital: float = 10_000.0,
                   semilla: int = 42) -> np.ndarray:
    """Curvas de capital con riesgo fijo por operacion -> matriz (n_sim, n_op + 1).

    Cada operacion gana `riesgo * riesgo_beneficio` o pierde `riesgo` del
    capital de ese momento (fraccion fija: tras perder, se arriesga menos).
    """
    if not 0 <= acierto <= 1:
        raise ValueError("el acierto es una probabilidad: entre 0 y 1")
    if not riesgo_beneficio > 0:
        raise ValueError("la relacion riesgo-beneficio debe ser positiva")
    if not 0 < riesgo < 1:
        raise ValueError("el riesgo por operacion debe estar entre 0 y 1 (sin incluir 1: "
                         "arriesgarlo todo en cada operacion arruina la cuenta a la primera)")
    rng = np.random.default_rng(semilla)
    gana = rng.random((n_sim, n_operaciones)) < acierto
    factor = np.where(gana, 1 + riesgo * riesgo_beneficio, 1 - riesgo)
    return capital * np.hstack([np.ones((n_sim, 1)), np.cumprod(factor, axis=1)])


def _racha_perdedora(gana: np.ndarray) -> np.ndarray:
    racha = np.zeros(gana.shape[0], dtype=int)
    actual = np.zeros_like(racha)
    for col in (~gana).T:
        actual = np.where(col, actual + 1, 0)
        racha = np.maximum(racha, actual)
    return racha


def resumen_cuenta(curvas: np.ndarray, acierto: float, riesgo_beneficio: float,
                   riesgo: float) -> dict:
    capital = curvas[:, 0]
    final = curvas[:, -1]
    pico = np.maximum.accumulate(curvas, axis=1)
    dd = (curvas / pico - 1).min(axis=1)
    gana = np.diff(curvas, axis=1) > 0
    esperanza_r = acierto * riesgo_beneficio - (1 - acierto)
    kelly = acierto - (1 - acierto) / riesgo_beneficio
    return {
        "esperanza_por_operacion_R": esperanza_r,
        "acierto_minimo_rentable": 1 / (1 + riesgo_beneficio),
        "kelly_completo": kelly,
        "riesgo_sobre_kelly": riesgo / kelly if kelly > 0 else np.inf,
        "capital_mediano_final": float(np.median(final)),
        "capital_p5_final": float(np.percentile(final, 5)),
        "capital_p95_final": float(np.percentile(final, 95)),
        "prob_perdida": float((final < capital).mean()),
        "drawdown_mediano": float(np.median(dd)),
        "drawdown_p95": float(np.percentile(dd, 5)),  # el 5% peor
        "prob_drawdown_20": float((dd <= -0.20).mean()),
        "prob_drawdown_50": float((dd <= -0.50).mean()),
        "racha_perdedora_mediana": float(np.median(_racha_perdedora(gana))),
    }
