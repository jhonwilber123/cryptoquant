"""Backtest del VaR: la medida de riesgo tambien hay que ponerla a prueba.

Un VaR al 95% promete que la perdida solo lo superara el 5% de los dias.
Tres preguntas lo ponen a prueba:

    Kupiec (1995)          Se supera el 5% de los dias, ni mas ni menos?
                           Mas: subestima el riesgo. Menos: lo exagera y
                           deja capital ocioso.
    Christoffersen (1998)  Las excepciones llegan sueltas o en rachas? En
                           rachas significa que el VaR no reacciona a tiempo
                           cuando el mercado cambia de regimen: justo cuando
                           mas importa.
    Cola (CVaR)            Cuando se supera, cuanto se pierde de media frente
                           a lo que prometia el CVaR? Un cociente > 1 indica
                           colas peores de lo previsto.

Un metodo se acepta si pasa Kupiec y la cobertura condicional al 5%, y solo
si hay al menos `MIN_OBS` dias con que juzgarlo: con 30 dias al 95% se
esperan 1,5 excepciones, y "no se rechaza" no significa nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .var import METODOS

MIN_OBS = 250  # un ano de dias invertidos, como el backtesting de Basilea


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(excepciones: np.ndarray, alpha: float) -> dict:
    """Razon de verosimilitud de cobertura incondicional (chi2, 1 g.l.)."""
    e = np.asarray(excepciones, dtype=bool)
    n, x, p = len(e), int(e.sum()), 1 - alpha
    ph = x / n if n else np.nan
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - ph) - _xlogy(x, ph))
    return {"n": n, "excepciones": x, "tasa": ph, "esperada": p,
            "lr_kupiec": lr, "p_kupiec": float(stats.chi2.sf(lr, 1))}


def christoffersen(excepciones: np.ndarray) -> dict:
    """Independencia de las excepciones: cadena de Markov de primer orden."""
    e = np.asarray(excepciones, dtype=int)
    a, b = e[:-1], e[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    ll_h0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    ll_h1 = _xlogy(n00, 1 - p0) + _xlogy(n01, p0) + _xlogy(n10, 1 - p1) + _xlogy(n11, p1)
    lr = max(0.0, -2 * (ll_h0 - ll_h1))
    return {"prob_tras_excepcion": p1, "prob_tras_normal": p0,
            "lr_independencia": lr, "p_independencia": float(stats.chi2.sf(lr, 1))}


def prueba_cola(perdidas: np.ndarray, es: np.ndarray) -> dict:
    """Perdida media en las excepciones frente al CVaR que se prometia.

    Contraste t de que el cociente perdida/CVaR vale 1 (en el espiritu de
    McNeil y Frey, 2000). Con pocas excepciones tiene poca potencia.
    """
    ratio = perdidas / es
    n = len(ratio)
    p = float(stats.ttest_1samp(ratio, 1.0).pvalue) if n >= 3 else np.nan
    return {"cola_ratio": float(ratio.mean()) if n else np.nan, "p_cola": p}


def evaluar(tabla: pd.DataFrame, alpha: float, metodos=METODOS,
            exposicion_minima: float = 0.01, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Backtest de cada metodo sobre una tabla de `var.pronosticar`.

    Solo cuentan los dias con exposicion: en efectivo el VaR es 0 y la unica
    "perdida" posible son costes de transaccion, que no son riesgo de
    mercado.
    """
    t = tabla[tabla["exposicion"] >= exposicion_minima]
    perdida = -t["retorno"].to_numpy()
    filas = {}
    for m in metodos:
        var, es = t[f"var_{m}"].to_numpy(), t[f"es_{m}"].to_numpy()
        exc = perdida > var
        k, c = kupiec(exc, alpha), christoffersen(exc)
        lr_cc = k["lr_kupiec"] + c["lr_independencia"]
        fila = {**k, **c, "p_cobertura_condicional": float(stats.chi2.sf(lr_cc, 2)),
                **prueba_cola(perdida[exc], es[exc]),
                "var_medio": float(var.mean()) if len(var) else np.nan}
        fila["suficiente"] = bool(fila["n"] >= min_obs)
        fila["aceptado"] = bool(fila["suficiente"] and fila["p_kupiec"] > 0.05
                                and fila["p_cobertura_condicional"] > 0.05)
        filas[m] = fila
    return pd.DataFrame(filas).T
