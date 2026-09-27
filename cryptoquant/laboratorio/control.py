"""Control de volatilidad: lo que si funciona, en un solo activo.

Es la idea central de la tesis reducida a una linea. La direccion del precio
apenas se puede prever (pasos 5 y 7 del laboratorio); su volatilidad si
(paso 6). Asi que, en lugar de apostar por la direccion, se ajusta cuanto
capital se expone para que el riesgo quede cerca de un objetivo:

    exposicion(t) = min(1, objetivo / volatilidad_prevista(t))

Con volatilidad prevista alta se invierte menos; con baja, mas; el resto
queda en efectivo. No predice nada del precio.

La prevision es EWMA (RiskMetrics, lambda = 0.94): una linea de codigo,
estrictamente causal y sin estimar parametros. El sistema principal usa un
GJR-GARCH; la idea es la misma.

La comparacion justa no es contra comprar y mantener, sino contra comprar y
mantener con la MISMA volatilidad (una parte fija en el activo y el resto en
efectivo): cualquiera reduce el riesgo invirtiendo menos. Esa fraccion se
calcula con la volatilidad de toda la muestra, asi que es una referencia ex
post, no una estrategia que se pudiera haber seguido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sigma_ewma(ret: pd.Series, lam: float = 0.94, arranque: int = 30) -> pd.Series:
    """Desviacion diaria prevista para el dia SIGUIENTE, con datos hasta hoy."""
    if not 0 < lam < 1:
        raise ValueError("lambda debe estar entre 0 y 1")
    r = ret.dropna()
    if len(r) <= arranque:
        raise ValueError(f"hacen falta mas de {arranque} retornos")
    x = r.to_numpy(dtype=float)
    s2 = np.full(len(x), np.nan)
    v = float(np.var(x[:arranque]))
    for i in range(len(x)):
        v = lam * v + (1 - lam) * x[i] ** 2
        s2[i] = v
    s2[:arranque] = np.nan  # el arranque no es una prevision: solo inicializa
    return pd.Series(np.sqrt(s2), index=r.index)


def control_volatilidad(ret: pd.Series, objetivo_anual: float = 0.15, anual: float = 365.0,
                        lam: float = 0.94, max_exposicion: float = 1.0,
                        coste: float = 0.0015) -> pd.DataFrame:
    """Retornos diarios del control de volatilidad y de sus dos referencias.

    Args:
        ret: retornos logaritmicos diarios del activo.
        coste: comision + deslizamiento por unidad de exposicion negociada
            (0.15%, como el backtest del sistema).

    La exposicion decidida al cierre de t se aplica al retorno de t+1, y el
    coste se cobra sobre lo que cambia: nada usa informacion futura.
    """
    if not 0 < objetivo_anual < 5:
        raise ValueError("el objetivo de volatilidad anual debe estar entre 0 y 5 (p. ej. 0.15)")
    if not 0 < max_exposicion <= 1:
        raise ValueError("la exposicion maxima debe estar entre 0 y 1 (sin apalancamiento)")
    sigma_anual = sigma_ewma(ret, lam) * np.sqrt(anual)
    expo = (objetivo_anual / sigma_anual).clip(upper=max_exposicion).dropna()
    simple = np.expm1(ret.dropna())
    usada = expo.shift(1).reindex(simple.index)  # decidida ayer, aplicada hoy
    giro = usada.diff().abs().fillna(usada.abs())
    t = pd.DataFrame({"exposicion": usada, "pasiva": simple})
    t["estrategia"] = usada * simple - coste * giro
    t = t.dropna()
    frac = float(t["estrategia"].std() / t["pasiva"].std())
    t["pasiva_igual_vol"] = frac * t["pasiva"]
    t.attrs["fraccion_igual_vol"] = frac
    return t


def resumen(t: pd.DataFrame, anual: float = 365.0, capital: float = 10_000.0) -> pd.DataFrame:
    """Rentabilidad anual, volatilidad, caida maxima, Sharpe y capital final."""
    filas = {}
    for col in ("estrategia", "pasiva", "pasiva_igual_vol"):
        r = t[col]
        eq = (1 + r).cumprod()
        anos = len(r) / anual
        filas[col] = {
            "rentabilidad_anual": float(eq.iloc[-1] ** (1 / anos) - 1),
            "volatilidad": float(r.std() * np.sqrt(anual)),
            "caida_maxima": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(anual)) if r.std() > 0 else np.nan,
            "capital_final": float(capital * eq.iloc[-1]),
        }
    out = pd.DataFrame(filas).T
    out.attrs["exposicion_media"] = float(t["exposicion"].mean())
    return out
