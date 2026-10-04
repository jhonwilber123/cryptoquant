r"""Dimensionamiento de posiciones y control de riesgo.

Este modulo es el que de verdad determina el perfil de riesgo de la cartera.
La seleccion de activos decide en que se invierte; el sizing decide cuanto se
pierde cuando la seleccion se equivoca, que es lo que realmente importa.

La formula es deliberadamente multiplicativa pero con UNA sola escala:

    exposicion = (vol_objetivo / vol_prevista) x conviccion x drawdown x regimen
                 \_______ escala de riesgo _______/  \____ factores en [0,1] ____/

El objetivo de volatilidad fija el tamano; los demas factores solo pueden
recortarlo. Es importante que solo haya una escala: si dos capas distintas
recortan por el mismo riesgo, la exposicion colapsa y `target_volatility`
deja de significar lo que dice.

Capas:
  1. Objetivo de volatilidad -> ESCALA el capital invertido
  2. Conviccion (Kelly)      -> [0,1] segun cuanta ventaja estimada hay
  3. Filtro de regimen       -> [0,1] recorta en mercado bajista amplio
  4. Freno por drawdown      -> [0,1] desapalanca tras las perdidas
  5. Topes duros             -> limites absolutos que nada puede sobrepasar
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExposureDecision:
    gross_exposure: float       # exposicion final 0..max
    kelly_raw: float
    conviction: float           # kelly_raw normalizado a [0,1]
    vol_scalar: float
    drawdown_scalar: float
    regime_scalar: float
    reason: str                 # categoria estable, agregable en informes
    detail: str = ""            # cifras concretas de esa decision


def kelly_fraction(prob_win: float, payoff_ratio: float = 1.0) -> float:
    """Criterio de Kelly para apuesta binaria.

    f* = (p*b - q) / b, con b = ratio ganancia/perdida.

    Kelly pleno maximiza el crecimiento logaritmico a largo plazo, pero asume
    que `p` se conoce con exactitud. Como aqui `p` es una estimacion con error,
    Kelly pleno sobreapuesta sistematicamente. Por eso este valor no se usa
    como tamano de posicion, sino que pasa por `conviction_from_kelly`.

    Devuelve 0 cuando no hay ventaja: nunca un tamano negativo.
    """
    p = float(np.clip(prob_win, 0.0, 1.0))
    b = max(payoff_ratio, 1e-6)
    f = (p * b - (1 - p)) / b
    return float(np.clip(f, 0.0, 1.0))


def conviction_from_kelly(kelly_raw: float, full_size_level: float) -> float:
    """Traduce Kelly bruto en un factor de conviccion en [0, 1].

    Por que no se multiplica Kelly directamente por la exposicion: el objetivo
    de volatilidad YA es un recorte de riesgo (si el mercado esta al 60% y el
    objetivo es 15%, autoriza el 25% del capital). Multiplicar ademas por un
    Kelly fraccionario de ~0.05 compensa el mismo riesgo dos veces y deja la
    exposicion en el 1%, con lo que la volatilidad realizada no se parece en
    nada a la pedida y el parametro `target_volatility` deja de significar algo.

    Aqui Kelly hace lo que sabe hacer -- decir CUANTA ventaja hay -- y el
    presupuesto de riesgo decide la ESCALA. `full_size_level` es el nivel de
    Kelly bruto a partir del cual se despliega el presupuesto completo:
    subirlo es mas exigente, bajarlo mas agresivo.
    """
    if full_size_level <= 0:
        return 0.0
    return float(np.clip(kelly_raw / full_size_level, 0.0, 1.0))


def volatility_scalar(forecast_vol: float, target_vol: float, cap: float = 3.0) -> float:
    """Escalar de volatilidad objetivo.

    Si la cartera prevé un 40% de vol y el objetivo es 15%, se invierte el
    37.5% del capital. Es el mecanismo mas efectivo para reducir el riesgo de
    cripto sin renunciar a estar expuesto: la vol futura es mucho mas predecible
    que la direccion futura.
    """
    if not np.isfinite(forecast_vol) or forecast_vol <= 0:
        return 0.0
    return float(np.clip(target_vol / forecast_vol, 0.0, cap))


def drawdown_scalar(current_dd: float, derisk_start: float, stop_level: float) -> float:
    """Reduccion lineal de exposicion en funcion del drawdown vigente.

    Entre `derisk_start` y `stop_level` la exposicion decrece linealmente hasta
    cero. Evita el escenario clasico de ruina: mantener el tamano completo
    mientras la cartera se desangra.
    """
    dd = abs(min(current_dd, 0.0))
    if dd <= derisk_start:
        return 1.0
    if dd >= stop_level:
        return 0.0
    span = stop_level - derisk_start
    return float(1.0 - (dd - derisk_start) / span) if span > 0 else 0.0


def decide_exposure(
    prob_win: float,
    payoff_ratio: float,
    forecast_vol: float,
    current_drawdown: float,
    regime_ok: bool,
    risk_cfg,
) -> ExposureDecision:
    """Combina las capas de control en una unica exposicion bruta.

        exposicion = escala_de_riesgo x conviccion x drawdown x regimen

    La escala la fija el objetivo de volatilidad; el resto son factores en
    [0, 1] que solo pueden reducirla. Asi `target_volatility` conserva su
    significado: es el riesgo que se asume cuando todo lo demas esta en orden.
    """
    k_raw = kelly_fraction(prob_win, payoff_ratio)
    k = conviction_from_kelly(k_raw, risk_cfg.kelly_full_size_level)
    v = volatility_scalar(forecast_vol, risk_cfg.target_volatility)
    d = drawdown_scalar(current_drawdown, risk_cfg.drawdown_derisk_start,
                        risk_cfg.max_drawdown_stop)
    # El regimen no apaga del todo la exposicion: la reduce fuerte. Apagarla
    # por completo genera whipsaw cuando el precio oscila en torno a la media.
    r = 1.0 if regime_ok else 0.25

    gross = min(v * k * d * r, risk_cfg.max_gross_exposure)
    gross = float(max(gross, 0.0))

    # La categoria se mantiene fija para que el informe pueda agregarla; las
    # cifras variables van en `detail`, nunca en `reason`.
    if d == 0.0:
        reason = "cortacircuito por drawdown maximo"
        detail = f"drawdown {current_drawdown:.1%}"
    elif d < 1.0:
        reason = "desapalancado por drawdown"
        detail = f"drawdown {current_drawdown:.1%}, escalar {d:.2f}"
    elif k_raw <= 0:
        reason = "sin ventaja estadistica (Kelly <= 0)"
        detail = f"p={prob_win:.3f}, payoff={payoff_ratio:.2f}"
    elif k < 0.5:
        reason = "conviccion baja"
        detail = f"Kelly bruto {k_raw:.3f} sobre nivel {risk_cfg.kelly_full_size_level:.2f}"
    elif not regime_ok:
        reason = "regimen bajista: exposicion reducida"
        detail = f"escalar de regimen {r:.2f}"
    elif v < 0.5:
        reason = "vol prevista muy por encima del objetivo"
        detail = f"prevista {forecast_vol:.1%} vs objetivo {risk_cfg.target_volatility:.1%}"
    elif v < 1.0:
        reason = "vol prevista por encima del objetivo"
        detail = f"prevista {forecast_vol:.1%} vs objetivo {risk_cfg.target_volatility:.1%}"
    else:
        reason = "condiciones normales"
        detail = f"vol prevista {forecast_vol:.1%}"

    return ExposureDecision(gross, k_raw, k, v, d, r, reason, detail)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray, ann: float) -> float:
    """Volatilidad anualizada de la cartera dada la matriz de covarianzas."""
    var = float(weights @ cov @ weights)
    return float(np.sqrt(max(var, 0.0) * ann))


def apply_turnover_filter(target: pd.Series, current: pd.Series, threshold: float
                          ) -> pd.Series:
    """Evita rebalanceos irrelevantes.

    Cada operacion cuesta comision y slippage. Si el peso objetivo difiere del
    actual en menos del umbral, se mantiene la posicion: en backtests reales
    este filtro suele valer mas rendimiento neto que cualquier mejora del
    modelo predictivo.
    """
    cur = current.reindex(target.index).fillna(0.0)
    delta = (target - cur).abs()
    return target.where(delta >= threshold, cur)
