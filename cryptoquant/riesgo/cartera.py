"""El riesgo de una cartera concreta: la que usted tiene hoy.

Entrada: unidades de cada activo (y efectivo en USDT). Salida, en dinero:

- cuanto puede perder manana en un mal dia (VaR y CVaR, por cuatro metodos),
  y si ese metodo habria acertado en el pasado con esta misma cartera;
- que rango de resultados cabe esperar en el proximo mes (Monte Carlo);
- que activo aporta cuanto riesgo (no es lo mismo que cuanto dinero);
- para cada activo, la sigma prevista por GARCH y el stop y tamano de una
  entrada nueva que arriesgue una fraccion fija del total.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..econometrics.risk import marginal_risk_contribution
from ..econometrics.volatility import ledoit_wolf_cov
from ..laboratorio import volatilidad as lab_vol
from ..laboratorio.datos import validar_ticker
from . import futuro, pruebas, var

EFECTIVO = {"USDT", "USDC", "USD", "FDUSD", "DAI"}
MIN_DIAS = 120  # historia comun minima para que un VaR de un dia signifique algo


def parsear(texto: str) -> dict[str, float]:
    """'BTC=0.05, ETH=1.2, USDT=500' -> {'BTC': 0.05, 'ETH': 1.2, 'USDT': 500.0}"""
    out: dict[str, float] = {}
    for parte in filter(None, (p.strip() for p in texto.replace(";", ",").split(","))):
        if "=" not in parte:
            raise ValueError(f"'{parte}': use el formato ACTIVO=cantidad, p. ej. BTC=0.05")
        k, v = parte.split("=", 1)
        activo = validar_ticker(k)
        try:
            cantidad = float(v)
        except ValueError:
            raise ValueError(f"{activo}: {v.strip()!r} no es una cantidad") from None
        if not math.isfinite(cantidad):
            raise ValueError(f"{activo}: la cantidad debe ser un numero finito")
        if cantidad < 0:
            raise ValueError(f"{activo}: cantidad negativa; solo se admiten posiciones largas")
        out[activo] = out.get(activo, 0.0) + cantidad
    if not out:
        raise ValueError("cartera vacia")
    return out


@dataclass
class RiesgoCartera:
    fecha: pd.Timestamp
    total: float
    posiciones: pd.DataFrame      # unidades, precio, valor, peso, contribucion al riesgo
    var_manana: pd.DataFrame      # por metodo: VaR y CVaR en % y en dinero
    validacion: pd.DataFrame      # backtest de cada metodo con esta cartera
    mes: dict                     # Monte Carlo a `horizonte` dias
    por_activo: pd.DataFrame      # sigma GARCH, VaR del activo, stop y tamano
    alpha: float
    horizonte: int


def analizar(tenencias: dict[str, float], closes: pd.DataFrame, alpha: float = 0.95,
             horizonte: int = 21, riesgo_por_operacion: float = 0.01,
             n_sim: int = 10_000, semilla: int = 42) -> RiesgoCartera:
    """`closes`: cierres diarios de, al menos, los activos no monetarios."""
    if not 0.5 < alpha < 1:
        raise ValueError("la confianza debe estar entre 0.5 y 1 (p. ej. 0.95)")
    if horizonte < 1 or n_sim < 1:
        raise ValueError("el horizonte y el numero de simulaciones deben ser >= 1")
    efectivo = sum(v for k, v in tenencias.items() if k in EFECTIVO)
    activos = [k for k, v in tenencias.items() if k not in EFECTIVO and v > 0]
    if not activos:
        raise ValueError("la cartera solo tiene efectivo: no hay riesgo de mercado que medir")
    faltan = [a for a in activos if a not in closes.columns]
    if faltan:
        raise ValueError(f"sin precios para: {', '.join(faltan)}")

    px = closes[activos].dropna()
    if len(px) < MIN_DIAS:
        raise ValueError(f"solo {len(px)} dias de historia comun; hacen falta {MIN_DIAS}")
    precio = px.iloc[-1]
    unidades = pd.Series({a: tenencias[a] for a in activos})
    valor = unidades * precio
    total = float(valor.sum() + efectivo)
    if total <= 0:
        raise ValueError("la cartera no tiene valor")
    w = valor / total

    R = px.pct_change().dropna()
    cov = ledoit_wolf_cov(R.iloc[-500:]) if len(activos) > 1 else np.array([[R.iloc[-500:, 0].var()]])
    contrib = marginal_risk_contribution(w.to_numpy(), cov)
    contrib = contrib / contrib.sum() if contrib.sum() > 0 else contrib

    posiciones = pd.DataFrame({"unidades": unidades, "precio": precio, "valor": valor,
                               "peso": w, "contribucion_riesgo": contrib})
    if efectivo:
        posiciones.loc["efectivo"] = [efectivo, 1.0, efectivo, efectivo / total, 0.0]

    # --- VaR de manana, y su historial con esta misma cartera ---------------
    manana = var.var_siguiente(R, w, alpha)
    var_manana = pd.DataFrame({"var_pct": manana["var"], "cvar_pct": manana["es"],
                               "var": manana["var"] * total, "cvar": manana["es"] * total})
    t = var.pronosticar(R, w, alpha=alpha, minimo=min(250, len(R) // 2))
    validacion = pruebas.evaluar(t, alpha)

    # --- Proximo mes: remuestreo conjunto (conserva las correlaciones) -------
    rng = np.random.default_rng(semilla)
    idx = futuro.indices_bloques(len(R), horizonte, n_sim, 10, rng)
    crec = np.cumprod(1 + R.to_numpy()[idx], axis=1)          # (n, h, activos)
    caminos = (crec @ valor.to_numpy() + efectivo) / total    # unidades fijas
    mes = futuro.resumen_caminos(np.hstack([np.ones((n_sim, 1)), caminos]))
    mes = {**mes, **{f"{k}_dinero": v * total for k, v in mes.items()
                     if k.startswith(("retorno", "cvar"))}}

    # --- Por activo --------------------------------------------------------
    filas = {}
    for a in activos:
        lr = np.log(px[a]).diff().dropna()
        g = lab_vol.pronostico_garch(lr)
        plan = lab_vol.plan_operacion(float(precio[a]), g["sigma_manana"], total,
                                      riesgo_por_operacion)
        v_a, _ = var.var_es(R[a].iloc[-500:].to_numpy(), "t", alpha)
        filas[a] = {"sigma_manana": g["sigma_manana"], "sigma_anual": g["sigma_manana_anual"],
                    "var_1d_pct": v_a, "var_1d_posicion": v_a * float(valor[a]),
                    "stop": plan["stop"], "distancia_stop_pct": plan["distancia_stop_pct"],
                    "tamano_entrada": plan["nominal"]}
    return RiesgoCartera(R.index[-1], total, posiciones, var_manana, validacion, mes,
                         pd.DataFrame(filas).T, alpha, horizonte)
