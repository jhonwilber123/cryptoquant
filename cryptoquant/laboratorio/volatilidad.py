"""Volatilidad: agrupamiento, GARCH(1,1) y su uso en stop y tamano.

El sistema principal usa un GJR-GARCH con t asimetrica
(`econometrics/volatility.py`). Aqui se usa el GARCH(1,1) basico a proposito:
tres parametros que se leen a mano,

    sigma2[t+1] = omega + alpha * r[t]^2 + beta * sigma2[t]

alpha: cuanto pesa el shock de hoy.  beta: cuanto dura la volatilidad de
ayer.  alpha + beta: persistencia; cerca de 1, los sobresaltos tardan
semanas en disiparse.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch


def agrupamiento(ret: pd.Series) -> dict:
    """ARCH-LM (H0: varianza constante) y ACF de |r| a varios rezagos."""
    r = ret.dropna()
    a = r.abs()
    return {
        "arch_lm_p": float(het_arch(r - r.mean(), nlags=10)[1]),
        **{f"acf_abs_l{k}": float(a.autocorr(k)) for k in (1, 5, 10, 20)},
    }


def ajustar_garch(ret: pd.Series, dist: str = "t", ultima_obs: int | None = None):
    """Ajusta GARCH(1,1) con media constante sobre retornos en porcentaje."""
    from arch import arch_model

    y = ret.dropna() * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist=dist).fit(
            disp="off", last_obs=ultima_obs)


def pronostico_garch(ret: pd.Series, anualizacion: float = 365.0, dist: str = "t") -> dict:
    res = ajustar_garch(ret, dist)
    p = res.params
    alpha, beta = float(p["alpha[1]"]), float(p["beta[1]"])
    persistencia = alpha + beta
    sigma_manana = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])) / 100
    incond = float(np.sqrt(p["omega"] / (1 - persistencia))) / 100 if persistencia < 1 else np.nan
    return {
        "omega": float(p["omega"]),
        "alpha": alpha,
        "beta": beta,
        "persistencia": persistencia,
        "vida_media_dias": float(np.log(0.5) / np.log(persistencia)) if 0 < persistencia < 1 else np.inf,
        "sigma_manana": sigma_manana,
        "sigma_manana_anual": sigma_manana * np.sqrt(anualizacion),
        "sigma_incondicional": incond,
        "sigma_condicional": res.conditional_volatility / 100,
    }


def evaluar_garch(ret: pd.Series, frac_entrenamiento: float = 0.7, ventana_movil: int = 20) -> dict:
    """Pronostico a un paso fuera de muestra: GARCH frente a desviacion movil.

    Los parametros se estiman solo con el entrenamiento y se congelan. La
    perdida es QLIKE (log h + r^2/h), la estandar para comparar previsiones de
    varianza: penaliza mas quedarse corto que pasarse, como el riesgo real.
    """
    r = ret.dropna() * 100
    n = int(len(r) * frac_entrenamiento)
    res = ajustar_garch(ret, ultima_obs=n)
    h_garch = res.forecast(horizon=1, start=n - 1, reindex=False).variance.iloc[:, 0]
    # La prevision hecha en t es para t+1: se alinea con el retorno siguiente.
    h_garch = pd.Series(h_garch.to_numpy()[:-1], index=r.index[n:])
    h_movil = (r.rolling(ventana_movil).std() ** 2).shift(1).loc[r.index[n:]]
    real2 = (r.loc[r.index[n:]] - r.iloc[:n].mean()) ** 2
    qlike = lambda h: float(np.mean(np.log(h) + real2 / h))  # noqa: E731
    return {
        "qlike_garch": qlike(h_garch),
        "qlike_movil": qlike(h_movil),
        "corr_garch_abs": float(np.corrcoef(np.sqrt(h_garch), np.sqrt(real2))[0, 1]),
        "corr_movil_abs": float(np.corrcoef(np.sqrt(h_movil), np.sqrt(real2))[0, 1]),
        "n_prueba": len(real2),
    }


def plan_operacion(precio: float, sigma: float, capital: float, riesgo_por_operacion: float = 0.01,
                   k_stop: float = 2.0, riesgo_beneficio: float = 2.0) -> dict:
    """Stop, objetivo y tamano de una compra a partir de la sigma prevista.

    El stop se coloca a `k_stop` sigmas: con volatilidad alta queda mas lejos,
    para que el ruido normal del dia no lo salte. El tamano se ajusta para que,
    si salta, se pierda exactamente `riesgo_por_operacion` del capital. Asi,
    mas volatilidad implica stop mas ancho y posicion mas pequena, y la
    perdida en dinero no cambia.
    """
    if not (precio > 0 and sigma > 0 and capital > 0 and k_stop > 0 and riesgo_beneficio > 0):
        raise ValueError("precio, sigma, capital, k_stop y riesgo_beneficio deben ser positivos")
    if not 0 < riesgo_por_operacion <= 1:
        raise ValueError("riesgo_por_operacion debe estar entre 0 y 1")
    distancia = precio * (1 - np.exp(-k_stop * sigma))
    riesgo_monetario = capital * riesgo_por_operacion
    unidades = riesgo_monetario / distancia
    nominal = unidades * precio
    limitado = nominal > capital  # sin apalancamiento
    if limitado:
        unidades, nominal = capital / precio, capital
    return {
        "precio": precio,
        "sigma_diaria": sigma,
        "stop": precio - distancia,
        "objetivo": precio + riesgo_beneficio * distancia,
        "distancia_stop_pct": distancia / precio,
        "unidades": unidades,
        "nominal": nominal,
        "fraccion_capital": nominal / capital,
        "perdida_si_stop": unidades * distancia,
        "limitado_por_capital": bool(limitado),
    }
