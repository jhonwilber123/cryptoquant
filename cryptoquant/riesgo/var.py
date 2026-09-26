"""VaR y CVaR a un dia a partir de las posiciones, por cuatro metodos.

El VaR de una cartera no se estima sobre sus retornos pasados: el sistema
cambia de exposicion cada semana, y la cartera de hoy no es la de hace un
ano. Se estima sobre la serie que habria dado la posicion de HOY aplicada a
los retornos pasados de cada activo (simulacion historica con posiciones
actuales). Asi el VaR del dia en que se esta en efectivo es cero, y el del
dia en que se esta invertido al 60% refleja ese 60%.

Metodos, de mas ingenuo a mas completo:

    historico   cuantil empirico de la ventana. Sin supuestos, pero lento en
                reaccionar: un crash tarda meses en entrar y en salir.
    normal      media y desviacion. Reacciona igual de lento y ademas
                subestima las colas (ver laboratorio, paso 4).
    t           t de Student con grados de libertad estimados: colas gruesas,
                pero volatilidad todavia constante en la ventana.
    garch_t     GARCH(1,1) con t: colas gruesas y volatilidad que reacciona al
                dia de ayer.

Todo VaR se expresa como perdida positiva en fraccion del capital.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

METODOS = ("historico", "normal", "t", "garch_t")
NU_MIN, NU_MAX = 2.5, 50.0


@dataclass
class ParamsGarch:
    """GARCH(1,1)-t estimado sobre la serie estandarizada (desviacion 1).

    Estimarlo sin escala permite reutilizar los parametros aunque la
    exposicion cambie entre reajustes: la escala la pone la serie de cada dia.
    """
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float


def ajustar_garch_t(z: np.ndarray) -> ParamsGarch | None:
    from arch import arch_model

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(z, mean="Constant", vol="GARCH", p=1, q=1, dist="t",
                             rescale=False).fit(disp="off", show_warning=False)
    except Exception:  # noqa: BLE001 - un ajuste fallido cae al metodo t
        return None
    p = res.params
    if not np.all(np.isfinite(p.to_numpy())) or p["alpha[1]"] + p["beta[1]"] >= 1:
        return None
    return ParamsGarch(float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]),
                       float(p["beta[1]"]), float(np.clip(p["nu"], NU_MIN, NU_MAX)))


def sigma_siguiente(z: np.ndarray, g: ParamsGarch) -> float:
    """Desviacion prevista para el dia siguiente al ultimo de `z`."""
    s2 = float(np.var(z))
    for r in z:
        s2 = g.omega + g.alpha * (r - g.mu) ** 2 + g.beta * s2
    return float(np.sqrt(s2))


def ajustar_nu(z: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nu = stats.t.fit(z, floc=0)[0]
    return float(np.clip(nu, NU_MIN, NU_MAX))


def cuantil_t(nu: float, cola: float) -> tuple[float, float]:
    """(cuantil, cola media) de una t con varianza 1, en la cola izquierda."""
    k = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(cola, nu)
    es = -stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1) * cola)
    return q * k, es * k


def var_es(x: np.ndarray, metodo: str, alpha: float = 0.95, nu: float | None = None,
           garch: ParamsGarch | None = None) -> tuple[float, float]:
    """(VaR, CVaR) del dia siguiente a la serie `x`, como perdidas positivas."""
    if metodo not in METODOS:
        raise ValueError(f"metodo desconocido: {metodo!r} (use {METODOS})")
    if not 0.5 < alpha < 1:
        raise ValueError("el nivel de confianza debe estar entre 0.5 y 1 (p. ej. 0.95)")
    if len(x) < 2:
        raise ValueError("hacen falta al menos dos retornos para estimar un VaR")
    cola = 1 - alpha
    if metodo == "historico":
        q = np.quantile(x, cola)
        return -q, -x[x <= q].mean()
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    if metodo == "normal":
        z = stats.norm.ppf(cola)
        return -(mu + sd * z), -(mu - sd * stats.norm.pdf(z) / cola)
    if metodo == "garch_t" and garch is not None:
        mu = garch.mu * sd
        sd = sigma_siguiente(x / sd, garch) * sd
        nu = garch.nu
    q, es = cuantil_t(nu or ajustar_nu((x - mu) / sd), cola)
    return -(mu + sd * q), -(mu + sd * es)


def var_siguiente(retornos: pd.DataFrame, pesos: pd.Series, alpha: float = 0.95,
                  ventana: int = 500, metodos=METODOS) -> pd.DataFrame:
    """VaR y CVaR del dia siguiente a la ultima fila, con los pesos dados."""
    w = pesos.reindex(retornos.columns, fill_value=0.0).fillna(0.0).to_numpy()
    x = retornos.fillna(0.0).to_numpy()[-ventana:] @ w
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    if np.abs(w).sum() < 1e-12 or not sd > 0:
        # Sin exposicion (o sin movimiento) no hay riesgo de mercado que medir.
        return pd.DataFrame({"var": 0.0, "es": 0.0}, index=list(metodos))
    nu = ajustar_nu((x - x.mean()) / sd)
    garch = ajustar_garch_t(x / sd) if "garch_t" in metodos else None
    filas = {m: dict(zip(("var", "es"), var_es(x, m, alpha, nu=nu, garch=garch))) for m in metodos}
    return pd.DataFrame(filas).T


def pronosticar(retornos: pd.DataFrame, posiciones: pd.DataFrame | pd.Series,
                alpha: float = 0.95, ventana: int = 500, minimo: int = 250,
                reajuste: int = 21, metodos=METODOS) -> pd.DataFrame:
    """VaR y CVaR de cada dia siguiente a una fecha de `posiciones`.

    Args:
        retornos: retornos simples diarios de cada activo.
        posiciones: pesos al cierre de cada fecha (fraccion del capital), o una
            Serie de pesos constantes para una cartera que no cambia.

    Devuelve una fila por dia pronosticado (el dia siguiente a la posicion)
    con `exposicion`, `retorno` realizado y `var_<metodo>`, `es_<metodo>`.
    Estrictamente causal: el pronostico de t+1 solo usa retornos hasta t.
    """
    retornos = retornos.fillna(0.0)
    if isinstance(posiciones, pd.Series):
        posiciones = pd.DataFrame([posiciones.to_numpy()] * len(retornos),
                                  index=retornos.index, columns=posiciones.index)
    posiciones = posiciones.reindex(columns=retornos.columns, fill_value=0.0).fillna(0.0)
    R = retornos.to_numpy()
    pos_en = retornos.index.get_indexer(posiciones.index)

    filas, fechas = [], []
    garch, nu, desde_ajuste = None, None, reajuste
    for k, (fecha, w) in enumerate(zip(posiciones.index, posiciones.to_numpy())):
        i = pos_en[k]
        if i < minimo - 1 or i + 1 >= len(R):
            continue
        fila = {"exposicion": float(np.abs(w).sum()), "retorno": float(R[i + 1] @ w)}
        x = R[max(0, i + 1 - ventana): i + 1] @ w
        if fila["exposicion"] < 1e-6 or np.std(x) == 0:
            fila |= {f"{p}_{m}": 0.0 for m in metodos for p in ("var", "es")}
        else:
            if desde_ajuste >= reajuste or nu is None:
                z = (x - x.mean()) / x.std(ddof=1)
                nu = ajustar_nu(z) if "t" in metodos else None
                garch = ajustar_garch_t(x / x.std(ddof=1)) if "garch_t" in metodos else None
                desde_ajuste = 0
            for m in metodos:
                fila[f"var_{m}"], fila[f"es_{m}"] = var_es(x, m, alpha, nu=nu, garch=garch)
        desde_ajuste += 1
        filas.append(fila)
        fechas.append(retornos.index[i + 1])
    return pd.DataFrame(filas, index=pd.Index(fechas, name="fecha"))
