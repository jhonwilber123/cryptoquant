"""Series de tiempo: retornos y no precios, autocorrelacion, calendario, ARIMA.

El precio de un criptoactivo no es estacionario: su media y su varianza
cambian con el tiempo, y una regresion entre dos precios que suben encuentra
relacion aunque no la haya (regresion espuria). El retorno logaritmico si es
aproximadamente estacionario, que es la condicion para que un modelo
estimado con el pasado diga algo sobre el futuro.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller

DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]


def precio_frente_a_retorno(close: pd.Series) -> pd.DataFrame:
    """ADF sobre el log-precio y sobre el retorno. H0: raiz unitaria."""
    lp = np.log(close.dropna())
    ret = lp.diff().dropna()
    filas = {}
    for nombre, s in (("log_precio", lp), ("retorno_log", ret)):
        est, p, *_ = adfuller(s, autolag="AIC")
        filas[nombre] = {"adf_estadistico": est, "adf_p": p,
                         "estacionaria": "si" if p < 0.05 else "no"}
    return pd.DataFrame(filas).T


def autocorrelacion(ret: pd.Series, rezagos: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ACF de retornos, de |retornos| y de retornos al cuadrado, y Ljung-Box.

    El resultado tipico en cripto es la clave de todo el laboratorio: los
    retornos casi no tienen memoria (dificil predecir la direccion), pero su
    magnitud si (la volatilidad se puede prever).
    """
    r = ret.dropna()
    tabla = pd.DataFrame({
        "retorno": acf(r, nlags=rezagos, fft=True)[1:],
        "abs_retorno": acf(r.abs(), nlags=rezagos, fft=True)[1:],
        "retorno_cuadrado": acf(r**2, nlags=rezagos, fft=True)[1:],
    }, index=pd.RangeIndex(1, rezagos + 1, name="rezago"))
    tabla["banda_95"] = 1.96 / np.sqrt(len(r))
    lb = {nombre: acorr_ljungbox(s, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
          for nombre, s in (("retorno", r), ("retorno_cuadrado", r**2))}
    return tabla, pd.DataFrame({"ljung_box_p_10": lb})


def patrones_calendario(ret: pd.Series) -> dict[str, pd.DataFrame]:
    """Retorno y volatilidad medios por hora UTC y por dia de la semana.

    Kruskal-Wallis contrasta si todos los grupos tienen la misma
    distribucion. Es habitual que la volatilidad dependa de la hora (abren
    Asia, Europa y Estados Unidos) y que el retorno medio no: la primera es
    estructura del mercado, lo segundo seria dinero gratis.
    """
    r = ret.dropna()
    out = {}
    grupos = {"dia": (r.index.dayofweek, DIAS)}
    if (r.index.hour != 0).any():
        grupos = {"hora": (r.index.hour, None), **grupos}
    for nombre, (clave, etiquetas) in grupos.items():
        g = r.groupby(clave)
        t = pd.DataFrame({"n": g.size(), "retorno_medio": g.mean(), "volatilidad": g.std()})
        t["t_estadistico"] = t["retorno_medio"] / (t["volatilidad"] / np.sqrt(t["n"]))
        if etiquetas:
            t.index = [etiquetas[i] for i in t.index]
        t.index.name = nombre
        muestras = [s.to_numpy() for _, s in g]
        t.attrs["kruskal_p_retorno"] = float(stats.kruskal(*muestras).pvalue)
        t.attrs["kruskal_p_volatilidad"] = float(stats.kruskal(*[np.abs(m) for m in muestras]).pvalue)
        out[nombre] = t
    return out


def arima_basico(ret: pd.Series, frac_entrenamiento: float = 0.8, max_p: int = 2,
                 max_q: int = 2) -> dict:
    """ARIMA(p,0,q) sobre retornos, elegido por AIC en el tramo de entrenamiento.

    Se evalua con prevision a un paso sobre el tramo de prueba, con los
    parametros congelados, y se compara con predecir siempre 0. Si el ARIMA
    no mejora a esa prevision trivial, la autocorrelacion de los retornos es
    demasiado debil para explotarla.
    """
    r = ret.dropna() * 100  # en porcentaje: el optimizador converge mejor
    r = pd.Series(r.to_numpy(), index=pd.RangeIndex(len(r)))
    n = int(len(r) * frac_entrenamiento)
    train = r.iloc[:n]

    mejor = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                res = ARIMA(train, order=(p, 0, q), trend="c").fit()
                if mejor is None or res.aic < mejor[1].aic:
                    mejor = ((p, 0, q), res)
        orden, res = mejor
        completo = res.apply(r)  # mismos parametros sobre toda la serie
    pred = completo.fittedvalues.iloc[n:]
    real = r.iloc[n:]
    rmse = lambda e: float(np.sqrt(np.mean(e**2)))  # noqa: E731
    return {
        "orden": orden,
        "aic": float(res.aic),
        "rmse_arima": rmse(real - pred),
        "rmse_cero": rmse(real),
        "rmse_media_historica": rmse(real - train.mean()),
        "mejora_sobre_cero": 1 - rmse(real - pred) / rmse(real),
        "acierto_direccion": float((np.sign(pred) == np.sign(real)).mean()),
        "n_prueba": len(real),
    }
