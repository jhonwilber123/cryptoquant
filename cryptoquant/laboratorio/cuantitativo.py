"""Modelamiento cuantitativo: forma de los retornos y estadistica de senales.

Dos preguntas previas a cualquier estrategia:

1. Que forma tienen los retornos. Si fueran normales, un movimiento de 4
   desviaciones ocurriria una vez cada ~43 anos de velas diarias. En cripto
   ocurre varias veces al ano. Todo lo que asume normalidad (el VaR
   parametrico, un stop a "3 sigmas") subestima el riesgo real.

2. Si una senal tecnica aporta algo. No basta con que acierte mas de la mitad
   de las veces: hay que compararla con lo que habria pasado sin senal en el
   mismo mercado. En un mercado alcista, "comprar siempre" ya acierta mas del
   50%, y una senal de compra que acierta el 53% puede no aportar nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def distribucion_retornos(ret: pd.Series) -> dict:
    r = ret.dropna().to_numpy(dtype=float)
    gl, _, _ = stats.t.fit(r)
    return {
        "n": len(r),
        "media": float(np.mean(r)),
        "desviacion": float(np.std(r, ddof=1)),
        "asimetria": float(stats.skew(r)),
        "curtosis_exceso": float(stats.kurtosis(r)),  # 0 en la normal
        "jarque_bera_p": float(stats.jarque_bera(r).pvalue),
        "t_grados_libertad": float(gl),  # < 5: colas muy gruesas
    }


def tabla_colas(ret: pd.Series, umbrales=(2, 3, 4, 5)) -> pd.DataFrame:
    """Frecuencia de movimientos de mas de k desviaciones: observada y normal."""
    r = ret.dropna()
    z = ((r - r.mean()) / r.std()).abs()
    filas = []
    for k in umbrales:
        obs = int((z > k).sum())
        esperado = 2 * stats.norm.sf(k) * len(z)
        filas.append({
            "umbral_sigmas": k,
            "observados": obs,
            "esperados_normal": esperado,
            "veces_mas": obs / esperado if esperado > 0 else np.nan,
        })
    return pd.DataFrame(filas).set_index("umbral_sigmas")


def _cruza_arriba(a: pd.Series, b) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift(1) <= b_prev)


def senales(v: pd.DataFrame) -> pd.DataFrame:
    """Eventos de senal en la vela en que ocurren (conocidos a su cierre).

    Devuelve una columna por senal con +1 (compra), -1 (venta) o 0.
    """
    c = v["close"]
    defs = {
        "rsi_sale_sobreventa": (_cruza_arriba(v["rsi_14"], 30), +1),
        "rsi_sale_sobrecompra": (_cruza_arriba(-v["rsi_14"], -70), -1),
        "macd_cruce_alcista": (_cruza_arriba(v["macd"], v["macd_senal"]), +1),
        "macd_cruce_bajista": (_cruza_arriba(v["macd_senal"], v["macd"]), -1),
        "cruce_dorado_20_50": (_cruza_arriba(v["sma_20"], v["sma_50"]), +1),
        "cruce_muerte_20_50": (_cruza_arriba(v["sma_50"], v["sma_20"]), -1),
        "bollinger_rompe_inferior": (_cruza_arriba(v["bb_inf"], c), +1),
        "bollinger_rompe_superior": (_cruza_arriba(c, v["bb_sup"]), -1),
    }
    return pd.DataFrame({k: ev.fillna(False).astype(int) * d for k, (ev, d) in defs.items()},
                        index=v.index)


def estadistica_senales(v: pd.DataFrame, horizontes=(1, 5, 10)) -> pd.DataFrame:
    """Acierto y retorno medio despues de cada senal, frente a no hacer nada.

    `retorno_medio` y `acierto` estan en la direccion de la senal: para una
    senal de venta, acertar es que el precio baje. La base es la misma
    direccion aplicada a todas las velas.

    `p_valor` contrasta el acierto frente a la base con un test binomial. Con
    horizontes > 1 los eventos cercanos se solapan y no son independientes:
    el p-valor sale optimista. Y con 8 senales x 3 horizontes, alguna saldra
    "significativa" por azar; exija p < 0.05 / 24 antes de creerse nada.
    """
    ev = senales(v)
    lc = np.log(v["close"])
    filas = []
    for h in horizontes:
        fwd = lc.shift(-h) - lc
        valido = fwd.notna()
        for nombre in ev.columns:
            d = int(ev[nombre][ev[nombre] != 0].iloc[0]) if (ev[nombre] != 0).any() else 0
            mask = (ev[nombre] != 0) & valido
            n = int(mask.sum())
            base = fwd[valido] * (d or 1)
            dir_ret = fwd[mask] * d
            aciertos = int((dir_ret > 0).sum())
            acierto_base = float((base > 0).mean())
            filas.append({
                "senal": nombre,
                "direccion": "compra" if d > 0 else "venta",
                "horizonte": h,
                "eventos": n,
                "acierto": aciertos / n if n else np.nan,
                "acierto_base": acierto_base,
                "retorno_medio": float(dir_ret.mean()) if n else np.nan,
                "retorno_medio_base": float(base.mean()),
                "p_valor": (float(stats.binomtest(aciertos, n, acierto_base,
                                                  alternative="greater").pvalue)
                            if n else np.nan),
            })
    return pd.DataFrame(filas)
