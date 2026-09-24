"""Contrastes econometricos sobre las series.

Sirven para responder a una pregunta previa a cualquier modelo: esta serie
tiene estructura explotable o es ruido? Si el precio es un paseo aleatorio puro
y los retornos no tienen autocorrelacion, ninguna cantidad de machine learning
va a extraer senal, y conviene saberlo antes de desplegar capital.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.technical import hurst_exponent


@dataclass
class SeriesDiagnostics:
    symbol: str
    adf_pvalue: float          # H0: raiz unitaria (no estacionaria)
    kpss_pvalue: float         # H0: estacionaria (contraste complementario)
    ljungbox_pvalue: float     # H0: sin autocorrelacion en retornos
    arch_pvalue: float         # H0: sin heterocedasticidad condicional
    hurst: float
    jarque_bera_pvalue: float  # H0: normalidad
    annual_vol: float
    verdict: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _safe(fn, default=np.nan):
    try:
        return fn()
    except Exception:
        return default


def diagnose(close: pd.Series, symbol: str, ann: float) -> SeriesDiagnostics:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tsa.stattools import adfuller, kpss

    px = close.dropna()
    log_px = np.log(px)
    ret = log_px.diff().dropna()

    # statsmodels emite avisos informativos que no son errores: KPSS avisa
    # cuando el p-valor sale del rango tabulado (se recorta a 0.01/0.10) y
    # het_arch anuncia un cambio futuro de tipo de retorno. Ninguno afecta al
    # valor calculado, asi que se silencian solo en este bloque.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = _safe(lambda: float(adfuller(log_px, autolag="AIC")[1]))
        kpss_p = _safe(lambda: float(kpss(log_px, regression="c", nlags="auto")[1]))
        lb_p = _safe(lambda: float(acorr_ljungbox(ret, lags=[10], return_df=True)["lb_pvalue"].iloc[0]))
        arch_p = _safe(lambda: float(het_arch(ret, nlags=10)[1]))
        jb_p = _safe(lambda: float(jarque_bera(ret)[1]))
    h = _safe(lambda: hurst_exponent(log_px))
    vol = float(ret.std(ddof=1) * np.sqrt(ann))

    # Sintesis en lenguaje llano de lo que dicen los contrastes.
    parts = []
    if np.isfinite(h):
        if h > 0.55:
            parts.append("tendencial (favorece momentum)")
        elif h < 0.45:
            parts.append("reversion a la media")
        else:
            parts.append("cercano a paseo aleatorio")
    if np.isfinite(lb_p) and lb_p < 0.05:
        parts.append("autocorrelacion significativa")
    if np.isfinite(arch_p) and arch_p < 0.05:
        parts.append("volatilidad agrupada (justifica GARCH)")
    if np.isfinite(jb_p) and jb_p < 0.05:
        parts.append("colas pesadas (VaR normal insuficiente)")

    return SeriesDiagnostics(
        symbol=symbol,
        adf_pvalue=adf_p,
        kpss_pvalue=kpss_p,
        ljungbox_pvalue=lb_p,
        arch_pvalue=arch_p,
        hurst=h,
        jarque_bera_pvalue=jb_p,
        annual_vol=vol,
        verdict="; ".join(parts) if parts else "sin estructura detectada",
    )


def diagnose_universe(closes: pd.DataFrame, ann: float) -> pd.DataFrame:
    rows = [diagnose(closes[c], c, ann).as_dict() for c in closes.columns]
    return pd.DataFrame(rows).set_index("symbol")


def engle_granger_pairs(closes: pd.DataFrame, pvalue_threshold: float = 0.05
                        ) -> pd.DataFrame:
    """Busca pares cointegrados mediante el contraste de Engle-Granger.

    Dos activos cointegrados comparten una tendencia estocastica comun: su
    spread es estacionario y revierte. Es la base de las estrategias de valor
    relativo, que son mucho menos direccionales (y por tanto menos arriesgadas)
    que comprar y mantener.

    Aviso estadistico: se contrastan todos los pares, asi que el riesgo de
    falso positivo por comparaciones multiples es real. Se reporta tambien el
    umbral de Bonferroni.
    """
    from statsmodels.tsa.stattools import coint

    cols = list(closes.columns)
    log_px = np.log(closes.dropna())
    n_tests = len(cols) * (len(cols) - 1) // 2
    bonferroni = pvalue_threshold / max(n_tests, 1)

    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            try:
                _, pval, _ = coint(log_px[a], log_px[b])
            except Exception:
                continue
            # Ratio de cobertura por OLS: spread = a - beta*b
            beta = float(np.polyfit(log_px[b], log_px[a], 1)[0])
            spread = log_px[a] - beta * log_px[b]
            half_life = _ou_half_life(spread)
            rows.append(
                {
                    "asset_a": a,
                    "asset_b": b,
                    "pvalue": float(pval),
                    "hedge_ratio": beta,
                    "half_life_days": half_life,
                    "spread_z": float((spread.iloc[-1] - spread.mean()) / spread.std(ddof=1)),
                    "significant": bool(pval < pvalue_threshold),
                    "significant_bonferroni": bool(pval < bonferroni),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)


def _ou_half_life(spread: pd.Series) -> float:
    """Vida media de reversion asumiendo proceso Ornstein-Uhlenbeck.

    Se estima por OLS de d(spread) sobre spread rezagado. Una vida media de 5
    dias es operable; una de 200 dias significa que el spread tardara mas en
    cerrar de lo que aguanta la mayoria de carteras.
    """
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    idx = lag.index.intersection(delta.index)
    if len(idx) < 30:
        return np.nan
    x = lag.loc[idx].to_numpy(dtype=float)
    y = delta.loc[idx].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    if slope >= 0:
        return np.inf  # no revierte
    return float(-np.log(2) / slope)
