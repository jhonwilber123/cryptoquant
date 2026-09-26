"""Preparacion de variables: indicadores como columnas, rezagos y objetivo.

Cada fila t describe lo que se sabia al cierre de la vela t. La unica
excepcion son las dos columnas objetivo (`retorno_futuro`, `sube`), que miran
a t+1 a proposito: son lo que se quiere predecir y jamas entran como variable
explicativa. `VARIABLES_MODELO` las excluye.

Hay dos clases de columnas. Las de nivel de precio (`sma_20`, `bb_sup`...)
sirven para graficar, pero no para un modelo: un arbol entrenado con precios
de 2022 no sabe que hacer con los de 2026. Las de `VARIABLES_MODELO` son
adimensionales (distancias relativas, osciladores, retornos) y comparables en
cualquier epoca.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.technical import rsi

REZAGOS = (1, 2, 3, 5, 10)

VARIABLES_MODELO = [
    "retorno", *(f"retorno_l{k}" for k in REZAGOS),
    "dist_sma_20", "dist_sma_50", "rsi_14",
    "macd_rel", "macd_senal_rel", "macd_hist_rel",
    "bb_pct", "bb_ancho", "vol_20", "vol_ratio", "rango_rel",
]


def preparar_variables(velas: pd.DataFrame, rezagos: tuple[int, ...] = REZAGOS) -> pd.DataFrame:
    c = velas["close"]
    v = velas[["open", "high", "low", "close", "volume"]].copy()

    v["retorno_simple"] = c.pct_change()
    v["retorno"] = np.log(c).diff()
    for k in rezagos:
        v[f"retorno_l{k}"] = v["retorno"].shift(k)

    # --- Medias ------------------------------------------------------------
    for w in (20, 50):
        v[f"sma_{w}"] = c.rolling(w).mean()
        v[f"dist_sma_{w}"] = c / v[f"sma_{w}"] - 1.0

    # --- RSI ---------------------------------------------------------------
    v["rsi_14"] = rsi(c, 14)
    v.loc[v.index[:14], "rsi_14"] = np.nan  # rsi() rellena con 50 el arranque

    # --- MACD (12, 26, 9) --------------------------------------------------
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    senal = macd.ewm(span=9, adjust=False).mean()
    v["macd"], v["macd_senal"], v["macd_hist"] = macd, senal, macd - senal
    v.loc[v.index[:34], ["macd", "macd_senal", "macd_hist"]] = np.nan  # 26 + 9 - 1 de arranque
    for col in ("macd", "macd_senal", "macd_hist"):
        v[f"{col}_rel"] = v[col] / c

    # --- Bandas de Bollinger (20, 2) ---------------------------------------
    sd = c.rolling(20).std()
    v["bb_media"] = v["sma_20"]
    v["bb_sup"] = v["bb_media"] + 2 * sd
    v["bb_inf"] = v["bb_media"] - 2 * sd
    v["bb_pct"] = (c - v["bb_inf"]) / (v["bb_sup"] - v["bb_inf"])
    v["bb_ancho"] = (v["bb_sup"] - v["bb_inf"]) / v["bb_media"]

    # --- Volatilidad movil -------------------------------------------------
    v["vol_20"] = v["retorno"].rolling(20).std()
    v["vol_ratio"] = v["vol_20"] / v["retorno"].rolling(100).std()
    v["rango_rel"] = (velas["high"] - velas["low"]) / c

    # --- Objetivo: lo unico que mira al futuro -----------------------------
    futuro = np.log(c.shift(-1) / c)
    v["retorno_futuro"] = futuro
    v["sube"] = (futuro > 0).astype(float).where(futuro.notna())

    return v.replace([np.inf, -np.inf], np.nan)


def matriz_modelo(variables: pd.DataFrame, columnas: list[str] | None = None
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) sin filas incompletas: el arranque de los indicadores y la ultima
    fila, que aun no tiene objetivo."""
    columnas = columnas or VARIABLES_MODELO
    datos = variables[columnas + ["sube"]].dropna()
    return datos[columnas], datos["sube"].astype(int)
