"""Ingenieria de caracteristicas.

Regla invariable de todo el modulo: toda feature en la barra t solo puede usar
informacion disponible hasta el cierre de t. Cualquier `.shift(-k)`, centrado o
estadistico calculado sobre la muestra completa introduce lookahead y hace que
el backtest mienta. Las funciones de aqui son estrictamente causales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(close).diff()


def realized_vol(returns: pd.Series, window: int, ann: float) -> pd.Series:
    """Volatilidad realizada anualizada sobre ventana movil."""
    return returns.rolling(window, min_periods=max(5, window // 3)).std() * np.sqrt(ann)


def ewma_vol(returns: pd.Series, halflife: int, ann: float) -> pd.Series:
    """Volatilidad EWMA: reacciona mas rapido que la ventana movil simple."""
    return returns.ewm(halflife=halflife, min_periods=10).std() * np.sqrt(ann)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = close.ewm(span=fast, min_periods=fast).mean() - close.ewm(span=slow, min_periods=slow).mean()
    return line - line.ewm(span=signal, min_periods=signal).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range: base para dimensionar stops y barreras."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window).mean()


def rolling_drawdown(close: pd.Series, window: int = 90) -> pd.Series:
    peak = close.rolling(window, min_periods=10).max()
    return close / peak - 1.0


def hurst_exponent(series: pd.Series, max_lag: int = 40) -> float:
    """Exponente de Hurst por la funcion de estructura (varianza de diferencias).

    Se ajusta log(std(s[t+lag] - s[t])) contra log(lag); la pendiente es H.

    H > 0.5 -> serie persistente (favorece momentum)
    H < 0.5 -> serie antipersistente (favorece reversion a la media)
    H ~ 0.5 -> paseo aleatorio (no hay senal explotable)

    Mide la memoria del componente ESTOCASTICO. Una deriva determinista no
    eleva H: en una recta las diferencias a lag fijo son constantes, asi que su
    varianza no crece con el lag. Es el comportamiento correcto (una tendencia
    lineal no es memoria larga), pero conviene saberlo al interpretar el valor.
    """
    s = series.dropna().to_numpy(dtype=float)
    if len(s) < max_lag * 3:
        return np.nan
    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        sd = np.std(diff)
        tau.append(sd if sd > 0 else np.nan)
    tau_arr = np.asarray(tau, dtype=float)
    mask = np.isfinite(tau_arr) & (tau_arr > 0)
    if mask.sum() < 5:
        return np.nan
    slope = np.polyfit(np.log(lags[mask]), np.log(tau_arr[mask]), 1)[0]
    return float(slope)


def rolling_hurst(close: pd.Series, window: int = 250, step: int = 5) -> pd.Series:
    """Hurst movil. Se calcula cada `step` barras y se propaga hacia delante
    porque es caro y cambia despacio."""
    log_px = np.log(close)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    positions = range(window, len(close), step)
    for i in positions:
        out.iloc[i] = hurst_exponent(log_px.iloc[i - window : i])
    return out.ffill()


def build_features(df: pd.DataFrame, ann: float, cfg) -> pd.DataFrame:
    """Construye la matriz de features para un activo.

    Args:
        df: OHLCV del activo.
        ann: factor de anualizacion.
        cfg: SignalConfig.
    """
    close = df["close"]
    ret = log_returns(close)
    feats = pd.DataFrame(index=df.index)

    # --- Momentum multi-horizonte, normalizado por volatilidad -------------
    # Dividir por la vol hace comparables activos con perfiles de riesgo muy
    # distintos (BTC vs una altcoin), que es lo que permite rankearlos.
    vol_d = ret.rolling(cfg.momentum_windows[0], min_periods=10).std()
    for w in cfg.momentum_windows:
        raw = np.log(close / close.shift(w))
        feats[f"mom_{w}"] = raw / (vol_d * np.sqrt(w)).replace(0.0, np.nan)

    # --- Volatilidad y su estructura temporal ------------------------------
    feats["vol_20"] = realized_vol(ret, 20, ann)
    feats["vol_60"] = realized_vol(ret, 60, ann)
    feats["vol_ewma"] = ewma_vol(ret, 20, ann)
    # Ratio vol corta / vol larga: >1 indica estres reciente.
    feats["vol_ratio"] = feats["vol_20"] / feats["vol_60"].replace(0.0, np.nan)

    # --- Osciladores y tendencia -------------------------------------------
    feats["rsi_14"] = rsi(close, 14) / 100.0
    feats["macd_hist"] = macd_hist(close) / close
    ma = close.rolling(cfg.regime_ma, min_periods=cfg.regime_ma // 2).mean()
    feats["dist_ma"] = close / ma - 1.0
    feats["regime_up"] = (close > ma).astype(float)

    # --- Riesgo de cola y estructura de la distribucion --------------------
    feats["skew_60"] = ret.rolling(60, min_periods=30).skew()
    feats["kurt_60"] = ret.rolling(60, min_periods=30).kurt()
    feats["dd_90"] = rolling_drawdown(close, 90)

    # --- Microestructura ---------------------------------------------------
    feats["atr_pct"] = atr(df, 14) / close
    vol_ma = df["volume"].rolling(60, min_periods=20).mean()
    vol_sd = df["volume"].rolling(60, min_periods=20).std()
    feats["volume_z"] = ((df["volume"] - vol_ma) / vol_sd.replace(0.0, np.nan)).clip(-5, 5)

    # --- Memoria larga ------------------------------------------------------
    feats["hurst"] = rolling_hurst(close, window=250, step=5)

    return feats.replace([np.inf, -np.inf], np.nan)
