"""Ingesta de datos de mercado.

Estrategia: descargar OHLCV via ccxt (API publica, sin claves), cachear en
parquet y, si no hay red, degradar a un generador sintetico reproducible para
que el pipeline completo siga siendo ejecutable y testeable offline.
"""
from __future__ import annotations

import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, Config, DataConfig

log = logging.getLogger(__name__)

_TF_MS = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000, "1w": 604_800_000}


def drop_incomplete_bars(df: pd.DataFrame, timeframe: str,
                         now: pd.Timestamp | None = None,
                         grace_seconds: int = 120) -> pd.DataFrame:
    """Elimina la barra en curso, que aun no ha cerrado.

    Los exchanges devuelven tambien el periodo actual a medio formar: su
    "cierre" es el precio de este instante, no el cierre real. Usarlo tiene dos
    consecuencias graves.

    En el backtest, una feature calculada sobre esa barra mezcla informacion de
    dentro del periodo con la decision de ese mismo periodo.

    En el forward test es peor: se registraria una decision sobre un precio que
    todavia va a cambiar, y al evaluarla mas tarde se compararia contra el
    cierre definitivo de esa misma barra. La diferencia entre ambos no la ha
    ganado la estrategia; es un artefacto del reloj.

    `grace_seconds` exige que el periodo lleve un rato cerrado. Un proceso
    automatico puede dispararse justo en el instante del cierre, y el exchange
    tarda unos segundos en consolidar la vela: sin margen, esa ejecucion se
    llevaria una barra a medio consolidar creyendola definitiva.
    """
    if df.empty:
        return df
    delta = pd.Timedelta(milliseconds=_TF_MS[timeframe])
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + delta + pd.Timedelta(seconds=grace_seconds) <= now]


def _cache_path(exchange: str, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-")
    return CACHE_DIR / f"{exchange}_{safe}_{timeframe}.parquet"


def _fetch_ccxt(exchange_id: str, symbol: str, timeframe: str, since_ms: int,
                max_bars: int) -> pd.DataFrame:
    """Descarga paginada de OHLCV. Lanza excepcion si no hay red/ccxt."""
    import ccxt  # import diferido: el pipeline offline no lo necesita

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 20_000})
    step = _TF_MS[timeframe]
    rows: list[list] = []
    cursor = since_ms
    while len(rows) < max_bars:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)

    if not rows:
        raise RuntimeError(f"sin datos para {symbol}")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    # La barra en curso no se guarda: cachearla la congelaria a medio formar.
    return drop_incomplete_bars(df, timeframe)


def _synthetic(symbol: str, timeframe: str, start: str, n: int, seed: int) -> pd.DataFrame:
    """Serie sintetica con propiedades realistas de cripto.

    Incluye deriva positiva, volatilidad con clustering tipo GARCH, colas
    pesadas (t de Student) y regimenes alcistas/bajistas persistentes. No sirve
    para sacar conclusiones de inversion, solo para validar que el codigo corre.
    """
    # crc32 y no hash(): hash() de un str cambia en cada proceso, y el generador
    # "reproducible" daba un mercado distinto en cada ejecucion de los tests.
    rng = np.random.default_rng(seed + zlib.crc32(symbol.encode("utf-8")) % 10_000)
    # Cada activo tiene su propio perfil beta/vol respecto al mercado.
    beta = 0.7 + rng.uniform(0.0, 1.1)
    base_vol = 0.035 * beta

    # Componente de mercado comun (para que las correlaciones sean realistas).
    mkt_rng = np.random.default_rng(seed)
    mkt = mkt_rng.standard_t(df=4, size=n) * 0.03

    omega, alpha, beta_g = base_vol**2 * 0.05, 0.09, 0.88
    var_t = base_vol**2
    idio = np.zeros(n)
    shocks = rng.standard_t(df=4, size=n) / np.sqrt(4 / 2)
    for i in range(n):
        idio[i] = np.sqrt(var_t) * shocks[i]
        var_t = omega + alpha * idio[i] ** 2 + beta_g * var_t

    # Regimenes persistentes mediante cadena de Markov de 2 estados.
    regime = np.zeros(n)
    state, p_stay = 1, 0.985
    for i in range(n):
        if rng.random() > p_stay:
            state *= -1
        regime[i] = state
    drift = np.where(regime > 0, 0.0022, -0.0016) * beta

    ret = drift + beta * mkt + idio
    close = 100.0 * np.exp(np.cumsum(ret))

    idx = pd.date_range(start=start, periods=n, freq="D" if timeframe == "1d" else "h", tz="UTC")
    intraday = np.abs(rng.normal(0, 0.008, n)) + 0.002
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.exp(rng.normal(13, 0.6, n)) * (1 + 3 * np.abs(ret))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.Index(idx, name="date"),
    )


def cache_is_stale(df: pd.DataFrame, timeframe: str,
                   now: pd.Timestamp | None = None,
                   grace_seconds: int = 120) -> bool:
    """True si ya ha cerrado alguna barra posterior a la ultima cacheada.

    Sin esta comprobacion la cache se sirve indefinidamente y el sistema se
    queda congelado en el pasado sin avisar. En un proceso automatico diario
    eso es letal y ademas silencioso: cada ejecucion recalcularia la decision
    de la misma fecha, el diario la rechazaria por duplicada, y el experimento
    no avanzaria nunca aparentando funcionar con normalidad.
    """
    if df.empty:
        return True
    delta = pd.Timedelta(milliseconds=_TF_MS[timeframe])
    now = now or pd.Timestamp.now(tz="UTC")
    # La barra siguiente a la ultima cacheada cierra dos periodos despues.
    return now >= df.index[-1] + 2 * delta + pd.Timedelta(seconds=grace_seconds)


def load_symbol(symbol: str, cfg: DataConfig, seed: int = 42,
                force_refresh: bool = False,
                offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Devuelve (ohlcv, origen).

    Origen: 'cache' (al dia) | 'live' (recien descargado) | 'cache-stale'
    (atrasada pero sin red para actualizarla) | 'synthetic' (sin datos reales).
    """
    pair = f"{symbol}/{cfg.quote}" if "/" not in symbol else symbol
    path = _cache_path(cfg.exchange, pair, cfg.timeframe)

    cached: pd.DataFrame | None = None
    if cfg.cache and path.exists():
        candidate = drop_incomplete_bars(pd.read_parquet(path), cfg.timeframe)
        if len(candidate) > 100:
            cached = candidate
            if not force_refresh and not cache_is_stale(candidate, cfg.timeframe):
                return cached, "cache"

    if offline:
        # Modo lectura: jamas toca la red. Lo usan los visores (el panel), que
        # deben mostrar lo que hay sin efectos secundarios y sin inventar nada.
        if cached is not None:
            return cached, "cache-stale"
        raise FileNotFoundError(f"sin cache local para {pair}")

    since_ms = int(pd.Timestamp(cfg.start, tz="UTC").timestamp() * 1000)
    try:
        df = _fetch_ccxt(cfg.exchange, pair, cfg.timeframe, since_ms, cfg.max_bars)
        if cfg.cache:
            df.to_parquet(path)
        return df, "live"
    except Exception as exc:  # red caida, ccxt ausente, simbolo inexistente
        # Una cache atrasada sigue siendo un dato real; una serie sintetica no.
        # Ante un corte de red se prefiere siempre la primera, y se etiqueta
        # como atrasada para que quien la use lo sepa.
        if cached is not None:
            log.warning("descarga fallida para %s (%s); usando cache atrasada "
                        "hasta %s", pair, exc, cached.index[-1].date())
            return cached, "cache-stale"
        if not cfg.allow_synthetic_fallback:
            raise
        log.warning("descarga fallida para %s (%s); usando serie sintetica", pair, exc)
        n = min(cfg.max_bars, 2000)
        return _synthetic(pair, cfg.timeframe, cfg.start, n, seed), "synthetic"


def load_universe(cfg: Config, force_refresh: bool = False, offline: bool = False
                  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Carga todos los simbolos y construye el panel de cierres alineado.

    Returns:
        ohlcv: dict simbolo -> DataFrame OHLCV
        closes: DataFrame de cierres alineados en un indice comun
        origins: dict simbolo -> origen del dato
    """
    ohlcv: dict[str, pd.DataFrame] = {}
    origins: dict[str, str] = {}
    for sym in cfg.data.symbols:
        try:
            df, origin = load_symbol(sym, cfg.data, seed=cfg.seed,
                                     force_refresh=force_refresh, offline=offline)
        except FileNotFoundError:
            if not offline:
                raise
            continue  # en modo lectura, un simbolo sin cache simplemente no se muestra
        ohlcv[sym] = df
        origins[sym] = origin
    if not ohlcv:
        raise FileNotFoundError("no hay datos en la cache local")

    closes = pd.DataFrame({s: d["close"] for s, d in ohlcv.items()})
    # Un activo debe existir en al menos el 60% del historico comun para entrar.
    coverage = closes.notna().mean()
    keep = coverage[coverage >= 0.60].index.tolist()
    dropped = sorted(set(closes.columns) - set(keep))
    if dropped:
        log.warning("descartados por historico insuficiente: %s", dropped)
    closes = closes[keep].ffill().dropna(how="all")
    # Recortamos al primer instante en que todos los supervivientes tienen dato.
    first_valid = closes.apply(lambda c: c.first_valid_index()).max()
    closes = closes.loc[first_valid:].dropna()
    ohlcv = {s: ohlcv[s].reindex(closes.index).ffill() for s in keep}
    return ohlcv, closes, {s: origins[s] for s in keep}
