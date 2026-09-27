"""Precios para el piloto: velas diarias de Binance y el precio de este momento.

Las velas se guardan en la carpeta del piloto (`velas/`), nunca en
`data/cache/`: esa cache alimenta al sistema y al forward test. Se vuelven a
bajar solo cuando cierra un dia nuevo (00:00 UTC). Si Binance no responde,
se usan las ultimas guardadas y se avisa.

Las decisiones usan solo velas cerradas; el precio de ahora solo valora la
cartera y las ordenes.
"""
from __future__ import annotations

import math

import pandas as pd
import requests

from ..laboratorio import datos
from . import diario

API = "https://data-api.binance.vision/api/v3"
DESDE = "2020-01-01"


def ultimo_cierre(ahora: pd.Timestamp | None = None) -> pd.Timestamp:
    """Fecha (apertura, UTC) de la ultima vela diaria cerrada."""
    ahora = ahora if ahora is not None else pd.Timestamp.now(tz="UTC")
    return ahora.tz_convert("UTC").normalize() - pd.Timedelta(days=1)


def velas(activo: str, ahora: pd.Timestamp | None = None) -> pd.DataFrame:
    """Velas diarias de ACTIVO/USDT desde 2020. `attrs['aviso']` si son viejas."""
    a = datos.validar_ticker(activo)
    ruta = diario.carpeta() / "velas" / f"{a}USDT_1d.csv"
    try:
        guardadas = datos.leer_csv(ruta) if ruta.exists() else None
    except Exception:  # noqa: BLE001 - cache danada: se vuelve a bajar
        guardadas = None
    if guardadas is not None and len(guardadas) and guardadas.index[-1] >= ultimo_cierre(ahora):
        return guardadas
    try:
        df = datos.descargar_velas(f"{a}/USDT", "1d", DESDE)
        if df.empty:
            raise ValueError("respuesta vacia")
    except Exception as exc:  # noqa: BLE001 - red caida o par inexistente
        if guardadas is not None and len(guardadas):
            guardadas.attrs["aviso"] = (f"{a}: Binance no respondio; se usan velas hasta el "
                                        f"{guardadas.index[-1].date()}")
            return guardadas
        raise ValueError(f"{a}: no hay par {a}/USDT en Binance o no hay conexion "
                         f"({type(exc).__name__})") from None
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos.exportar_csv(df, ruta)
    return df


def cierres(activos, ahora: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Cierres diarios de cada activo (columnas) y los avisos de datos viejos."""
    series, avisos = {}, []
    for a in activos:
        v = velas(a, ahora)
        series[a] = v["close"]
        if v.attrs.get("aviso"):
            avisos.append(v.attrs["aviso"])
    return pd.DataFrame(series).sort_index(), avisos


def precios_ahora(activos, timeout: float = 10.0) -> dict[str, float]:
    """Ultimo precio de cada ACTIVO/USDT. Los que fallen, no aparecen."""
    out = {}
    for a in activos:
        try:
            r = requests.get(f"{API}/ticker/price", params={"symbol": f"{datos.validar_ticker(a)}USDT"},
                             timeout=timeout)
            r.raise_for_status()
            p = float(r.json()["price"])
            if math.isfinite(p) and p > 0:
                out[a] = p
        except Exception:  # noqa: BLE001 - sin precio de ahora se usa el ultimo cierre
            continue
    return out
