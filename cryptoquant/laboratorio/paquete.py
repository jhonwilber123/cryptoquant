"""Paquete de datos de respaldo para una clase.

Con decenas de alumnos descargando a la vez, desde servidores (Colab, Posit
Cloud) que Binance puede bloquear, la clase no debe depender de ninguna API.
Este modulo deja en una carpeta los CSV de velas ya descargados, listos para
subir a Drive y a Posit Cloud, con una ficha (LEEME.txt) que dice de donde
salen, que periodo cubren y su huella SHA-256 para comprobar que no se han
corrompido al copiarlos.

Formato: fecha (ISO, UTC), open, high, low, close, volume. El mismo que lee
la version en R del laboratorio y el script de Posit Cloud de la clase.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from . import datos


def nombre_csv(simbolo: str, temporalidad: str) -> str:
    return f"{datos.par(simbolo).replace('/', '')}_{temporalidad}.csv"


def preparar(destino: Path, simbolos=("BTC", "ETH", "SOL"),
             temporalidades: dict[str, str] | None = None) -> list[dict]:
    """Descarga y guarda un CSV por simbolo y temporalidad. Devuelve la ficha."""
    temporalidades = temporalidades or {"1d": "2020-01-01", "1h": "2025-01-01"}
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    ficha = []
    for s in simbolos:
        for tf, desde in temporalidades.items():
            df = datos.descargar_velas(s, tf, desde)
            if df.empty:
                raise ValueError(f"{s} {tf}: la descarga llego vacia")
            ruta = destino / nombre_csv(s, tf)
            datos.exportar_csv(df, ruta)
            ficha.append({
                "archivo": ruta.name, "filas": len(df),
                "desde": str(df.index[0].date()), "hasta": str(df.index[-1].date()),
                "fuente": df.attrs.get("fuente", "?"),
                "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest(),
            })
    _escribir_ficha(destino, ficha)
    return ficha


def _escribir_ficha(destino: Path, ficha: list[dict]) -> None:
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lineas = [
        "Datos de respaldo para la clase",
        f"Generados el {ahora} con: python -m cryptoquant paquete",
        "",
        "Velas de Binance (spot, par contra USDT). Columnas: fecha (UTC), open, high,",
        "low, close, volume. La vela en curso no se incluye: solo velas cerradas.",
        "",
        "Uso: subir la carpeta a Google Drive y al proyecto de Posit Cloud. En Colab,",
        "arrastrar el CSV al panel de archivos y leerlo con pandas.read_csv.",
        "",
        f"{'archivo':22s} {'filas':>7s}  {'desde':10s}  {'hasta':10s}  fuente",
    ]
    for f in ficha:
        lineas.append(f"{f['archivo']:22s} {f['filas']:7d}  {f['desde']}  {f['hasta']}  {f['fuente']}")
    lineas += ["", "Huellas SHA-256 (para comprobar que un archivo copiado esta intacto):"]
    lineas += [f"{f['sha256']}  {f['archivo']}" for f in ficha]
    (destino / "LEEME.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
