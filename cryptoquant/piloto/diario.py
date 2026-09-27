"""Lo que el piloto recuerda: la cartera, los ajustes y una foto por cambio.

Vive en `data/piloto/` (o en la carpeta de la variable CRYPTOQUANT_PILOTO) y
NO va a git: son datos personales. Cada escritura es atomica (archivo
temporal y reemplazo) para que un corte no deje un JSON a medias.

Basta una foto cada vez que cambian las tenencias (compra, venta, ingreso o
retiro): entre fotos, `motor.valor_cuenta` valora las mismas unidades con los
cierres diarios, asi que la caida de la cuenta se sigue midiendo cada dia.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from .. import config as cfgmod
from ..laboratorio.datos import validar_ticker

CARTERA = "cartera.json"
FOTOS = "fotos.jsonl"


def carpeta() -> Path:
    # Se resuelve en cada llamada para que los tests puedan redirigirla.
    d = Path(os.environ.get("CRYPTOQUANT_PILOTO") or cfgmod.DATA_DIR / "piloto")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _escribir(ruta: Path, texto: str) -> None:
    tmp = ruta.with_name(ruta.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(texto)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ruta)


def _limpias(tenencias: dict) -> dict[str, float]:
    if not isinstance(tenencias, dict):
        raise ValueError("las tenencias deben ser pares moneda: cantidad")
    out: dict[str, float] = {}
    for a, v in tenencias.items():
        a = validar_ticker(str(a))
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError(f"{a}: la cantidad debe ser un numero")
        try:
            v = float(v)
        except ValueError:
            raise ValueError(f"{a}: {v!r} no es una cantidad") from None
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{a}: la cantidad debe ser un numero finito y no negativo")
        out[a] = out.get(a, 0.0) + v
    return out


def leer_cartera() -> tuple[dict, str | None]:
    """-> ({'tenencias', 'mezcla', 'ajustes'} o {} si no hay, aviso).

    Un archivo danado no bloquea la app: se aparta con otro nombre (no se
    borra, por si se quiere recuperar a mano) y se empieza de cero.
    """
    ruta = carpeta() / CARTERA
    if not ruta.exists():
        return {}, None
    try:
        d = json.loads(ruta.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("no es un objeto JSON")
        d["tenencias"] = _limpias(d.get("tenencias") or {})
        if d.get("mezcla") is not None:
            d["mezcla"] = _limpias(d["mezcla"])
        if d.get("ajustes") is not None and not isinstance(d["ajustes"], dict):
            raise ValueError("ajustes no validos")
        return d, None
    except (ValueError, UnicodeDecodeError) as exc:
        aparte = ruta.with_name(f"cartera.danada-{datetime.now():%Y%m%d-%H%M%S}.json")
        os.replace(ruta, aparte)
        return {}, (f"{CARTERA} estaba danado ({str(exc)[:80]}); se aparto como {aparte.name} "
                    "y se empieza de cero.")


def guardar_cartera(tenencias: dict, mezcla: dict | None, ajustes: dict) -> None:
    d = {"tenencias": _limpias(tenencias), "mezcla": _limpias(mezcla) if mezcla else None,
         "ajustes": ajustes, "actualizada_utc": _ahora()}
    _escribir(carpeta() / CARTERA, json.dumps(d, ensure_ascii=False, indent=1))


def anotar_foto(tenencias: dict, precios: dict, nota: str = "") -> dict:
    px = {validar_ticker(str(a)): float(p) for a, p in precios.items()}
    if not all(math.isfinite(p) and p > 0 for p in px.values()):
        raise ValueError("los precios de la foto deben ser numeros positivos")
    foto = {"fecha_utc": _ahora(), "tenencias": _limpias(tenencias), "precios": px, "nota": nota}
    ruta = carpeta() / FOTOS
    # Si una escritura anterior se corto a media linea, la nueva no debe pegarse a ella.
    suelta = ruta.exists() and ruta.stat().st_size > 0 and not ruta.read_bytes().endswith(b"\n")
    with ruta.open("a", encoding="utf-8", newline="\n") as f:
        f.write(("\n" if suelta else "") + json.dumps(foto, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return foto


def leer_fotos() -> tuple[list[dict], list[str]]:
    """-> (fotos validas, avisos). Una linea danada (p. ej. un corte a mitad de
    escritura) se salta con un aviso: no debe impedir ver la cartera."""
    ruta = carpeta() / FOTOS
    if not ruta.exists():
        return [], []
    fotos, malas = [], []
    for i, linea in enumerate(ruta.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not linea.strip():
            continue
        try:
            f = json.loads(linea)
            f["tenencias"] = _limpias(f["tenencias"])
            fecha = datetime.fromisoformat(f["fecha_utc"])
            if fecha.tzinfo is None:
                raise ValueError("fecha sin zona horaria")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
            malas.append(i)
            continue
        fotos.append(f)
    avisos = [f"{FOTOS}: se ignoran {len(malas)} linea(s) danadas ({', '.join(map(str, malas[:5]))})"] if malas else []
    return fotos, avisos


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
