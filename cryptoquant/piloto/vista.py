"""Formatos en espanol y lectura de las tablas que edita el usuario.

Aparte de la app para poder probarlos sin Streamlit.
"""
from __future__ import annotations

import math

import pandas as pd

from ..laboratorio.datos import validar_ticker

NBSP = " "


def num(x: float, dec: int = 0) -> str:
    """1234.5 -> '1 234,5' (espacio fino como separador de miles, coma decimal)."""
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:,.{dec}f}".replace(",", NBSP).replace(".", ",").replace("-", "−")


def pct(x: float, dec: int = 0) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x * 100:.{dec}f}".replace(".", ",").replace("-", "−") + NBSP + "%"


def usdt(x: float, dec: int = 0) -> str:
    n = num(x, dec)
    return n if n == "—" else n + NBSP + "USDT"


def cantidad(u: float) -> str:
    """Unidades de una moneda con los decimales que importan."""
    if u is None or not math.isfinite(u):
        return "—"
    a = abs(u)
    texto = f"{u:,.{2 if a >= 1000 else 4 if a >= 1 else 8}f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")      # 1,2000 -> 1,2; 500,0000 -> 500
    return texto.replace(",", NBSP).replace(".", ",").replace("-", "−")


def precio(p: float) -> str:
    if p is None or not math.isfinite(p):
        return "—"
    return num(p, 2) if abs(p) >= 1 else f"{p:.6g}".replace(".", ",")


def fecha(t) -> str:
    return pd.Timestamp(t).strftime("%d-%m-%Y")


def leer_tabla(tabla: pd.DataFrame, columna: str = "Cantidad") -> dict[str, float]:
    """Filas del editor -> {activo: cantidad}. Suma los repetidos, ignora filas vacias."""
    out: dict[str, float] = {}
    for _, fila in tabla.iterrows():
        activo, cant = fila.get("Activo"), fila.get(columna)
        if activo is None or (isinstance(activo, float) and math.isnan(activo)) or not str(activo).strip():
            if cant is not None and not pd.isna(cant) and float(cant) != 0:
                raise ValueError("hay una fila con cantidad pero sin moneda")
            continue
        a = validar_ticker(str(activo))
        if cant is None or pd.isna(cant):
            raise ValueError(f"{a}: falta la cantidad")
        try:
            c = float(cant)
        except (TypeError, ValueError):
            raise ValueError(f"{a}: {cant!r} no es un número") from None
        if not math.isfinite(c) or c < 0:
            raise ValueError(f"{a}: la cantidad debe ser un número positivo")
        out[a] = out.get(a, 0.0) + c
    return out
