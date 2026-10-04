"""Riesgo del modelo frente a la inversion pasiva.

Responde, con cifras, a la variable dependiente de la tesis (riesgo:
volatilidad, caida maxima, VaR y CVaR al 95%) y a una pregunta previa que la
tesis necesita: si el VaR con que se mide ese riesgo es de fiar.

Tres carteras:
    modelo          las posiciones del backtest walk-forward, dia a dia
    equiponderada   1/N en cada activo del universo (la pasiva de la tesis)
    BTC             todo en bitcoin (la pasiva del inversor tipico)

Las posiciones del modelo se reconstruyen de sus pesos objetivo: el motor
vuelve cada dia a la cartera objetivo vigente (`engine.run`), asi que la
posicion al cierre de t es el ultimo objetivo fijado hasta t.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import config as cfgmod
from ..backtest.engine import WalkForwardBacktest
from ..config import Config
from ..dashboard.state import clean
from . import futuro, pruebas, var


def carpeta() -> Path:
    d = cfgmod.REPORTS_DIR / "riesgo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def posiciones_modelo(targets: pd.DataFrame, index: pd.Index, hasta) -> pd.DataFrame:
    return targets.reindex(index).ffill().loc[targets.index[0]:hasta].fillna(0.0)


@dataclass
class InformeRiesgo:
    fecha_datos: pd.Timestamp
    pronosticos: dict[str, pd.DataFrame]
    validacion: pd.DataFrame
    futuro: pd.DataFrame
    var_hoy: pd.DataFrame
    exposicion_hoy: float
    archivos: list[Path] = field(default_factory=list)


def ejecutar(ohlcv: dict, closes: pd.DataFrame, cfg: Config, alphas=(0.95, 0.99),
             horizontes=(21, 63, 252), n_sim: int = 10_000) -> InformeRiesgo:
    if not all(0.5 < a < 1 for a in alphas):
        raise ValueError("los niveles de confianza deben estar entre 0.5 y 1")
    res = WalkForwardBacktest(cfg).run(ohlcv, closes)
    R = closes.pct_change().fillna(0.0)
    inicio = res.returns.index[0]
    pos = posiciones_modelo(res.targets, R.index, R.index[-1])

    carteras = {"modelo": pos, "equiponderada": pd.Series(1 / closes.shape[1], index=closes.columns)}
    if "BTC" in closes.columns:
        carteras["BTC"] = pd.Series({c: float(c == "BTC") for c in closes.columns})

    pronosticos, filas = {}, []
    for alpha in alphas:
        for nombre, p in carteras.items():
            t = var.pronosticar(R, p, alpha=alpha)
            t = t.loc[t.index >= inicio]  # mismo periodo para las tres carteras
            if alpha == alphas[0]:
                pronosticos[nombre] = t
            ev = pruebas.evaluar(t, alpha)
            ev.insert(0, "alpha", alpha)
            ev.insert(0, "cartera", nombre)
            filas.append(ev.rename_axis("metodo").reset_index())
    validacion = pd.concat(filas, ignore_index=True)

    # --- Riesgo a futuro -------------------------------------------------------
    bench = R.loc[res.returns.index].mean(axis=1)
    frac = float(res.returns.std() / bench.std())
    fut = futuro.riesgo_futuro({"modelo": res.returns, "equiponderada": bench,
                                "equiponderada_igual_vol": bench * frac},
                               horizontes=horizontes, n=n_sim, semilla=cfg.seed)

    # --- VaR de manana con la posicion vigente ---------------------------------
    hoy = []
    for nombre, p in carteras.items():
        w = p.iloc[-1] if isinstance(p, pd.DataFrame) else p
        if w.abs().sum() < 1e-6:
            v = pd.DataFrame({"var": 0.0, "es": 0.0}, index=list(var.METODOS))
        else:
            v = var.var_siguiente(R, w, alphas[0])
        hoy.append(v.assign(cartera=nombre).rename_axis("metodo").reset_index())
    var_hoy = pd.concat(hoy, ignore_index=True)

    inf = InformeRiesgo(R.index[-1], pronosticos, validacion, fut, var_hoy,
                        float(pos.iloc[-1].abs().sum()))
    _exportar(inf, cfg, alphas[0])
    return inf


def _exportar(inf: InformeRiesgo, cfg: Config, alpha: float) -> None:
    d = carpeta()
    for nombre, t in inf.pronosticos.items():
        ruta = d / f"var_{nombre}.csv"
        t.to_csv(ruta)
        inf.archivos.append(ruta)
    for nombre, df in (("validacion_var", inf.validacion), ("riesgo_futuro", inf.futuro),
                       ("var_manana", inf.var_hoy)):
        ruta = d / f"{nombre}.csv"
        df.to_csv(ruta, index=False)
        inf.archivos.append(ruta)

    ultimo_ano = {n: t.iloc[-365:] for n, t in inf.pronosticos.items()}
    estado = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fecha_datos": str(inf.fecha_datos.date()),
        "alpha": alpha,
        "exposicion_hoy": inf.exposicion_hoy,
        "var_manana": inf.var_hoy.to_dict(orient="records"),
        "validacion": inf.validacion.to_dict(orient="records"),
        "futuro": inf.futuro.to_dict(orient="records"),
        "serie": {
            n: {"x": [int(i.timestamp() * 1000) for i in t.index],
                "retorno": [round(float(v), 6) for v in t["retorno"]],
                "var_garch_t": [round(float(v), 6) for v in t["var_garch_t"]],
                "var_historico": [round(float(v), 6) for v in t["var_historico"]]}
            for n, t in ultimo_ano.items()
        },
    }
    # JSON estricto (NaN -> null) y escritura atomica: el panel puede leerlo en
    # cualquier momento y nunca debe ver un fichero a medio escribir.
    ruta = d / "estado.json"
    tmp = ruta.with_name(ruta.name + ".tmp")
    tmp.write_text(json.dumps(clean(estado), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(ruta)
    inf.archivos.append(ruta)


def leer_estado() -> dict | None:
    """Para el panel: lo que dejo la ultima ejecucion de `riesgo`, sin recalcular."""
    ruta = cfgmod.REPORTS_DIR / "riesgo" / "estado.json"
    if not ruta.exists():
        return None
    try:
        estado = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return estado if isinstance(estado, dict) else None
