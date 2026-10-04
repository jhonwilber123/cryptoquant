"""Ejecuta el laboratorio completo y deja tablas (CSV) y graficos (PNG).

Todo va a `reports/laboratorio/`. Los CSV son los mismos que produce la
version en R, con los mismos nombres de columna, para poder contrastar
ambas implementaciones cifra a cifra.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import (aprendizaje, control, cuantitativo, datos, montecarlo, series, variables,
               volatilidad)


@dataclass
class Informe:
    simbolo: str
    origen: str
    velas: pd.DataFrame
    diario: pd.DataFrame
    resultados: dict = field(default_factory=dict)
    archivos: list[Path] = field(default_factory=list)


def _csv(inf: Informe, df: pd.DataFrame, nombre: str, index: bool = True) -> None:
    ruta = datos.carpeta() / f"{nombre}.csv"
    df.to_csv(ruta, index=index)
    inf.archivos.append(ruta)


def _png(inf: Informe, fig, nombre: str) -> None:
    import matplotlib.pyplot as plt

    ruta = datos.carpeta() / f"{nombre}.png"
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    inf.archivos.append(ruta)


def ejecutar(simbolo: str = "BTC", temporalidad: str = "1h", desde: str = "2022-01-01",
             descargar: bool = False, acierto: float = 0.45, riesgo_beneficio: float = 2.0,
             riesgo: float = 0.01, capital: float = 10_000.0, graficos: bool = True,
             semilla: int = 42, objetivo_anual: float = 0.15) -> Informe:
    velas, origen = datos.cargar_velas(simbolo, temporalidad, desde, descargar)
    diario = datos.a_diario(velas) if temporalidad != "1d" else velas
    inf = Informe(datos.par(simbolo), origen, velas, diario)
    inf.archivos.append(datos.ruta_csv(inf.simbolo, temporalidad))
    R = inf.resultados

    # --- Variables -------------------------------------------------------------
    v = variables.preparar_variables(diario)
    _csv(inf, v.rename_axis("fecha"), "variables_diarias")
    ret = v["retorno"].dropna()

    # --- Cuantitativo ----------------------------------------------------------
    R["distribucion"] = cuantitativo.distribucion_retornos(ret)
    R["colas"] = cuantitativo.tabla_colas(ret)
    R["senales"] = cuantitativo.estadistica_senales(v)
    _csv(inf, R["colas"], "colas")
    _csv(inf, R["senales"], "senales", index=False)

    # --- Series de tiempo ------------------------------------------------------
    R["estacionariedad"] = series.precio_frente_a_retorno(diario["close"])
    R["acf"], R["ljung_box"] = series.autocorrelacion(ret)
    ret_intradia = np.log(velas["close"]).diff().dropna()
    R["calendario"] = series.patrones_calendario(ret_intradia)
    R["arima"] = series.arima_basico(ret)
    _csv(inf, R["acf"], "autocorrelacion")
    for k, t in R["calendario"].items():
        _csv(inf, t, f"patron_{k}")

    # --- Volatilidad -----------------------------------------------------------
    R["agrupamiento"] = volatilidad.agrupamiento(ret)
    g = volatilidad.pronostico_garch(ret)
    R["garch"] = g
    R["garch_eval"] = volatilidad.evaluar_garch(ret)
    R["plan"] = volatilidad.plan_operacion(float(diario["close"].iloc[-1]), g["sigma_manana"],
                                           capital, riesgo)
    _csv(inf, pd.DataFrame({"sigma_garch": g["sigma_condicional"],
                            "sigma_movil_20": v["vol_20"]}).rename_axis("fecha"), "volatilidad")

    # --- Machine learning ------------------------------------------------------
    rf = aprendizaje.entrenar_bosque(v, semilla=semilla)
    R["bosque"] = rf
    _csv(inf, rf.metricas, "ml_metricas")
    _csv(inf, rf.importancia, "ml_importancia")

    # --- Monte Carlo -----------------------------------------------------------
    caminos = montecarlo.trayectorias_precio(ret, float(diario["close"].iloc[-1]), semilla=semilla)
    R["mc_precio"] = montecarlo.resumen_trayectorias(caminos)
    curvas = montecarlo.simular_cuenta(acierto, riesgo_beneficio, riesgo, capital=capital,
                                       semilla=semilla)
    R["mc_cuenta"] = montecarlo.resumen_cuenta(curvas, acierto, riesgo_beneficio, riesgo)
    R["mc_parametros"] = {"acierto": acierto, "riesgo_beneficio": riesgo_beneficio,
                          "riesgo": riesgo, "capital": capital}
    _csv(inf, pd.Series({**R["mc_precio"], **R["mc_cuenta"]}, name="valor").to_frame(),
         "montecarlo")

    # --- Control de volatilidad: la idea de la tesis ---------------------------
    ctl = control.control_volatilidad(ret, objetivo_anual)
    R["control"] = control.resumen(ctl, capital=capital)
    R["control_tabla"] = ctl
    _csv(inf, ctl.rename_axis("fecha"), "control_volatilidad")
    _csv(inf, R["control"], "control_resumen")

    if graficos:
        _graficos(inf, v, caminos, curvas)
    return inf


# ------------------------------------------------------------------------------
def _graficos(inf: Informe, v: pd.DataFrame, caminos: np.ndarray, curvas: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = inf.resultados
    ret = v["retorno"].dropna()

    fig, ax = plt.subplots(figsize=(11, 5))
    datos.grafico_velas(inf.diario, ax, ultimas=120, titulo=f"{inf.simbolo} diario")
    _png(inf, fig, "01_velas")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    z = (ret - ret.mean()) / ret.std()
    a1.hist(z, bins=120, density=True, alpha=0.6, label="observado")
    xs = np.linspace(-8, 8, 400)
    a1.plot(xs, stats.norm.pdf(xs), label="normal")
    a1.set_yscale("log")
    a1.set_ylim(1e-5, 1)
    a1.set_title("Retornos estandarizados (escala log)")
    a1.legend(frameon=False)
    stats.probplot(z, dist="norm", plot=a2)
    a2.set_title("QQ frente a la normal")
    _png(inf, fig, "02_distribucion")

    fig, ax = plt.subplots(figsize=(11, 4))
    acf = R["acf"]
    ax.bar(acf.index - 0.2, acf["retorno"], width=0.4, label="retorno")
    ax.bar(acf.index + 0.2, acf["abs_retorno"], width=0.4, label="|retorno|")
    ax.axhspan(-acf["banda_95"].iloc[0], acf["banda_95"].iloc[0], color="grey", alpha=0.2)
    ax.set_title("Autocorrelacion: la direccion no tiene memoria, la magnitud si")
    ax.set_xlabel("rezago (dias)")
    ax.legend(frameon=False)
    _png(inf, fig, "03_autocorrelacion")

    cal = R["calendario"]
    fig, axes = plt.subplots(1, len(cal), figsize=(11, 4))
    for ax, (k, t) in zip(np.atleast_1d(axes), cal.items()):
        ax.bar(range(len(t)), t["volatilidad"])
        ax.set_xticks(range(len(t)), [str(i) for i in t.index], rotation=90 if k == "dia" else 0,
                      fontsize=7)
        ax.set_title(f"Volatilidad por {k} (UTC)")
    _png(inf, fig, "04_calendario")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(R["garch"]["sigma_condicional"] * np.sqrt(365), label="GARCH(1,1)", linewidth=0.9)
    ax.plot(v["vol_20"] * np.sqrt(365), label="desviacion movil 20d", linewidth=0.9, alpha=0.7)
    ax.set_title("Volatilidad anualizada")
    ax.legend(frameon=False)
    _png(inf, fig, "05_garch")

    imp = R["bosque"].importancia
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index[::-1], imp["permutacion_auc"][::-1], xerr=imp["permutacion_sd"][::-1])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title("Importancia por permutacion (caida de AUC en prueba)")
    _png(inf, fig, "06_importancia")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    q = np.percentile(caminos, [5, 25, 50, 75, 95], axis=0)
    t = np.arange(caminos.shape[1])
    a1.fill_between(t, q[0], q[4], alpha=0.2, label="5-95%")
    a1.fill_between(t, q[1], q[3], alpha=0.35, label="25-75%")
    a1.plot(t, q[2], label="mediana")
    a1.set_title("Precio: trayectorias remuestreadas")
    a1.set_xlabel("dias")
    a1.legend(frameon=False)
    for c in curvas[:100]:
        a2.plot(c, linewidth=0.5, alpha=0.3)
    a2.plot(np.median(curvas, axis=0), color="black", linewidth=1.5, label="mediana")
    a2.set_yscale("log")
    a2.set_title("Cuenta: 100 de las simulaciones")
    a2.set_xlabel("operaciones")
    a2.legend(frameon=False)
    _png(inf, fig, "07_montecarlo")

    ctl = R["control_tabla"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[3, 1])
    for col, nombre in (("pasiva", "Comprar y mantener"),
                        ("pasiva_igual_vol", "Comprar y mantener con la misma volatilidad (ex post)"),
                        ("estrategia", "Control de volatilidad")):
        a1.plot((1 + ctl[col]).cumprod(), label=nombre, linewidth=1.1)
    a1.set_yscale("log")
    a1.set_title("Control de volatilidad: el riesgo lo fija usted, no el mercado")
    a1.legend(frameon=False)
    a2.fill_between(ctl.index, ctl["exposicion"], step="pre", alpha=0.4)
    a2.set_ylabel("expuesto")
    a2.set_ylim(0, 1.05)
    _png(inf, fig, "08_control")
