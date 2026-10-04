"""Contrastes retrospectivos del proyecto de tesis: HE3, HE4 y HE1 retrospectiva.

Por que un modulo aparte y no mas salida en `backtest`: la tesis fija el
periodo de evaluacion y el contraste de cada hipotesis, y eso tiene que quedar
escrito en codigo, no en una sesion de consola. Aqui:

- el periodo se corta en una fecha fija (sin ella, cada dia que pasa alarga el
  historico y los resultados cambian);
- los datos salen solo de la cache local, sin red;
- cada ejecucion guarda la huella de sus resultados y las versiones de las
  librerias. El aprendizaje automatico puede dar decisiones distintas con
  versiones distintas, y la huella es la forma de comprobar si dos equipos
  reproducen lo mismo.

HE1 y HE2 se pre-registraron como contrastes prospectivos (forward test).
HE1 retrospectiva mide lo mismo sobre la simulacion walk-forward: es evidencia
complementaria, no el contraste pre-registrado.

HE5 a HE7 amplian la tesis al nivel doctoral con las capas de riesgo y de
evidencia del sistema (`riesgo`, `laboratorio`). Se formularon despues del
pre-registro y despues de ver resultados exploratorios, y la tesis lo declara:
su evidencia tiene menos jerarquia que la de HE1 a HE4.

    HE5  riesgo de cola: CVaR del modelo frente a la pasiva y frente a la
         pasiva a igual volatilidad (la version exigente)
    HE6  validez de la medicion: el VaR GARCH-t del modelo supera Kupiec y
         la cobertura condicional al 95% y al 99%
    HE7  asimetria de predecibilidad: la volatilidad se preve fuera de
         muestra; la direccion del precio, no

Dos analisis mas, que no cambian el modelo:

- Sensibilidad a dos errores de implementacion hallados en la auditoria del
  01-10-2026 (pesos por unicidad y bloque de calibracion). El modelo
  pre-registrado no se corrige, porque es el que decide en el forward test:
  se mide cuanto cambian las cifras con los errores corregidos
  (`ModeloCorregido`) y la tesis lo declara.
- Comprar y mantener sin rebalanceo, para dejar escrito en que se diferencia
  de la pasiva de la tesis: equiponderada, rebalanceada cada dia al cierre y
  sin costes.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .backtest.engine import BacktestResult, WalkForwardBacktest, buy_and_hold
from .backtest.metrics import summarize
from .config import DATA_DIR, REPORTS_DIR, Config, annualization_factor
from .forward.evaluate import newey_west_tstat, observations_required
from .models.labeling import binary_target, sample_weights_by_uniqueness
from .models.validation import deflated_sharpe_ratio, probabilistic_sharpe_ratio

TOLERANCIA_H1 = 0.40   # +-40% relativo del objetivo: el umbral pre-registrado de H1
VENTANA_H1 = 180       # observaciones que el pre-registro exige para H1
N_ENSAYOS = 12         # configuraciones probadas; la misma cota que usa `backtest`
BLOQUE = 21            # ~un mes de datos diarios por bloque de bootstrap
LIBRERIAS = ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "arch")
SALIDA = REPORTS_DIR / "tesis"


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def recortar(ohlcv: dict[str, pd.DataFrame], closes: pd.DataFrame, hasta: str
             ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Datos hasta `hasta`, inclusive."""
    fin = pd.Timestamp(hasta, tz=closes.index.tz)
    if closes.index[-1] < fin:
        raise ValueError(f"la cache llega al {closes.index[-1].date()}, antes del corte {fin.date()}")
    return {s: d.loc[:fin] for s, d in ohlcv.items()}, closes.loc[:fin]


# --------------------------------------------------------------------------
# Estadisticos
# --------------------------------------------------------------------------
def vol_anual(r: np.ndarray, ann: float) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(ann))


def caida_maxima(r: np.ndarray) -> float:
    """Mayor caida desde un maximo previo (negativa), partiendo de capital 1."""
    eq = np.concatenate([[1.0], np.cumprod(1 + np.asarray(r, float))])
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def indices_bloques(n: int, bloque: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap de bloques moviles: bloques contiguos elegidos al azar.

    Dentro de cada bloque se conserva la dependencia serial (el agrupamiento
    de la volatilidad), que un remuestreo dia a dia destruiria.
    """
    k = int(np.ceil(n / bloque))
    inicios = rng.integers(0, n - bloque + 1, k)
    return (inicios[:, None] + np.arange(bloque)).ravel()[:n]


def bootstrap(x: np.ndarray, estadistico, n_boot: int = 5000, bloque: int = BLOQUE,
              seed: int = 42) -> np.ndarray:
    """Distribucion bootstrap de `estadistico` sobre las filas de `x`.

    Con varias columnas (modelo y referencia) se remuestrean las mismas fechas
    en todas: el bootstrap es pareado y la diferencia conserva la correlacion
    entre las dos series.
    """
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    return np.array([estadistico(x[indices_bloques(len(x), bloque, rng)]) for _ in range(n_boot)])


@dataclass
class Contraste:
    hipotesis: str
    medida: str
    estimacion: float
    ic95_inf: float
    ic95_sup: float
    prob: float            # probabilidad bootstrap o p-valor; ver `prob_tipo`
    prob_tipo: str
    criterio: str
    resultado: str         # 'se cumple' | 'no se cumple' | 'no concluyente'


# --------------------------------------------------------------------------
# Contrastes
# --------------------------------------------------------------------------
def contraste_he1(res: BacktestResult, objetivo: float, ann: float, n_boot: int
                  ) -> tuple[Contraste, pd.DataFrame]:
    """Volatilidad realizada mientras se esta invertido frente al rango objetivo.

    Criterio pre-registrado: la estimacion puntual dentro de +-40% del
    objetivo. Se anade su IC y el mismo calculo en ventanas consecutivas de
    180 dias invertido, que dicen si el control se sostiene en cada tramo del
    ciclo y no solo en promedio.
    """
    invertido = (res.gross_exposure >= 0.01).to_numpy()
    r = res.returns.to_numpy()[invertido]
    fechas = res.returns.index[invertido]
    lo_obj, hi_obj = objetivo * (1 - TOLERANCIA_H1), objetivo * (1 + TOLERANCIA_H1)
    est = vol_anual(r, ann)
    dist = bootstrap(r, lambda x: vol_anual(x, ann), n_boot)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    c = Contraste("HE1 (retrospectiva)", "volatilidad anualizada mientras se esta invertido",
                  est, float(lo), float(hi), float(((dist >= lo_obj) & (dist <= hi_obj)).mean()),
                  "P bootstrap de estar en el rango",
                  f"entre {lo_obj:.1%} y {hi_obj:.1%} (objetivo {objetivo:.0%} +-{TOLERANCIA_H1:.0%})",
                  "se cumple" if lo_obj <= est <= hi_obj else "no se cumple")
    filas = []
    for i in range(0, len(r) - VENTANA_H1 + 1, VENTANA_H1):
        v = vol_anual(r[i:i + VENTANA_H1], ann)
        filas.append({"desde": fechas[i].date(), "hasta": fechas[i + VENTANA_H1 - 1].date(),
                      "volatilidad": v, "dentro_del_rango": lo_obj <= v <= hi_obj})
    return c, pd.DataFrame(filas)


def contraste_he3(modelo: np.ndarray, pasiva: np.ndarray, ann: float, n_boot: int
                  ) -> list[Contraste]:
    """Volatilidad y caida maxima del modelo frente a la cartera pasiva.

    Se contrasta la diferencia (modelo - pasiva) de cada medida, en magnitud:
    negativa significa menos riesgo. Se cumple si el IC95% queda entero por
    debajo de cero.
    """
    x = np.column_stack([modelo, pasiva])
    medidas = [
        ("diferencia de volatilidad anualizada (modelo - pasiva)",
         lambda m: vol_anual(m[:, 0], ann) - vol_anual(m[:, 1], ann)),
        ("diferencia de caida maxima en magnitud (modelo - pasiva)",
         lambda m: abs(caida_maxima(m[:, 0])) - abs(caida_maxima(m[:, 1]))),
    ]
    out = []
    for nombre, fn in medidas:
        dist = bootstrap(x, fn, n_boot)
        lo, hi = np.percentile(dist, [2.5, 97.5])
        out.append(Contraste(
            "HE3", nombre, fn(x), float(lo), float(hi), float((dist >= 0).mean()),
            "P bootstrap de diferencia >= 0", "IC95% entero por debajo de cero",
            "se cumple" if hi < 0 else ("no se cumple" if lo > 0 else "no concluyente")))
    return out


def pasiva_igual_vol(modelo: pd.Series, pasiva: pd.Series) -> tuple[float, pd.Series]:
    """La cartera pasiva escalada a la volatilidad del modelo (el resto, en efectivo).

    Es la comparacion justa: con menos riesgo se captura menos subida, y
    comparar contra la pasiva completa mezcla la ventaja con el nivel de riesgo.
    """
    k = float(modelo.std(ddof=1) / pasiva.std(ddof=1))
    return k, pasiva * k


def contraste_he4(modelo: pd.Series, pasiva: pd.Series, ann: float, n_boot: int
                  ) -> tuple[list[Contraste], dict]:
    """Rentabilidad del modelo frente a la pasiva a igual volatilidad.

    Criterio pre-registrado (H3 del forward test): IC95% por bootstrap de
    bloques que excluya el cero. Se declaro no resoluble en el horizonte de la
    tesis; se informa la estimacion, su incertidumbre y cuantos anos harian
    falta para concluir.
    """
    k, igual = pasiva_igual_vol(modelo, pasiva)
    d = (modelo - igual).to_numpy()
    exceso = float(d.mean() * ann)
    te = float(d.std(ddof=1) * np.sqrt(ann))
    t, p, lag = newey_west_tstat(d)
    dist = bootstrap(d, np.mean, n_boot) * ann
    lo, hi = np.percentile(dist, [2.5, 97.5])
    s = summarize(modelo, ann)
    psr = probabilistic_sharpe_ratio(s["sharpe"], s["n_periods"], 0.0, s["skew"], s["kurtosis"] + 3, ann)
    dsr = deflated_sharpe_ratio(s["sharpe"], s["n_periods"], N_ENSAYOS, s["skew"], s["kurtosis"] + 3, ann)
    req = observations_required(exceso, te, periods_per_year=ann)
    contrastes = [
        Contraste("HE4", "exceso de rentabilidad anual frente a la pasiva a igual volatilidad",
                  exceso, float(lo), float(hi), float(p), f"p-valor Newey-West (t={t:.2f}, {lag} rezagos)",
                  "IC95% por bootstrap de bloques por encima de cero",
                  "se cumple" if lo > 0 else ("no se cumple" if hi < 0 else "no concluyente")),
        Contraste("HE4", "ratio de Sharpe deflactado del modelo", float(dsr), np.nan, np.nan,
                  float(dsr), f"probabilidad (corrige {N_ENSAYOS} ensayos)", ">= 95 %",
                  "se cumple" if dsr >= 0.95 else "no se cumple"),
    ]
    extra = {"fraccion_pasiva": k, "exceso_anual": exceso, "tracking_error": te,
             "p_bootstrap_exceso_positivo": float((dist > 0).mean()),
             "sharpe_probabilistico": psr, "sharpe_deflactado": dsr, "n_ensayos": N_ENSAYOS,
             "ratio_informacion": req["information_ratio"],
             "anos_para_significancia": req["years_significance"],
             "anos_con_potencia_80": req["years_powered"]}
    return contrastes, extra


def resumen(modelo: pd.Series, exposicion: pd.Series, pasiva: pd.Series, objetivo: float,
            ann: float, n_boot: int) -> dict:
    """Las cifras que deciden cada hipotesis, para comparar corridas entre si
    (variantes del modelo, o la misma estrategia con otro codigo o entorno)."""
    s = summarize(modelo, ann)
    res = BacktestResult(equity=(1 + modelo).cumprod(), returns=modelo, weights=pd.DataFrame(),
                         gross_exposure=exposicion, benchmark_returns=pasiva,
                         diagnostics=pd.DataFrame(), turnover=modelo * 0, costs=modelo * 0)
    he1, ventanas = contraste_he1(res, objetivo, ann, n_boot)
    he3 = contraste_he3(modelo.to_numpy(), pasiva.to_numpy(), ann, n_boot)
    he4, extra = contraste_he4(modelo, pasiva, ann, n_boot)
    return {
        "rentabilidad_anual": s["cagr"], "volatilidad": s["volatility"],
        "caida_maxima": s["max_drawdown"], "sharpe": s["sharpe"],
        "capital_final": 10_000 * (1 + s["total_return"]),
        "vol_invertido": he1.estimacion, "he1": he1.resultado,
        "ventanas_he1_en_rango": f"{int(ventanas['dentro_del_rango'].sum())}/{len(ventanas)}",
        "he3_volatilidad": he3[0].resultado, "he3_caida": he3[1].resultado,
        "exceso_igual_vol": extra["exceso_anual"], "exceso_ic95_inf": he4[0].ic95_inf,
        "exceso_ic95_sup": he4[0].ic95_sup, "p_newey_west": he4[0].prob, "he4": he4[0].resultado,
        "sharpe_deflactado": extra["sharpe_deflactado"],
        "anos_para_significancia": extra["anos_para_significancia"],
    }


# --------------------------------------------------------------------------
# HE5: riesgo de cola
# --------------------------------------------------------------------------
ALPHAS_VAR = (0.95, 0.99)
# Uno, tres y doce meses de calendario: los criptoactivos cotizan todos los dias,
# asi que 21/63/252 (la convencion bursatil) serian tres semanas, dos meses y ocho meses.
HORIZONTES_MC = (30, 91, 365)
FRAC_PRUEBA = 0.3               # el tramo fuera de muestra del laboratorio y del filtro de evidencia


def var_cvar(r: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR y CVaR historicos a un dia, como perdidas positivas."""
    r = np.asarray(r, float)
    q = np.quantile(r, 1 - alpha)
    return float(-q), float(-r[r <= q].mean())


def contraste_he5(modelo: np.ndarray, pasiva: np.ndarray, igual: np.ndarray, n_boot: int
                  ) -> tuple[list[Contraste], pd.DataFrame]:
    """CVaR del modelo frente a la pasiva y frente a la pasiva a igual volatilidad.

    Frente a la pasiva completa, menos cola es casi una consecuencia de
    invertir menos. La prueba exigente es a igual volatilidad: si el control
    de volatilidad adelgaza las colas, el CVaR del modelo debe quedar por
    debajo del de la pasiva reducida a su mismo riesgo.
    """
    x = np.column_stack([modelo, pasiva, igual])
    filas, contrastes = [], []
    for alpha in ALPHAS_VAR:
        for i, medida in ((0, "VaR"), (1, "CVaR")):
            fila = {"medida": f"{medida} {alpha:.0%}", "alpha": alpha,
                    **{n: var_cvar(x[:, j], alpha)[i] for j, n in enumerate(("modelo", "pasiva", "pasiva_igual_vol"))}}
            for j, ref in ((1, "pasiva"), (2, "igual")):
                fn = (lambda m, j=j, i=i, a=alpha: var_cvar(m[:, 0], a)[i] - var_cvar(m[:, j], a)[i])
                dist = bootstrap(x, fn, n_boot)
                lo, hi = np.percentile(dist, [2.5, 97.5])
                fila |= {f"dif_{ref}": fn(x), f"dif_{ref}_ic_inf": float(lo), f"dif_{ref}_ic_sup": float(hi),
                         f"dif_{ref}_p_no_menor": float((dist >= 0).mean())}
                if medida == "CVaR" and alpha == ALPHAS_VAR[0]:
                    nombre = "pasiva" if ref == "pasiva" else "pasiva a igual volatilidad"
                    contrastes.append(Contraste(
                        "HE5", f"diferencia de CVaR {alpha:.0%} a un dia (modelo - {nombre})",
                        fn(x), float(lo), float(hi), float((dist >= 0).mean()),
                        "P bootstrap de diferencia >= 0", "IC95% entero por debajo de cero",
                        "se cumple" if hi < 0 else ("no se cumple" if lo > 0 else "no concluyente")))
            filas.append(fila)
    forma = pd.DataFrame({n: {"asimetria": float(pd.Series(x[:, j]).skew()),
                              "curtosis_exceso": float(pd.Series(x[:, j]).kurt())}
                          for j, n in enumerate(("modelo", "pasiva", "pasiva_igual_vol"))})
    tabla = pd.DataFrame(filas).set_index("medida")
    tabla.attrs["forma"] = forma
    return contrastes, tabla


# --------------------------------------------------------------------------
# HE6: validez de la medicion del riesgo (backtest del VaR)
# --------------------------------------------------------------------------
def validar_var(res: BacktestResult, closes: pd.DataFrame
                ) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame], list[Contraste]]:
    """VaR y CVaR a un dia por cuatro metodos, pronosticados sin mirar al futuro
    con las posiciones reales del modelo, y su backtest (Kupiec, Christoffersen).

    Mismas funciones que `python -m cryptoquant riesgo`, pero sobre los datos
    cortados de la tesis. El metodo principal, fijado antes de mirar la tabla,
    es GARCH-t: es el unico de los cuatro con colas gruesas y volatilidad que
    reacciona al dia anterior.
    """
    from .riesgo import pruebas, var
    from .riesgo.informe import posiciones_modelo

    R = closes.pct_change().fillna(0.0)
    inicio = res.returns.index[0]
    carteras = {"modelo": posiciones_modelo(res.targets, R.index, R.index[-1]),
                "equiponderada": pd.Series(1 / closes.shape[1], index=closes.columns)}
    if "BTC" in closes.columns:
        carteras["BTC"] = pd.Series({c: float(c == "BTC") for c in closes.columns})
    pronosticos, filas = {}, []
    for alpha in ALPHAS_VAR:
        for nombre, p in carteras.items():
            t = var.pronosticar(R, p, alpha=alpha)
            t = t.loc[t.index >= inicio]
            pronosticos[(nombre, alpha)] = t
            ev = pruebas.evaluar(t, alpha)
            ev.insert(0, "alpha", alpha)
            ev.insert(0, "cartera", nombre)
            filas.append(ev.rename_axis("metodo").reset_index())
    validacion = pd.concat(filas, ignore_index=True)
    contrastes = []
    for alpha in ALPHAS_VAR:
        f = validacion[(validacion["cartera"] == "modelo") & (validacion["alpha"] == alpha)
                       & (validacion["metodo"] == "garch_t")].iloc[0]
        contrastes.append(Contraste(
            "HE6", f"VaR GARCH-t al {alpha:.0%}: tasa de excepciones (esperada {1 - alpha:.0%})",
            float(f["tasa"]), np.nan, np.nan, float(f["p_cobertura_condicional"]),
            f"p cobertura condicional (Kupiec p={f['p_kupiec']:.3f}, independencia p={f['p_independencia']:.3f})",
            f"Kupiec y cobertura condicional p > 0,05 con >= {pruebas.MIN_OBS} dias invertido",
            "se cumple" if f["aceptado"] else "no se cumple"))
    return validacion, pronosticos, contrastes


def riesgo_a_futuro(series: dict[str, pd.Series], semilla: int, n: int = 10_000
                    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Monte Carlo por bloques: probabilidades de perder y de caer a 1, 3 y 12
    meses, y la distribucion de la caida maxima a 12 meses (para la figura)."""
    from .laboratorio.montecarlo import trayectorias_precio
    from .riesgo import futuro

    tabla = futuro.riesgo_futuro(series, horizontes=HORIZONTES_MC, n=n, semilla=semilla)
    caidas = {}
    for nombre, r in series.items():
        c = trayectorias_precio(np.log1p(r.dropna()), 1.0, dias=HORIZONTES_MC[-1], n=n, bloque=10,
                                semilla=semilla)
        caidas[nombre] = (c / np.maximum.accumulate(c, axis=1) - 1).min(axis=1)
    return tabla, caidas


# --------------------------------------------------------------------------
# HE7: asimetria de predecibilidad
# --------------------------------------------------------------------------
def _qlike(h: pd.Series, real2: pd.Series) -> pd.Series:
    return np.log(h) + real2 / h


def volatilidad_fuera_de_muestra(ret: pd.Series, frac_prueba: float = FRAC_PRUEBA
                                 ) -> tuple[dict, pd.Series]:
    """Prevision a un dia de la varianza: GARCH(1,1)-t frente a varianza constante.

    La hipotesis nula de "la volatilidad no se puede prever" es la varianza
    constante del entrenamiento. La desviacion movil de 20 dias se informa
    como segunda referencia. Parametros congelados al final del
    entrenamiento; la perdida es QLIKE (Patton, 2011).
    """
    from .laboratorio.volatilidad import ajustar_garch

    r = ret.dropna() * 100
    n = int(len(r) * (1 - frac_prueba))
    res = ajustar_garch(ret, ultima_obs=n)
    h = res.forecast(horizon=1, start=n - 1, reindex=False).variance.iloc[:, 0]
    prueba = r.index[n:]
    h_garch = pd.Series(h.to_numpy()[:-1], index=prueba)   # la prevision hecha en t es para t+1
    h_const = pd.Series(float(r.iloc[:n].var()), index=prueba)
    h_movil = (r.rolling(20).std() ** 2).shift(1).loc[prueba]
    real2 = (r.loc[prueba] - r.iloc[:n].mean()) ** 2
    d = _qlike(h_garch, real2) - _qlike(h_const, real2)
    t, p, _ = newey_west_tstat(d.to_numpy())
    p_ = res.params
    return {"n_prueba": len(prueba), "qlike_garch": float(_qlike(h_garch, real2).mean()),
            "qlike_constante": float(_qlike(h_const, real2).mean()),
            "qlike_movil": float(_qlike(h_movil, real2).mean()),
            "dm_t": t, "dm_p": float(stats.norm.cdf(t)),   # unilateral: GARCH mejor
            "persistencia": float(p_["alpha[1]"] + p_["beta[1]"])}, d


def direccion_arima(ret: pd.Series, frac_prueba: float = FRAC_PRUEBA, max_p: int = 2,
                    max_q: int = 2) -> dict:
    """ARIMA(p,0,q) por AIC en el entrenamiento, prevision a un paso con
    parametros congelados, frente a prever siempre 0 (Diebold-Mariano)."""
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    r = pd.Series(ret.dropna().to_numpy() * 100)
    n = int(len(r) * (1 - frac_prueba))
    mejor = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                ajuste = ARIMA(r.iloc[:n], order=(p, 0, q), trend="c").fit()
                if mejor is None or ajuste.aic < mejor[1].aic:
                    mejor = ((p, 0, q), ajuste)
        orden, ajuste = mejor
        pred = ajuste.apply(r).fittedvalues.iloc[n:]
    real = r.iloc[n:]
    d = ((real - pred) ** 2 - real ** 2).to_numpy()
    t, _, _ = newey_west_tstat(d)
    return {"orden": "({},{},{})".format(*orden), "r2_fuera_muestra": float(1 - ((real - pred) ** 2).sum() / (real ** 2).sum()),
            "dm_t": t, "dm_p": float(stats.norm.cdf(t)),
            "acierto_signo": float((np.sign(pred) == np.sign(real)).mean())}


def predecibilidad(ohlcv: dict[str, pd.DataFrame], semilla: int = 42
                   ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Por activo: volatilidad (GARCH), direccion (ARIMA y bosque aleatorio), y
    las variables del laboratorio que la version en R debe reproducir."""
    from .laboratorio import aprendizaje, variables

    filas, perdidas, exportar = {}, {}, []
    for activo, velas in ohlcv.items():
        v = variables.preparar_variables(velas)
        ret = v["retorno"].dropna()
        vol, perdidas[activo] = volatilidad_fuera_de_muestra(ret)
        ar = direccion_arima(ret)
        rf = aprendizaje.entrenar_bosque(v, frac_entrenamiento=1 - FRAC_PRUEBA, semilla=semilla)
        m = rf.metricas
        filas[activo] = {
            **{f"vol_{k}": x for k, x in vol.items()},
            **{f"arima_{k}": x for k, x in ar.items()},
            "rf_acierto": float(m.loc["random_forest", "acierto"]),
            "rf_auc": float(m.loc["random_forest", "auc"]),
            "rf_mejor_base": float(m.drop(index="random_forest")["acierto"].max()),
            "rf_p": rf.p_valor_vs_base,
            "rf_n_prueba": len(rf.prueba),
        }
        cols = [*variables.VARIABLES_MODELO, "sma_20", "macd", "bb_sup", "sube"]
        exportar.append(v[cols].rename_axis("fecha").reset_index().assign(activo=activo))
    conjunta = pd.concat(perdidas, axis=1).mean(axis=1)
    return pd.DataFrame(filas).T, conjunta, pd.concat(exportar, ignore_index=True)


def filtro_senales(ohlcv: dict[str, pd.DataFrame], horizonte: int = 5) -> pd.DataFrame:
    """Las ocho senales tecnicas del laboratorio por el filtro de evidencia.

    Sin anotarlas en el registro (`registrar=False`): la tesis no debe
    endurecer el umbral de las ideas futuras por repetir una evaluacion que el
    registro ya contiene con los mismos parametros.
    """
    from .laboratorio import cuantitativo, variables
    from .riesgo import evidencia

    muestra = next(iter(ohlcv.values()))
    nombres = list(cuantitativo.senales(variables.preparar_variables(muestra)).columns)
    filas = []
    for nombre in nombres:
        gen = (lambda n: lambda df: cuantitativo.senales(variables.preparar_variables(df))[n])(nombre)
        v = evidencia.evaluar_senal(nombre, gen, ohlcv, horizonte=horizonte, frac_prueba=FRAC_PRUEBA,
                                    registrar=False)
        filas.append({"senal": nombre, "causal": v.causal, "eventos": v.eventos, "acierto": v.acierto,
                      "acierto_base": v.acierto_base, "p_valor": v.p_valor, "umbral": v.umbral,
                      "ideas_evaluadas": v.ideas_evaluadas, "retorno_neto_medio": v.retorno_neto_medio,
                      "pasa": v.pasa, "desde": v.desde, "motivos": "; ".join(v.motivos)})
    return pd.DataFrame(filas)


def contraste_he7(pred: pd.DataFrame, conjunta: pd.Series, senales: pd.DataFrame) -> list[Contraste]:
    """(a) la volatilidad se preve: GARCH mejora a la varianza constante en la
    perdida QLIKE media de los activos (Diebold-Mariano con Newey-West,
    unilateral, 5%).
    (b) la direccion no: ninguno de los contrastes de direccion (senales,
    ARIMA y bosque aleatorio en cada activo) supera a su base con correccion
    de Bonferroni sobre toda la familia."""
    t, _, lag = newey_west_tstat(conjunta.to_numpy())
    media = float(conjunta.mean())
    se = abs(media / t) if t else np.nan
    p_a = float(stats.norm.cdf(t))
    ps = np.concatenate([senales["p_valor"].to_numpy(), pred["arima_dm_p"].to_numpy(),
                         pred["rf_p"].to_numpy()]).astype(float)
    umbral = 0.05 / len(ps)
    return [
        Contraste("HE7", "diferencia de perdida QLIKE (GARCH - varianza constante), media de los activos",
                  media, media - 1.96 * se, media + 1.96 * se, p_a,
                  f"p unilateral Diebold-Mariano (t={t:.2f}, {lag} rezagos)", "p < 0,05 con diferencia negativa",
                  "se cumple" if p_a < 0.05 and media < 0 else "no se cumple"),
        Contraste("HE7", f"menor p-valor de los {len(ps)} contrastes de direccion", float(np.nanmin(ps)),
                  np.nan, np.nan, float(np.nanmin(ps)), "p-valor sin corregir",
                  f">= {umbral:.5f} (0,05/{len(ps)}, Bonferroni)",
                  "se cumple" if np.nanmin(ps) >= umbral else "no se cumple"),
    ]


def referencia_guardada(closes: pd.DataFrame, carpeta) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """Retornos, exposicion y pasiva de una corrida anterior guardada por `backtest`."""
    rp, wp = carpeta / "backtest_returns.csv", carpeta / "backtest_weights.csv"
    if not (rp.exists() and wp.exists()):
        return None
    r = pd.read_csv(rp, index_col=0, parse_dates=True)["return"]
    w = pd.read_csv(wp, index_col=0, parse_dates=True)
    r.index = r.index.tz_convert(closes.index.tz) if r.index.tz else r.index.tz_localize(closes.index.tz)
    w.index = r.index
    pasiva = closes.pct_change().fillna(0.0).loc[r.index].mean(axis=1)
    return r, w.sum(axis=1), pasiva


# --------------------------------------------------------------------------
# Figuras (sin titulo dentro de la imagen: en APA el titulo va en el documento)
# --------------------------------------------------------------------------
def _etiquetas_finales(ax, puntos: list[tuple], minimo_pt: float = 11.0) -> None:
    """Etiqueta el final de cada serie separando las que caerian encima."""
    import matplotlib.dates as mdates

    from .reporting.report import INK_SECONDARY

    ax.figure.canvas.draw()
    pix = [(ax.transData.transform((mdates.date2num(x), y))[1], x, y, t) for x, y, t in puntos]
    pix.sort()
    ajustadas, previa = [], -np.inf
    minimo_px = minimo_pt * ax.figure.dpi / 72
    for py, x, y, t in pix:
        nueva = max(py, previa + minimo_px)
        ajustadas.append((nueva - py, x, y, t))
        previa = nueva
    for dpx, x, y, t in ajustadas:
        ax.annotate(t, xy=(x, y), xytext=(6, dpx * 72 / ax.figure.dpi), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=9, va="center")


def _pct(v, decimales=0) -> str:
    return f"{v * 100:.{decimales}f} %".replace(".", ",")


def figuras(res: BacktestResult, igual: pd.Series, k: float, objetivo: float, ann: float,
            capital: float) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from .reporting.report import INK_MUTED, SERIES, SURFACE, _style_axes

    def lienzo(alto=3.6):
        fig, ax = plt.subplots(figsize=(7.5, alto), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        _style_axes(ax, "")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        return fig, ax

    def guardar(fig, nombre):
        fig.tight_layout()
        ruta = SALIDA / nombre
        fig.savefig(ruta, facecolor=SURFACE)
        plt.close(fig)
        return ruta

    rutas = []
    series = [(res.returns, SERIES[0], "-", "Modelo"),
              (res.benchmark_returns, SERIES[1], "-", "Pasiva equiponderada"),
              (igual, SERIES[2], "--", f"Pasiva al {_pct(k, 1)} + efectivo")]
    miles = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))

    # 1. Capital (escala logaritmica: compara variaciones relativas).
    fig, ax = lienzo(3.8)
    finales = []
    for s, color, estilo, nombre in series:
        eq = capital * (1 + s).cumprod()
        ax.plot(eq.index, eq.to_numpy(), color=color, linestyle=estilo, linewidth=2, label=nombre)
        finales.append((eq.index[-1], float(eq.iloc[-1]), f"{eq.iloc[-1]:,.0f}".replace(",", " ")))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(miles)
    ax.yaxis.set_minor_formatter(miles)
    ax.tick_params(axis="y", which="minor", labelsize=7, colors=INK_MUTED)
    ax.set_ylabel("Capital (USDT, escala logarítmica)", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    _etiquetas_finales(ax, finales)
    rutas.append(guardar(fig, "figura_capital.png"))

    # 2. Caida desde el maximo previo.
    fig, ax = lienzo()
    finales = []
    for s, color, estilo, nombre in series[:2]:
        eq = pd.concat([pd.Series([1.0]), (1 + s).cumprod()], ignore_index=True)
        dd = (eq / eq.cummax() - 1).iloc[1:].set_axis(s.index)
        ax.plot(dd.index, dd.to_numpy(), color=color, linestyle=estilo, linewidth=1.6,
                label=f"{nombre} (máx. {_pct(dd.min(), 1)})")
        finales.append((dd.index[-1], float(dd.iloc[-1]), _pct(dd.iloc[-1], 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct(v)))
    ax.set_ylabel("Caída desde el máximo previo", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8)
    _etiquetas_finales(ax, finales)
    rutas.append(guardar(fig, "figura_caidas.png"))

    # 3. Exposicion del modelo (una sola serie: el titulo del documento la nombra).
    fig, ax = lienzo(3.0)
    ex = res.gross_exposure
    ax.fill_between(ex.index, 0, ex.to_numpy(), color=SERIES[0], alpha=0.18, linewidth=0)
    ax.plot(ex.index, ex.to_numpy(), color=SERIES[0], linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct(v)))
    ax.set_ylabel("Capital invertido", fontsize=9)
    rutas.append(guardar(fig, "figura_exposicion.png"))

    # 4. Volatilidad movil mientras se esta invertido frente al rango de HE1.
    fig, ax = lienzo(3.2)
    inv = res.returns[res.gross_exposure >= 0.01]
    vm = inv.rolling(VENTANA_H1).std(ddof=1) * np.sqrt(ann)
    lo, hi = objetivo * (1 - TOLERANCIA_H1), objetivo * (1 + TOLERANCIA_H1)
    ax.axhspan(lo, hi, color="#c3c2b7", alpha=0.35, linewidth=0)
    ax.axhline(objetivo, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.plot(vm.index, vm.to_numpy(), color=SERIES[0], linewidth=1.8)
    borde = ax.get_yaxis_transform()        # x en fraccion del eje, y en datos
    ax.text(0.99, hi, f"rango de HE1: {_pct(lo)} a {_pct(hi)}", transform=borde, ha="right",
            color=INK_MUTED, fontsize=8, va="bottom")
    ax.text(0.99, objetivo, f"objetivo {_pct(objetivo)}", transform=borde, ha="right",
            color=INK_MUTED, fontsize=8, va="bottom")
    ax.set_ylim(0, max(0.30, float(np.nanmax(vm.to_numpy())) * 1.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct(v)))
    ax.set_ylabel(f"Volatilidad anualizada\n({VENTANA_H1} días invertido)", fontsize=9)
    rutas.append(guardar(fig, "figura_volatilidad.png"))
    return rutas


NOMBRES_SERIE = {"modelo": "Modelo", "pasiva": "Pasiva equiponderada",
                 "pasiva_igual_vol": "Pasiva a igual volatilidad"}


def figuras_ampliadas(pron: pd.DataFrame, caidas: dict[str, np.ndarray], pred: pd.DataFrame) -> list:
    """VaR frente a retornos, caidas simuladas a 12 meses y asimetria de predecibilidad."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from .reporting.report import INK_MUTED, SERIES, SURFACE, _style_axes

    pct = FuncFormatter(lambda v, _: _pct(v))
    rutas = []

    def guardar(fig, nombre):
        fig.tight_layout()
        ruta = SALIDA / nombre
        fig.savefig(ruta, facecolor=SURFACE)
        plt.close(fig)
        rutas.append(ruta)

    # 5. Retorno del modelo frente al VaR GARCH-t previsto la vispera.
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, "")
    exc = (-pron["retorno"] > pron["var_garch_t"]) & (pron["exposicion"] >= 0.01)
    ax.vlines(pron.index, 0, pron["retorno"], color="#b9b8b0", linewidth=0.6)
    ax.vlines(pron.index[exc], 0, pron["retorno"][exc], color="#d1495b", linewidth=0.9,
              label=f"Excepciones ({int(exc.sum())} días)")
    ax.plot(pron.index, -pron["var_garch_t"], color=SERIES[0], linewidth=0.9, label="−VaR GARCH-t al 95 %")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("Retorno diario del modelo", fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    guardar(fig, "figura_var.png")

    # 6. Distribucion de la caida maxima a 12 meses (Monte Carlo).
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, "")
    for (nombre, dd), color, estilo in zip(caidas.items(), (SERIES[0], SERIES[1], SERIES[2]), ("-", "-", "--")):
        x = np.sort(-dd)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, linestyle=estilo, linewidth=1.8,
                label=f"{NOMBRES_SERIE.get(nombre, nombre)} (mediana {_pct(np.median(x), 1)})")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Caída máxima en los próximos 12 meses", fontsize=9)
    ax.set_ylabel("Probabilidad acumulada", fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    guardar(fig, "figura_montecarlo.png")

    # 7. Volatilidad frente a direccion, activo por activo.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    orden = list(pred.index)[::-1]
    y = np.arange(len(orden))
    mejora = (pred["vol_qlike_constante"] - pred["vol_qlike_garch"]).loc[orden]
    ventaja = (pred["rf_acierto"] - pred["rf_mejor_base"]).loc[orden] * 100
    for ax, valores, color, titulo, etiqueta in (
            (a1, mejora, SERIES[0], "Volatilidad", "Reducción de la pérdida QLIKE\n(GARCH frente a varianza constante)"),
            (a2, ventaja, SERIES[1], "Dirección", "Acierto del bosque aleatorio\nmenos la mejor línea base (pp)")):
        _style_axes(ax, "")
        ax.barh(y, valores.to_numpy(), color=color, height=0.6)
        ax.axvline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_title(titulo, fontsize=10, loc="left")
        ax.set_xlabel(etiqueta, fontsize=8)
    a1.set_yticks(y, orden, fontsize=8)
    lim = max(abs(ventaja).max(), 1) * 1.15
    a2.set_xlim(-lim, lim)
    guardar(fig, "figura_predecibilidad.png")
    return rutas


def figura_modelo() -> Path:
    """Modelo teorico de la investigacion: de los datos a la decision y a su evaluacion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from .reporting.report import INK_MUTED, SURFACE

    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def caja(x, y, w, h, titulo, texto, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    facecolor=color, edgecolor="#6b6a65", linewidth=0.8))
        ax.text(x + w / 2, y + h - 2.6, titulo, ha="center", va="top", fontsize=8.2, fontweight="bold",
                color="#2b2a27")
        ax.text(x + w / 2, y + h - 7.4, texto, ha="center", va="top", fontsize=7, color="#3d3c38",
                linespacing=1.35)

    def flecha(x0, y0, x1, y1, estilo="-", texto=None, dx=0.0, dy=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     color="#55544f", linewidth=1.1, linestyle=estilo))
        if texto:
            ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, texto, fontsize=6.6, color=INK_MUTED,
                    ha="center", va="center", style="italic")

    azul, naranja, verde, gris = "#dce8f7", "#fbe3d6", "#d9f1e6", "#eeede8"
    caja(2, 74, 25, 20, "Datos de mercado", "Velas diarias de Binance\n8 criptoactivos\nsolo información pasada", gris)
    caja(36, 83, 25, 13, "Econometría financiera", "GJR-GARCH(1,1), Ledoit-Wolf\nmomentum ajustado por riesgo", azul)
    caja(36, 63, 25, 15, "Aprendizaje automático", "Triple barrera, validación\npurgada, probabilidad\ncalibrada", azul)
    caja(73, 74, 25, 20, "Control de volatilidad", "E = min(1, σ*/σ̂) · c · d · r\nconvicción, caída y régimen\nsolo pueden reducirla", naranja)
    caja(73, 47, 25, 17, "Cartera", "Paridad de riesgo\njerárquica (HRP)\npeso máximo 30 %", naranja)
    caja(36, 40, 25, 15, "Decisión diaria", "Fracción invertida y\npesos por activo, neta\nde costes", naranja)
    caja(2, 40, 25, 22, "Riesgo de la inversión", "Volatilidad, caída máxima\nVaR y CVaR, riesgo a\n1, 3 y 12 meses\n(variable dependiente)", verde)
    caja(2, 3, 96, 27, "Capa de evidencia verificable", "", gris)
    capa = [("Walk-forward sin\ninformación futura", 4.5), ("Bootstrap de bloques\nNewey-West, DSR", 20.5),
            ("Pre-registro y\ndiario sellado", 36.5), ("Backtest del VaR\nKupiec, Christoffersen", 52.5),
            ("Filtro de evidencia\nBonferroni acumulado", 68.5), ("Huella SHA-256\ny réplica en R", 84.5)]
    for texto, x in capa:
        ax.add_patch(FancyBboxPatch((x, 7), 13.5, 13, boxstyle="round,pad=0.3,rounding_size=1",
                                    facecolor="white", edgecolor="#9d9c96", linewidth=0.6))
        ax.text(x + 6.75, 13.5, texto, ha="center", va="center", fontsize=6.4, color="#3d3c38", linespacing=1.3)

    flecha(27.8, 88, 35.2, 89)
    flecha(27.8, 80, 35.2, 71)
    flecha(61.8, 89.5, 72.2, 86, texto="volatilidad\nprevista\n(predecible)", dy=6.5)
    flecha(61.8, 70, 72.2, 78, estilo="--", texto="dirección\n(poco\npredecible)", dy=-7.5)
    flecha(85.5, 73.2, 85.5, 64.8)
    flecha(72.2, 55, 61.8, 49)
    flecha(35.2, 47.5, 27.8, 50)
    flecha(14.5, 39.2, 14.5, 30.8)
    ruta = SALIDA / "figura_modelo.png"
    fig.tight_layout()
    fig.savefig(ruta, facecolor=SURFACE)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------
# Tablas descriptivas del universo
# --------------------------------------------------------------------------
def tabla_activos(closes: pd.DataFrame, cfg: Config, ann: float) -> pd.DataFrame:
    """Descriptivos, contrastes econometricos, riesgo y GJR-GARCH por activo."""
    from .econometrics.diagnostics import diagnose_universe
    from .econometrics.risk import compute_risk
    from .econometrics.volatility import forecast_universe

    r = closes.pct_change().dropna()
    desc = pd.DataFrame({
        "rent_media_anual": r.mean() * ann,
        "volatilidad_anual": r.std(ddof=1) * np.sqrt(ann),
        "asimetria": r.skew(),
        "curtosis_exceso": r.kurt(),
        "peor_dia": r.min(),
        "mejor_dia": r.max(),
    })
    diag = diagnose_universe(closes, ann).drop(columns=["annual_vol"], errors="ignore")
    riesgo = pd.DataFrame({s: {k: v for k, v in asdict(compute_risk(r[s], ann, cfg.risk.var_confidence)).items()
                               if isinstance(v, (int, float, np.floating))}
                           for s in closes.columns}).T.add_prefix("riesgo_")
    garch = forecast_universe(np.log(closes).diff().dropna(), ann).add_prefix("garch_")
    return pd.concat([desc, diag, riesgo, garch], axis=1)


# --------------------------------------------------------------------------
# Historia larga: todos los ciclos de mercado disponibles
# --------------------------------------------------------------------------
# La evaluacion principal empieza cuando cotizan los ocho activos (2020) y
# cubre un solo ciclo. La historia larga pone a prueba lo mismo en los ciclos
# anteriores: Binance desde que lista cada activo (BTC y ETH en 2017) y
# Bitstamp, BTC/USD desde agosto de 2011, casi toda la vida de mercado de
# Bitcoin. Se guarda aparte de data/cache/, que alimenta el forward test: una
# descarga para la tesis nunca debe poder alterar el experimento prospectivo.
CARPETA_HISTORIA = DATA_DIR / "historico"
HALVINGS = ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")
UNIVERSO_2018 = ("BTC", "ETH", "BNB", "XRP", "ADA")   # los que cotizan en Binance desde mayo de 2018
_DIA_MS = 86_400_000


def fuentes_historia(simbolos) -> list[tuple[str, str, str]]:
    return [("bitstamp", "BTC/USD", "2011-01-01"),
            *[("binance", f"{s}/USDT", "2017-01-01") for s in simbolos]]


def ruta_historia(exchange: str, par: str) -> Path:
    return CARPETA_HISTORIA / f"{exchange}_{par.replace('/', '-')}_1d.parquet"


def descargar_historia(simbolos) -> dict[str, tuple[str, str, int]]:
    """Velas diarias desde el primer dia disponible en cada fuente.

    Pagina hasta la ultima vela cerrada. Una descarga mas corta que la que ya
    hay en disco (corte de red a medias) no la sustituye.
    """
    import time

    import ccxt

    from .data.sources import drop_incomplete_bars

    CARPETA_HISTORIA.mkdir(parents=True, exist_ok=True)
    ahora = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    out = {}
    for exid, par, desde in fuentes_historia(simbolos):
        ex = getattr(ccxt, exid)({"enableRateLimit": True, "timeout": 30_000})
        cursor, filas = int(pd.Timestamp(desde, tz="UTC").timestamp() * 1000), []
        while cursor < ahora:
            lote = ex.fetch_ohlcv(par, "1d", since=cursor, limit=1000)
            if not lote:
                break
            filas.extend(lote)
            siguiente = lote[-1][0] + _DIA_MS
            if siguiente <= cursor:
                break
            cursor = siguiente
            time.sleep(ex.rateLimit / 1000)
        df = pd.DataFrame(filas, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts")
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True).rename("date")
        df = drop_incomplete_bars(df.astype(float), "1d")
        ruta = ruta_historia(exid, par)
        if ruta.exists() and len(pd.read_parquet(ruta)) > len(df):
            df = pd.read_parquet(ruta)
        else:
            df.to_parquet(ruta)
        out[f"{exid} {par}"] = (str(df.index[0].date()), str(df.index[-1].date()), len(df))
    return out


def cargar_historia(simbolos, hasta: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]] | None:
    """(BTC/USD de Bitstamp, {simbolo: velas de Binance}) hasta el corte, o None."""
    rb = ruta_historia("bitstamp", "BTC/USD")
    rutas = {s: ruta_historia("binance", f"{s}/USDT") for s in simbolos}
    if not rb.exists() or not all(r.exists() for r in rutas.values()):
        return None
    fin = pd.Timestamp(hasta, tz="UTC")
    btc = pd.read_parquet(rb).loc[:fin]
    # Dias sin negociacion en los primeros meses de Bitstamp: se arrastra el
    # ultimo cierre y el volumen es cero (no se inventan precios).
    dias = pd.date_range(btc.index[0], fin, freq="D", tz="UTC")
    btc = btc.reindex(dias)
    btc["close"] = btc["close"].ffill()
    for c in ("open", "high", "low"):
        btc[c] = btc[c].fillna(btc["close"])
    btc["volume"] = btc["volume"].fillna(0.0)
    return btc.rename_axis("date"), {s: pd.read_parquet(r).loc[:fin] for s, r in rutas.items()}


def _descriptivos(r: pd.Series, ann: float) -> dict:
    v, c = var_cvar(r.to_numpy(), 0.95)
    return {"inicio": str(r.index[0].date()), "fin": str(r.index[-1].date()), "dias": len(r),
            "rent_anual": float((1 + r).prod() ** (ann / len(r)) - 1),
            "volatilidad": vol_anual(r.to_numpy(), ann), "caida_maxima": caida_maxima(r.to_numpy()),
            "peor_dia": float(r.min()), "curtosis_exceso": float(r.kurt()), "var95": v, "cvar95": c}


def tabla_historia(btc: pd.DataFrame, binance: dict[str, pd.DataFrame], ann: float) -> pd.DataFrame:
    filas = {"BTC (Bitstamp, USD)": _descriptivos(btc["close"].pct_change().dropna(), ann)}
    for s, d in binance.items():
        filas[f"{s} (Binance, USDT)"] = _descriptivos(d["close"].pct_change().dropna(), ann)
    return pd.DataFrame(filas).T


def tramos_halving(indice: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    cortes = [indice[0], *[pd.Timestamp(h, tz="UTC") for h in HALVINGS], indice[-1] + pd.Timedelta(days=1)]
    return [(f"{a.year}–{(b - pd.Timedelta(days=1)).year}", a, b) for a, b in zip(cortes[:-1], cortes[1:])]


def ciclos_btc(btc: pd.DataFrame, ann: float, semilla: int = 42) -> pd.DataFrame:
    """Cada ciclo entre halvings: riesgo y, desde el segundo, predecibilidad.

    La prediccion de cada ciclo se entrena con toda la historia anterior a su
    inicio y se evalua dentro del ciclo: nunca ve el ciclo que predice.
    """
    from .laboratorio import aprendizaje, variables

    r = btc["close"].pct_change().dropna()
    v_all = variables.preparar_variables(btc)
    filas = []
    for i, (nombre, a, b) in enumerate(tramos_halving(r.index)):
        fila = {"ciclo": nombre, **_descriptivos(r.loc[a:b - pd.Timedelta(days=1)], ann)}
        if i > 0:
            hasta = b - pd.Timedelta(days=1)
            ret = v_all["retorno"].loc[:hasta].dropna()
            frac = float((ret.index >= a).mean())
            vol, _ = volatilidad_fuera_de_muestra(ret, frac)
            ar = direccion_arima(ret, frac)
            v = v_all.loc[:hasta].copy()
            v.loc[v.index[-1], ["retorno_futuro", "sube"]] = np.nan   # el objetivo del ultimo dia mira fuera del ciclo
            X, _ = variables.matriz_modelo(v)
            rf = aprendizaje.entrenar_bosque(v, frac_entrenamiento=float((X.index < a).mean()), semilla=semilla)
            m = rf.metricas
            fila |= {"vol_qlike_garch": vol["qlike_garch"], "vol_qlike_constante": vol["qlike_constante"],
                     "vol_dm_t": vol["dm_t"], "vol_dm_p": vol["dm_p"],
                     "arima_orden": ar["orden"], "arima_r2": ar["r2_fuera_muestra"], "arima_dm_p": ar["dm_p"],
                     "rf_acierto": float(m.loc["random_forest", "acierto"]),
                     "rf_mejor_base": float(m.drop(index="random_forest")["acierto"].max()),
                     "rf_p": rf.p_valor_vs_base}
        filas.append(fila)
    return pd.DataFrame(filas).set_index("ciclo")


def _panel(ohlcv: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Como `load_universe`: desde el primer dia en que cotizan todos."""
    closes = pd.DataFrame({s: d["close"] for s, d in ohlcv.items()})
    inicio = closes.apply(lambda c: c.first_valid_index()).max()
    closes = closes.loc[inicio:].ffill().dropna()
    return {s: d.reindex(closes.index).ffill() for s, d in ohlcv.items()}, closes


def evaluacion_extendida(ohlcv: dict[str, pd.DataFrame], cfg: Config, ann: float, n_boot: int,
                         peso_maximo: float | None = None) -> tuple[dict, BacktestResult]:
    """El modelo completo, con los mismos parametros, sobre otro universo y periodo.

    Con un solo activo el tope de concentracion del 30 % dejaria la exposicion
    en el 30 % como maximo: ahi no protege la diversificacion, solo impide
    invertir, por eso se levanta (`peso_maximo=1`) y se declara.
    """
    import copy

    c = copy.deepcopy(cfg)
    if peso_maximo is not None:
        c.risk.max_weight_per_asset = peso_maximo
    o, closes = _panel(ohlcv)
    res = WalkForwardBacktest(c, portfolio_method="hrp", use_ml=True).run(o, closes)
    modelo, pasiva = res.returns, res.benchmark_returns
    _, igual = pasiva_igual_vol(modelo, pasiva)
    fila = resumen(modelo, res.gross_exposure, pasiva, c.risk.target_volatility, ann, n_boot)
    he5, _ = contraste_he5(modelo.to_numpy(), pasiva.to_numpy(), igual.to_numpy(), n_boot)
    sp = summarize(pasiva, ann)
    fila |= {"inicio": str(modelo.index[0].date()), "fin": str(modelo.index[-1].date()), "dias": len(modelo),
             "activos": ", ".join(closes.columns), "exposicion_media": float(res.gross_exposure.mean()),
             "pasiva_rentabilidad": sp["cagr"], "pasiva_volatilidad": sp["volatility"],
             "pasiva_caida": sp["max_drawdown"],
             "cvar_dif_pasiva": he5[0].estimacion, "he5_pasiva": he5[0].resultado,
             "cvar_dif_igual": he5[1].estimacion, "cvar_dif_igual_ic_inf": he5[1].ic95_inf,
             "cvar_dif_igual_ic_sup": he5[1].ic95_sup, "he5_igual": he5[1].resultado,
             "peso_maximo": c.risk.max_weight_per_asset}
    return fila, res


def ciclos_modelo(res: BacktestResult, ann: float) -> pd.DataFrame:
    """El modelo frente a su pasiva en cada ciclo entre halvings."""
    filas = []
    for nombre, a, b in tramos_halving(res.returns.index):
        m = res.returns.loc[a:b - pd.Timedelta(days=1)]
        if len(m) < 30:
            continue
        p = res.benchmark_returns.loc[m.index]
        filas.append({"ciclo": nombre, "inicio": str(m.index[0].date()), "fin": str(m.index[-1].date()),
                      "dias": len(m), "exposicion_media": float(res.gross_exposure.loc[m.index].mean()),
                      **{f"modelo_{k}": v for k, v in _descriptivos(m, ann).items()
                         if k in ("rent_anual", "volatilidad", "caida_maxima", "cvar95")},
                      **{f"pasiva_{k}": v for k, v in _descriptivos(p, ann).items()
                         if k in ("rent_anual", "volatilidad", "caida_maxima", "cvar95")}})
    return pd.DataFrame(filas).set_index("ciclo")


def figura_historia(btc: pd.DataFrame, res_btc: BacktestResult) -> Path:
    """Precio de BTC desde 2011 con los halvings, y la caida del modelo frente a BTC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from .reporting.report import INK_MUTED, SERIES, SURFACE, _style_axes

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.5), dpi=200, sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        _style_axes(ax, "")
        for h in HALVINGS:
            ax.axvline(pd.Timestamp(h, tz="UTC"), color=INK_MUTED, linewidth=0.8, linestyle=":")
    a1.plot(btc.index, btc["close"], color=SERIES[1], linewidth=1.1)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    a1.set_ylabel("BTC/USD (escala log.)", fontsize=9)
    for h in HALVINGS:
        a1.text(pd.Timestamp(h, tz="UTC"), a1.get_ylim()[0] * 1.5, " halving", fontsize=7, color=INK_MUTED,
                rotation=90, va="bottom")
    for s, color, nombre in ((res_btc.benchmark_returns, SERIES[1], "BTC (comprar y mantener)"),
                             (res_btc.returns, SERIES[0], "Modelo aplicado a BTC")):
        eq = pd.concat([pd.Series([1.0]), (1 + s).cumprod()], ignore_index=True)
        dd = (eq / eq.cummax() - 1).iloc[1:].set_axis(s.index)
        a2.plot(dd.index, dd.to_numpy(), color=color, linewidth=1.3, label=f"{nombre} (máx. {_pct(dd.min(), 1)})")
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct(v)))
    a2.set_ylabel("Caída desde el máximo", fontsize=9)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8)
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ruta = SALIDA / "figura_historia.png"
    fig.tight_layout()
    fig.savefig(ruta, facecolor=SURFACE)
    plt.close(fig)
    return ruta


def historia_larga(cfg: Config, hasta: str, ann: float, n_boot: int) -> dict | None:
    datos = cargar_historia(cfg.data.symbols, hasta)
    if datos is None:
        return None
    btc, binance = datos
    tabla_historia(btc, binance, ann).to_csv(SALIDA / "historia_activos.csv")
    ciclos_btc(btc, ann, cfg.seed).to_csv(SALIDA / "historia_ciclos_btc.csv")
    filas = {}
    filas["5 activos desde 2018"], _ = evaluacion_extendida({s: binance[s] for s in UNIVERSO_2018}, cfg, ann, n_boot)
    filas["Solo BTC desde 2011"], res_btc = evaluacion_extendida({"BTC": btc}, cfg, ann, n_boot, peso_maximo=1.0)
    pd.DataFrame(filas).to_csv(SALIDA / "robustez_temporal.csv")
    ciclos_modelo(res_btc, ann).to_csv(SALIDA / "historia_ciclos_modelo.csv")
    figura_historia(btc, res_btc)
    return filas


# --------------------------------------------------------------------------
# Sensibilidad a dos errores de implementacion (el modelo NO se cambia)
# --------------------------------------------------------------------------
class ModeloCorregido(WalkForwardBacktest):
    """El modelo con dos errores de implementacion corregidos, solo para medir su efecto.

    El modelo pre-registrado, el que decide en el forward test, sigue siendo
    `WalkForwardBacktest` sin cambios. Aqui se replica su
    `_pooled_training_set` con dos correcciones independientes:

    - `pesos_relativos`: `exit_bar` guarda la posicion de salida en la serie
      completa, pero `sample_weights_by_uniqueness` la lee como posicion
      dentro de la ventana de entrenamiento. Mientras la ventana empieza en la
      primera barra coinciden; cuando se desliza, los pesos dejan de medir la
      unicidad y sobreponderan las observaciones mas antiguas.
    - `calibracion_temporal`: el panel se apila activo por activo, asi que el
      "ultimo 20 %" que `ProbabilityModel` reserva para calibrar son los
      ultimos activos de la lista en las mismas fechas, no el tramo mas
      reciente. Ordenar las filas por fecha antes del corte lo convierte en
      el bloque temporal que el diseno prometia.
    """

    def __init__(self, cfg: Config, pesos_relativos: bool = True, calibracion_temporal: bool = False,
                 **kwargs):
        super().__init__(cfg, **kwargs)
        self.pesos_relativos = pesos_relativos
        self.calibracion_temporal = calibracion_temporal

    def panel(self, feats, labels, upto: int, index: pd.Index):
        """Como `WalkForwardBacktest._pooled_training_set`, pero cada fila conserva su fecha."""
        purge = self.cfg.signal.label_horizon + 5
        hi = max(0, upto - purge)
        lo = max(0, hi - self.cfg.backtest.train_window)
        if hi - lo < 150:
            return None, None, None
        window = index[lo:hi]
        X_parts, y_parts, w_parts = [], [], []
        for sym in feats:
            f = feats[sym].loc[window]
            lab = labels[sym].loc[window]
            if self.pesos_relativos:
                lab = lab.assign(exit_bar=lab["exit_bar"] - lo)
            y = binary_target(lab)
            w = sample_weights_by_uniqueness(lab)
            mask = y.notna() & f.notna().all(axis=1)
            if mask.sum() < 30:
                continue
            X_parts.append(f.loc[mask])
            y_parts.append(y.loc[mask])
            w_parts.append(w.loc[mask].fillna(1.0))
        if not X_parts:
            return None, None, None
        X, y, w = pd.concat(X_parts), pd.concat(y_parts), pd.concat(w_parts)
        if self.calibracion_temporal:
            orden = np.argsort(X.index.to_numpy(), kind="stable")
            X, y, w = X.iloc[orden], y.iloc[orden], w.iloc[orden]
        return X, y, w

    def _pooled_training_set(self, feats, labels, upto: int, index: pd.Index):
        X, y, w = self.panel(feats, labels, upto, index)
        if X is None:
            return None, None, None
        return X.reset_index(drop=True), y.reset_index(drop=True), w.reset_index(drop=True)


def diagnostico_implementacion(ohlcv: dict[str, pd.DataFrame], closes: pd.DataFrame,
                               cfg: Config) -> dict:
    """Las dos pruebas de que los errores existen, en la ultima ventana de entrenamiento.

    Pesos: correlacion entre los pesos que usa el modelo y los correctos, y
    peso medio por quintil de antiguedad (1 = lo mas antiguo). Calibracion:
    que parte de las filas de cada activo cae en el 20 % que se reserva.
    """
    bt = WalkForwardBacktest(cfg)
    feats, labels, _ = bt.prepare(ohlcv, closes)
    index = closes.index
    upto = len(index) - 1
    hi = upto - (cfg.signal.label_horizon + 5)
    lo = max(0, hi - cfg.backtest.train_window)
    if hi - lo < 150:
        return {}
    window = index[lo:hi]

    corr, q_act, q_cor = [], [], []
    for sym in feats:
        lab = labels[sym].loc[window]
        w_act = sample_weights_by_uniqueness(lab)
        w_cor = sample_weights_by_uniqueness(lab.assign(exit_bar=lab["exit_bar"] - lo))
        ok = (w_act.notna() & w_cor.notna()).to_numpy()
        a, c = w_act.to_numpy()[ok], w_cor.to_numpy()[ok]
        corr.append(float(np.corrcoef(a, c)[0, 1]))
        partes = np.array_split(np.arange(len(a)), 5)
        q_act.append([float(a[p].mean()) for p in partes])
        q_cor.append([float(c[p].mean()) for p in partes])

    # Composicion del bloque de calibracion: el panel tal como lo usa el modelo,
    # apilado activo por activo con las mismas filas validas.
    X, _, _ = ModeloCorregido(cfg, pesos_relativos=False).panel(feats, labels, upto, index)
    if X is None:
        return {}
    corte = int(len(X) * 0.8)
    reservado, pos = {}, 0
    for sym in feats:
        n = int((feats[sym].loc[window].notna().all(axis=1)
                 & binary_target(labels[sym].loc[window]).notna()).sum())
        if n < 30:
            continue
        reservado[sym] = round(max(0, pos + n - max(pos, corte)) / n, 4)
        pos += n
    return {
        "ventana": [str(window[0].date()), str(window[-1].date())],
        "pesos_correlacion_media": float(np.mean(corr)),
        "pesos_correlacion_min": float(np.min(corr)),
        "pesos_quintiles_modelo": [float(v) for v in np.mean(q_act, axis=0)],
        "pesos_quintiles_correctos": [float(v) for v in np.mean(q_cor, axis=0)],
        "calibracion_parte_reservada": reservado,
    }


def sensibilidad_implementacion(ohlcv, closes, cfg: Config, modelo_actual: dict, objetivo: float,
                                ann: float, n_boot: int, metodo: str = "hrp") -> pd.DataFrame:
    """Las cifras de las hipotesis con cada correccion, junto al modelo pre-registrado."""
    columnas = {"Modelo pre-registrado": modelo_actual}
    for nombre, calibracion in (("Pesos por unicidad corregidos", False),
                                ("Pesos y calibración corregidos", True)):
        rv = ModeloCorregido(cfg, pesos_relativos=True, calibracion_temporal=calibracion,
                             portfolio_method=metodo, use_ml=True).run(ohlcv, closes)
        columnas[nombre] = {**resumen(rv.returns, rv.gross_exposure, rv.benchmark_returns, objetivo, ann,
                                      n_boot),
                            "dias_invertido": int((rv.gross_exposure >= 0.01).sum())}
    return pd.DataFrame(columnas)


def comprar_y_mantener(closes: pd.DataFrame, desde) -> pd.Series:
    """Comprar a partes iguales el primer dia y no volver a tocar la cartera.

    No es la pasiva de la tesis: `benchmark_returns` del motor es la media de
    los rendimientos diarios, es decir, una cartera equiponderada que se
    rebalancea cada dia al cierre y sin costes. Esta es comprar y mantener en
    sentido estricto, para que la tesis diga en que se diferencian.
    """
    return buy_and_hold(closes.loc[desde:]).pct_change().dropna()


# --------------------------------------------------------------------------
# Ejecucion
# --------------------------------------------------------------------------
def entorno() -> dict:
    versiones = {}
    for lib in LIBRERIAS:
        try:
            versiones[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versiones[lib] = "no instalada"
    return {"python": platform.python_version(), "sistema": platform.platform(), **versiones}


def huella(r: pd.Series) -> str:
    """SHA-256 de los retornos del modelo redondeados: igual huella, mismas decisiones."""
    texto = "\n".join(f"{d.date()} {v:.10f}" for d, v in r.items())
    return hashlib.sha256(texto.encode()).hexdigest()


def ejecutar(ohlcv: dict[str, pd.DataFrame], closes: pd.DataFrame, cfg: Config, hasta: str,
             n_boot: int = 5000, metodo: str = "hrp") -> dict:
    ann = annualization_factor(cfg.data.timeframe)
    ohlcv, closes = recortar(ohlcv, closes, hasta)
    res = WalkForwardBacktest(cfg, portfolio_method=metodo, use_ml=True).run(ohlcv, closes)
    modelo, pasiva = res.returns, res.benchmark_returns
    k, igual = pasiva_igual_vol(modelo, pasiva)

    capital = cfg.backtest.initial_capital
    metricas = {}
    for nombre, serie in (("modelo", modelo), ("pasiva equiponderada", pasiva),
                          (f"pasiva al {k:.1%} + efectivo", igual)):
        s = summarize(serie, ann)
        metricas[nombre] = {"rentabilidad_anual": s["cagr"], "volatilidad": s["volatility"],
                            "caida_maxima": s["max_drawdown"], "sharpe": s["sharpe"],
                            "sortino": s["sortino"], "calmar": s["calmar"],
                            "capital_final": capital * (1 + s["total_return"])}
    metricas = pd.DataFrame(metricas)

    he1, ventanas = contraste_he1(res, cfg.risk.target_volatility, ann, n_boot)
    he3 = contraste_he3(modelo.to_numpy(), pasiva.to_numpy(), ann, n_boot)
    he4, he4_extra = contraste_he4(modelo, pasiva, ann, n_boot)
    contrastes = pd.DataFrame([asdict(c) for c in [he1, *he3, *he4]])

    SALIDA.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(SALIDA / "metricas.csv")
    contrastes.to_csv(SALIDA / "contrastes.csv", index=False)
    ventanas.to_csv(SALIDA / "he1_ventanas.csv", index=False)
    pd.DataFrame({"modelo": modelo, "pasiva": pasiva, "pasiva_igual_vol": igual,
                  "exposicion": res.gross_exposure, "coste": res.costs}).to_csv(SALIDA / "retornos.csv")
    activos = tabla_activos(closes, cfg, ann)
    activos.to_csv(SALIDA / "activos.csv")

    # Que aporta cada componente: la misma simulacion sin aprendizaje
    # automatico, y con pesos iguales en lugar de HRP.
    objetivo = cfg.risk.target_volatility
    variantes = {"Modelo completo (HRP + ML)": resumen(modelo, res.gross_exposure, pasiva, objetivo, ann, n_boot)}
    for nombre, met, ml in (("Sin aprendizaje automático", metodo, False),
                            ("Pesos iguales + control de riesgo", "equal", True)):
        rv = WalkForwardBacktest(cfg, portfolio_method=met, use_ml=ml).run(ohlcv, closes)
        variantes[nombre] = resumen(rv.returns, rv.gross_exposure, rv.benchmark_returns, objetivo, ann, n_boot)
    variantes = pd.DataFrame(variantes)
    variantes.to_csv(SALIDA / "variantes.csv")

    # Sensibilidad a dos errores de implementacion. El modelo de arriba es el
    # pre-registrado y no se toca; aqui solo se mide cuanto cambian las cifras.
    actual = {**variantes["Modelo completo (HRP + ML)"].to_dict(),
              "dias_invertido": int((res.gross_exposure >= 0.01).sum())}
    sensibilidad = sensibilidad_implementacion(ohlcv, closes, cfg, actual, objetivo, ann, n_boot, metodo)
    sensibilidad.to_csv(SALIDA / "sensibilidad_implementacion.csv")
    diagnostico = diagnostico_implementacion(ohlcv, closes, cfg)

    # La pasiva de la tesis frente a comprar y mantener en sentido estricto.
    s_cym = summarize(comprar_y_mantener(closes, res.equity.index[0]), ann)
    pasiva_def = {
        "definicion": ("equiponderada entre los activos del universo, rebalanceada cada dia al cierre "
                       "y sin costes de transaccion"),
        "comprar_y_mantener": {"rentabilidad_anual": s_cym["cagr"], "volatilidad": s_cym["volatility"],
                               "caida_maxima": s_cym["max_drawdown"], "sharpe": s_cym["sharpe"]},
    }

    # Robustez: la corrida del prototipo (codigo anterior a la enmienda E1).
    robustez = None
    ref = referencia_guardada(closes, SALIDA / "referencia_2026-09-11")
    if ref is not None:
        robustez = pd.DataFrame({"Código actual": variantes.iloc[:, 0],
                                 "Prototipo (11-09-2026)": resumen(*ref[:2], ref[2], objetivo, ann, n_boot)})
        robustez.to_csv(SALIDA / "robustez.csv")

    figuras(res, igual, k, objetivo, ann, capital)

    # Ampliacion doctoral (HE5 a HE7): riesgo de cola, validez del VaR y
    # asimetria de predecibilidad, sobre los mismos datos cortados.
    he5, cola = contraste_he5(modelo.to_numpy(), pasiva.to_numpy(), igual.to_numpy(), n_boot)
    cola.to_csv(SALIDA / "riesgo_cola.csv")
    cola.attrs["forma"].to_csv(SALIDA / "forma_retornos.csv")
    validacion, pronosticos, he6 = validar_var(res, closes)
    validacion.to_csv(SALIDA / "validacion_var.csv", index=False)
    for (nombre, alpha), t in pronosticos.items():
        t.to_csv(SALIDA / f"var_{nombre}_{alpha * 100:.0f}.csv")
    futuro, caidas = riesgo_a_futuro({"modelo": modelo, "pasiva": pasiva, "pasiva_igual_vol": igual}, cfg.seed)
    futuro.to_csv(SALIDA / "riesgo_futuro.csv", index=False)
    pred, conjunta, variables_py = predecibilidad(ohlcv, cfg.seed)
    pred.to_csv(SALIDA / "predecibilidad.csv")
    senales = filtro_senales(ohlcv)
    senales.to_csv(SALIDA / "senales.csv", index=False)
    he7 = contraste_he7(pred, conjunta, senales)
    ampliacion = pd.DataFrame([asdict(c) for c in [*he5, *he6, *he7]])
    ampliacion.to_csv(SALIDA / "contrastes_ampliacion.csv", index=False)
    # Lo que la version en R (laboratorio/r/tesis.R) recalcula por su cuenta.
    pd.concat([d.rename_axis("fecha").reset_index().assign(activo=a) for a, d in ohlcv.items()],
              ignore_index=True).to_csv(SALIDA / "velas_tesis.csv", index=False)
    variables_py.to_csv(SALIDA / "variables_tesis.csv", index=False)
    figuras_ampliadas(pronosticos[("modelo", ALPHAS_VAR[0])], caidas, pred)
    figura_modelo()
    # Otros ciclos: solo si se descargo la historia larga (--descargar-historia).
    historia = historia_larga(cfg, hasta, ann, n_boot)

    resultado = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corte": hasta,
        "periodo": [str(modelo.index[0].date()), str(modelo.index[-1].date())],
        "observaciones": int(len(modelo)),
        "datos_desde": str(closes.index[0].date()),
        "exposicion_media": float(res.gross_exposure.mean()),
        "dias_invertido": int((res.gross_exposure >= 0.01).sum()),
        "costes_totales": float(res.costs.sum()),
        "rotacion_media": float(res.turnover.mean()),
        "reentrenamientos": len(res.model_reports),
        "bootstrap": {"replicas": n_boot, "bloque": BLOQUE, "semilla": 42},
        "he4": he4_extra,
        "pasiva": pasiva_def,
        "implementacion": diagnostico,
        "huella_retornos": huella(modelo),
        "entorno": entorno(),
    }
    (SALIDA / "resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2, default=float),
                                           encoding="utf-8")
    return {"metricas": metricas, "contrastes": contrastes, "ventanas": ventanas,
            "activos": activos, "variantes": variantes, "robustez": robustez,
            "sensibilidad": sensibilidad, "comprar_y_mantener": s_cym,
            "ampliacion": ampliacion, "riesgo_cola": cola, "validacion_var": validacion,
            "riesgo_futuro": futuro, "predecibilidad": pred, "senales": senales,
            "historia": historia, "resultado": resultado, "backtest": res}
