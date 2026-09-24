"""Evaluacion del forward test.

Reconstruye el resultado realizado a partir del diario de decisiones y de los
precios posteriores, y contrasta las hipotesis pre-registradas.

Nota sobre que puede y que no puede concluir esto: la volatilidad se estima con
mucha menos informacion que la media. Por eso H1 (el control de riesgo) se
resuelve en meses y H3 (la ventaja) no se resuelve nunca en la practica. El
informe lo dice de forma explicita en lugar de dejar que el numero parezca mas
concluyente de lo que es.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ForwardResult:
    returns: pd.Series               # retornos realizados de la estrategia
    benchmark: pd.Series             # buy & hold equiponderado, mismos tramos
    benchmark_scaled: pd.Series      # el anterior, ajustado a igual volatilidad
    exposure: pd.Series
    costs: pd.Series
    scale_factor: float
    n_gaps: int
    span_days: int
    notes: list[str] = field(default_factory=list)
    # Duracion de cada tramo en dias: 1 normalmente, mas si hubo dias sin anotar.
    days: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def annualised_volatility(returns: pd.Series, days: pd.Series,
                          periods_per_year: float = 365.0) -> float:
    """Volatilidad anualizada con tramos de distinta duracion (en dias).

    Un tramo de 3 dias tiene el triple de varianza que uno de 1. Tratarlos a
    todos como diarios (std * sqrt(365)) infla la cifra: con el patron real del
    diario (huecos de fin de semana) daba un 19.8% donde la correcta era 15.5%.
    Aqui cada tramo se normaliza por su duracion (minimos cuadrados ponderados,
    con la deriva estimada por dia). Con todos los tramos de un dia se reduce
    exactamente a la desviacion tipica habitual.
    """
    r = np.asarray(returns, dtype=float)
    d = np.asarray(days, dtype=float)
    if len(r) < 2 or len(d) != len(r) or (d <= 0).any():
        return np.nan
    drift = r.sum() / d.sum()
    var_day = float(((r - drift * d) ** 2 / d).sum() / (len(r) - 1))
    return float(np.sqrt(var_day * periods_per_year))


def realised_returns(records: list[dict], prices_after: pd.DataFrame,
                     fee: float, slippage: float) -> ForwardResult:
    """Calcula el resultado real de las decisiones registradas.

    Args:
        records: anotaciones del diario, en orden.
        prices_after: panel de cierres que cubre al menos hasta la ultima
            fecha de decision (para poder cerrar el ultimo tramo).
        fee, slippage: mismos costes que usa el backtest, para que la
            comparacion sea homogenea.
    """
    if len(records) < 2:
        empty = pd.Series(dtype=float)
        return ForwardResult(empty, empty, empty, empty, empty, 1.0, 0, 0,
                             ["hacen falta al menos 2 anotaciones para medir un tramo"])

    idx, strat, bench, expo, cost_l, days = [], [], [], [], [], []
    notes: list[str] = []
    prev_w: dict[str, float] = {}   # cartera al final del tramo anterior, ya derivada

    for a, b in zip(records, records[1:]):
        d0, d1 = pd.Timestamp(a["decision_date"]), pd.Timestamp(b["decision_date"])
        w = a["weights"]
        # Precios al inicio del tramo: los que se registraron ese dia.
        p0 = a["prices"]
        # Precios al cierre del tramo: los registrados en la anotacion siguiente.
        p1 = b["prices"]

        common = [s for s in w if s in p0 and s in p1 and p0[s] > 0]
        missing = [s for s in w if s not in common and w.get(s, 0) > 1e-6]
        if missing:
            notes.append(f"{d1.date()}: sin precio de cierre para {missing}; "
                         "esos pesos se tratan como efectivo en ese tramo")

        r_assets = {s: p1[s] / p0[s] - 1.0 for s in common}
        gross_ret = sum(w[s] * r_assets[s] for s in common)

        # Coste: rotacion respecto a la cartera que habia, derivada por precio.
        # Un dia de "mantener" tambien cuesta algo: volver a los pesos objetivo
        # despues de que los precios los muevan. El backtest lo cobra igual.
        universe = set(w) | set(prev_w)
        turnover = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in universe)
        cost = turnover * (fee + slippage)
        grown = {s: w[s] * (1 + r_assets.get(s, 0.0)) for s in w}
        prev_w = {s: v / (1 + gross_ret) for s, v in grown.items()} if gross_ret > -1 else grown

        # Referencia: cesta equiponderada sobre los mismos activos y tramo.
        bench_ret = float(np.mean([r_assets[s] for s in common])) if common else 0.0

        idx.append(d1)
        strat.append(gross_ret - cost)
        bench.append(bench_ret)
        expo.append(float(a["gross_exposure"]))
        cost_l.append(cost)
        days.append(max((d1 - d0).days, 1))

    index = pd.Index(idx, name="date")
    s = pd.Series(strat, index=index)
    b = pd.Series(bench, index=index)

    sd_s = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    sd_b = float(b.std(ddof=1)) if len(b) > 1 else 0.0
    scale = (sd_s / sd_b) if sd_b > 0 else 0.0

    return ForwardResult(
        returns=s, benchmark=b, benchmark_scaled=b * scale,
        exposure=pd.Series(expo, index=index), costs=pd.Series(cost_l, index=index),
        scale_factor=scale, n_gaps=len(s),
        # Desde la primera decision hasta la ultima, no desde el final del
        # primer tramo: con un solo tramo eso daba 0 dias en vez de los reales.
        span_days=int((pd.Timestamp(records[-1]["decision_date"])
                       - pd.Timestamp(records[0]["decision_date"])).days),
        notes=notes,
        days=pd.Series(days, index=index, dtype=float),
    )


def portfolio_state_at(records: list[dict], decision_date: str, prices: dict[str, float],
                       fee: float, slippage: float) -> tuple[dict[str, float], float]:
    """Pesos derivados y drawdown al cierre de `decision_date`, antes de decidir.

    Incluye el tramo desde la ultima anotacion hasta hoy: el backtest decide
    con el capital y los pesos de esa misma barra, ya movidos por el precio.
    """
    if not records:
        return {}, 0.0
    probe = {"decision_date": decision_date, "prices": prices, "weights": {},
             "gross_exposure": 0.0}
    res = realised_returns(records + [probe], pd.DataFrame(), fee, slippage)
    eq = (1 + res.returns).cumprod()
    # El capital empieza en 1: si el primer tramo ya pierde, eso es drawdown.
    dd = float(eq.iloc[-1] / max(1.0, float(eq.max())) - 1.0) if len(eq) else 0.0

    last = records[-1]
    w, p0 = last["weights"], last["prices"]
    r = {s: prices[s] / p0[s] - 1.0 for s in w
         if s in p0 and s in prices and p0[s] > 0 and np.isfinite(prices[s])}
    gross = sum(w[s] * r[s] for s in r)
    drifted = {s: w[s] * (1 + r.get(s, 0.0)) / (1 + gross) for s in w} if gross > -1 else dict(w)
    return drifted, dd


# --------------------------------------------------------------------------
# Contrastes
# --------------------------------------------------------------------------
def newey_west_tstat(x: np.ndarray) -> tuple[float, float, int]:
    """t robusto a autocorrelacion y heterocedasticidad (Newey-West).

    Los retornos diarios no son independientes; usar el error estandar simple
    infla el t y hace parecer significativo lo que no lo es.
    """
    n = len(x)
    if n < 10:
        return np.nan, np.nan, 0
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xm = x - x.mean()
    lrv = (xm @ xm) / n
    for l in range(1, lag + 1):
        cov = (xm[l:] @ xm[:-l]) / n
        lrv += 2 * (1 - l / (lag + 1)) * cov
    if lrv <= 0:
        return np.nan, np.nan, lag
    se = np.sqrt(lrv / n)
    t = x.mean() / se
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t)))), lag


def block_bootstrap_ci(x: np.ndarray, block: int = 21, n_boot: int = 10_000,
                       seed: int = 42) -> tuple[float, float, float]:
    """IC del 95% por bootstrap de bloques, que preserva la dependencia serial.

    Devuelve (limite inferior, limite superior, P(media > 0)).
    """
    n = len(x)
    if n < block * 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        means[i] = np.concatenate([x[s:s + block] for s in starts])[:n].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means > 0).mean())


def observations_required(excess_ann: float, tracking_error_ann: float,
                          power: float = 0.80, alpha: float = 0.05,
                          periods_per_year: float = 365.0) -> dict[str, float]:
    """Cuantas observaciones harian falta para que la ventaja sea concluyente.

    Es la pregunta que casi nadie se hace antes de montar un forward test, y la
    que determina si el experimento tiene sentido. Con una ventaja pequena y un
    tracking error grande, la respuesta suele medirse en decadas.
    """
    if not np.isfinite(excess_ann) or excess_ann <= 0 or tracking_error_ann <= 0:
        return {"years_significance": np.inf, "years_powered": np.inf,
                "information_ratio": np.nan}
    ir = excess_ann / tracking_error_ann
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    years_sig = float((z_a / ir) ** 2)
    years_pow = float(((z_a + z_b) / ir) ** 2)
    return {
        "information_ratio": float(ir),
        "years_significance": years_sig,
        "years_powered": years_pow,
        "observations_significance": years_sig * periods_per_year,
        "observations_powered": years_pow * periods_per_year,
    }


def volatility_precision(n_obs: int) -> float:
    """Error relativo del estimador de volatilidad.

    ~1/sqrt(2(n-1)). Converge mucho mas rapido que la media, y por eso el
    control de riesgo si es contrastable en meses.
    """
    return float(1 / np.sqrt(2 * max(n_obs - 1, 1)))


@dataclass
class HypothesisVerdict:
    hid: str
    claim: str
    status: str          # 'cumple' | 'no cumple' | 'datos insuficientes' | 'no resoluble'
    observed: str
    required: str
    detail: str


def evaluate_hypotheses(result: ForwardResult, target_vol: float,
                        prereg: list[dict], ann: float = 365.0
                        ) -> list[HypothesisVerdict]:
    verdicts: list[HypothesisVerdict] = []
    n = result.n_gaps
    by_id = {h["id"]: h for h in prereg}

    # --- H1: se cumple el objetivo de volatilidad? -----------------------
    h = by_id.get("H1", {})
    invested = result.exposure >= 0.01
    need = int(h.get("min_observations", 180))
    if invested.sum() < 30:
        verdicts.append(HypothesisVerdict(
            "H1", h.get("claim", ""), "datos insuficientes",
            f"{int(invested.sum())} tramos invertido", f"{need}",
            "aun no hay suficientes dias con posicion abierta"))
    else:
        vol_in = annualised_volatility(result.returns[invested], result.days[invested], ann)
        rel = vol_in / target_vol - 1 if target_vol > 0 else np.nan
        prec = volatility_precision(int(invested.sum()))
        within = abs(rel) <= 0.40
        status = ("cumple" if within else "no cumple") if invested.sum() >= need \
            else "datos insuficientes"
        verdicts.append(HypothesisVerdict(
            "H1", h.get("claim", ""), status,
            f"{vol_in:.1%} realizada vs {target_vol:.1%} objetivo ({rel:+.0%})",
            f"{need} tramos invertido (hay {int(invested.sum())})",
            f"precision actual del estimador: +-{prec:.1%}"))

    # --- H3: hay ventaja frente al indice ajustado? ----------------------
    h = by_id.get("H3", {})
    if n < 60:
        verdicts.append(HypothesisVerdict(
            "H3", h.get("claim", ""), "datos insuficientes",
            f"{n} tramos", "al menos 60 para un contraste preliminar", ""))
    else:
        d = (result.returns - result.benchmark_scaled).dropna().to_numpy()
        excess = float(d.mean() * ann)
        te = float(d.std(ddof=1) * np.sqrt(ann))
        t, p, _ = newey_west_tstat(d)
        lo, hi, pos = block_bootstrap_ci(d)
        req = observations_required(excess, te, periods_per_year=ann)

        observed = f"{excess:+.2%} anual"
        if np.isfinite(p):
            observed += f" (p={p:.3f}"
            if np.isfinite(lo):
                observed += f", IC95% [{lo * ann:+.2%}, {hi * ann:+.2%}]"
            observed += ")"

        if excess <= 0:
            # No tiene sentido calcular cuantos anos harian falta para
            # demostrar una ventaja que, de momento, es negativa.
            required = "la ventaja observada es negativa: nada que demostrar"
            detail = f"tracking error {te:.2%}"
        else:
            required = f"~{req['years_significance']:.0f} anos de datos"
            detail = (f"tracking error {te:.2%}, "
                      f"information ratio {req['information_ratio']:.3f}")
        if np.isfinite(pos):
            detail += f", P(ventaja>0)={pos:.0%}"

        verdicts.append(HypothesisVerdict(
            "H3", h.get("claim", ""), "no resoluble", observed, required, detail))

    return verdicts
