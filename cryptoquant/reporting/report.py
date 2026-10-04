"""Informes: tablas en consola y graficos.

La paleta y las reglas de marca proceden de un sistema de visualizacion
validado para daltonismo (deuteranopia/protanopia/tritanopia). Los colores se
asignan por identidad de serie en orden fijo, nunca ciclados, y toda serie va
acompanada de leyenda o etiqueta directa: el color nunca es el unico canal que
transporta informacion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPORTS_DIR

# --- Tokens de color (paleta categorica validada, modo claro) --------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#006300"
BAD = "#e34948"

_PCT = {"total_return", "cagr", "volatility", "max_drawdown", "win_rate",
        "worst_period", "best_period", "ulcer_index", "pct_underwater"}


def fmt_metric(key: str, value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/d"
    if key in _PCT:
        return f"{value:>8.2%}"
    return f"{value:>8.2f}"


def print_metrics_table(comparison: pd.DataFrame, title: str = "Rendimiento") -> None:
    """Tabla comparativa en consola. Usa rich si esta disponible."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(f"\n{title}\n" + comparison.to_string())
        return

    console = Console()
    table = Table(title=title, header_style="bold", title_style="bold")
    table.add_column("Metrica", style="cyan", no_wrap=True)
    for col in comparison.index:
        table.add_column(str(col), justify="right")

    labels = {
        "cagr": "CAGR", "volatility": "Volatilidad anual", "sharpe": "Sharpe",
        "sortino": "Sortino", "calmar": "Calmar", "max_drawdown": "Max drawdown",
        "ulcer_index": "Indice de Ulcer", "martin_ratio": "Martin ratio",
        "win_rate": "% barras positivas", "profit_factor": "Profit factor",
        "worst_period": "Peor periodo", "skew": "Asimetria",
        "kurtosis": "Curtosis", "n_periods": "N periodos",
    }
    for metric in comparison.columns:
        row = [labels.get(metric, metric)]
        for strat in comparison.index:
            v = comparison.loc[strat, metric]
            txt = fmt_metric(metric, v)
            if metric in ("sharpe", "sortino", "calmar", "martin_ratio") and np.isfinite(v):
                color = "green" if v > 1 else ("yellow" if v > 0 else "red")
                txt = f"[{color}]{txt}[/{color}]"
            elif metric == "max_drawdown" and np.isfinite(v):
                color = "green" if v > -0.20 else ("yellow" if v > -0.40 else "red")
                txt = f"[{color}]{txt}[/{color}]"
            row.append(txt)
        table.add_row(*row)
    console.print(table)


def print_allocation(weights: pd.Series, prices: pd.Series | None = None,
                     capital: float | None = None) -> None:
    """Asignacion recomendada actual. Cumple la regla de relieve: la
    informacion del grafico de pesos existe tambien en forma de tabla."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(weights.to_string())
        return

    console = Console()
    table = Table(title="Asignacion recomendada", header_style="bold", title_style="bold")
    table.add_column("Activo", style="cyan")
    table.add_column("Peso", justify="right")
    if capital:
        table.add_column("Importe", justify="right")
    if prices is not None:
        table.add_column("Precio ref.", justify="right")

    active = weights[weights > 1e-4].sort_values(ascending=False)
    for sym, w in active.items():
        row = [str(sym), f"{w:.2%}"]
        if capital:
            row.append(f"{w * capital:,.2f}")
        if prices is not None:
            row.append(f"{prices.get(sym, float('nan')):,.4f}")
        table.add_row(*row)

    cash = max(0.0, 1.0 - float(active.sum()))
    row = ["EFECTIVO / stablecoin", f"{cash:.2%}"]
    if capital:
        row.append(f"{cash * capital:,.2f}")
    if prices is not None:
        row.append("-")
    table.add_row(*row, style="dim")
    console.print(table)


# --------------------------------------------------------------------------
# Graficos
# --------------------------------------------------------------------------
def _style_axes(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)


def plot_backtest(result, ann: float, filename: str = "backtest.png") -> Path:
    """Panel de cuatro graficos. Un solo eje por panel: nunca doble escala."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from ..backtest.metrics import rolling_sharpe
    from ..econometrics.risk import drawdown_series

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    eq = result.equity
    bench_eq = (1 + result.benchmark_returns).cumprod() * eq.iloc[0]

    # --- 1. Curva de capital (escala log: hace comparables las variaciones
    #        relativas a lo largo de todo el rango) -------------------------
    ax = axes[0]
    ax.plot(eq.index, eq.to_numpy(), color=SERIES[0], linewidth=2.0, label="Estrategia")
    ax.plot(bench_eq.index, bench_eq.to_numpy(), color=SERIES[1], linewidth=2.0,
            label="Buy & hold equiponderado")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _style_axes(ax, "Curva de capital (escala logaritmica)", "Capital")
    # Etiqueta directa al final de cada serie: identidad sin depender del color.
    for series, color, name in ((eq, SERIES[0], "Estrategia"), (bench_eq, SERIES[1], "Buy & hold")):
        ax.annotate(f"{name}  {series.iloc[-1]:,.0f}",
                    xy=(series.index[-1], series.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=8, va="center")
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    # --- 2. Drawdown ------------------------------------------------------
    ax = axes[1]
    dd_s = drawdown_series(eq)
    dd_b = drawdown_series(bench_eq)
    ax.fill_between(dd_b.index, dd_b.to_numpy(), 0, color=SERIES[1], alpha=0.18, linewidth=0)
    ax.plot(dd_b.index, dd_b.to_numpy(), color=SERIES[1], linewidth=1.4,
            label=f"Buy & hold (max {dd_b.min():.1%})")
    ax.fill_between(dd_s.index, dd_s.to_numpy(), 0, color=SERIES[0], alpha=0.22, linewidth=0)
    ax.plot(dd_s.index, dd_s.to_numpy(), color=SERIES[0], linewidth=2.0,
            label=f"Estrategia (max {dd_s.min():.1%})")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style_axes(ax, "Drawdown: cuanto se pierde desde el maximo previo", "Caida")
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    # --- 3. Exposicion bruta ---------------------------------------------
    ax = axes[2]
    ge = result.gross_exposure
    ax.fill_between(ge.index, ge.to_numpy(), 0, color=SERIES[0], alpha=0.20, linewidth=0)
    ax.plot(ge.index, ge.to_numpy(), color=SERIES[0], linewidth=1.6)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, linestyle="--")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style_axes(ax, "Exposicion al mercado (el resto permanece en efectivo)", "% invertido")

    # --- 4. Sharpe movil --------------------------------------------------
    ax = axes[3]
    window = min(180, max(30, len(result.returns) // 6))
    rs = rolling_sharpe(result.returns, window, ann)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.plot(rs.index, rs.to_numpy(), color=SERIES[0], linewidth=1.8)
    _style_axes(ax, f"Sharpe movil ({window} barras): estabilidad de la ventaja", "Sharpe")

    axes[-1].tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    # Espacio a la derecha para las etiquetas directas de la curva de capital.
    fig.subplots_adjust(right=0.86)
    out = REPORTS_DIR / filename
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_allocation(weights: pd.DataFrame, filename: str = "allocation.png") -> Path:
    """Evolucion de la asignacion por activo.

    Areas apiladas con separador de 2px entre segmentos, para que los limites
    se lean aunque dos colores adyacentes sean parecidos.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    w = weights.clip(lower=0.0)
    # Orden estable: el color sigue al activo, no a su ranking del momento.
    cols = sorted(w.columns, key=str)
    w = w[cols]
    if len(cols) > len(SERIES):
        keep = w.mean().nlargest(len(SERIES) - 1).index.tolist()
        other = w.drop(columns=keep).sum(axis=1)
        w = w[keep].assign(Otros=other)
        cols = list(w.columns)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.stackplot(w.index, [w[c].to_numpy() for c in cols],
                 labels=cols, colors=SERIES[:len(cols)],
                 edgecolor=SURFACE, linewidth=1.0)
    # La escala se ajusta a la exposicion real. Forzar 0-100% cuando el sistema
    # nunca pasa del 45% deja mas de la mitad del panel vacio y aplasta la
    # composicion justo donde hay que leerla.
    peak = float(w.sum(axis=1).max())
    ax.set_ylim(0, min(1.0, max(0.1, peak * 1.12)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style_axes(ax, "Composicion de la cartera en el tiempo", "% del capital")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=min(len(cols), 5),
              labelcolor=INK_SECONDARY)
    fig.tight_layout()
    out = REPORTS_DIR / filename
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out


def export_csv(result, prefix: str = "backtest") -> list[Path]:
    """Exporta las series a CSV. Junto con el grafico de pesos cumple el
    requisito de que exista una vista tabular de todo lo que se dibuja."""
    paths = []
    frames = {
        "equity": result.equity.to_frame("equity"),
        "returns": result.returns.to_frame("return"),
        "weights": result.weights,
        "diagnostics": result.diagnostics,
    }
    for name, df in frames.items():
        if df is None or len(df) == 0:
            continue
        p = REPORTS_DIR / f"{prefix}_{name}.csv"
        df.to_csv(p)
        paths.append(p)
    return paths
