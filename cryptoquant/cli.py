"""Interfaz de linea de comandos.

Subcomandos:
    analyze      Diagnostico econometrico del universo
    pairs        Busqueda de pares cointegrados (valor relativo)
    backtest     Backtest walk-forward con comparativa de metodos
    recommend    Asignacion recomendada para hoy
    laboratorio  De las velas al modelo, paso a paso (y CSV para R)
    riesgo       Riesgo del modelo frente a la pasiva; backtest del VaR
    cartera      Riesgo de una cartera concreta
    evidencia    Filtro que toda senal nueva debe pasar
    paquete      CSV de respaldo para una clase
    piloto       Piloto de riesgo: cuanto tener en cripto con su cartera real (app)
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from .backtest.engine import WalkForwardBacktest
from .backtest.metrics import compare, summarize
from .config import Config, annualization_factor
from .data.sources import load_universe
from .econometrics.diagnostics import diagnose_universe, engle_granger_pairs
from .econometrics.risk import compute_risk
from .econometrics.volatility import forecast_universe, ledoit_wolf_cov
from .models.validation import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from .reporting import report as rp


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _echo(msg: str = "") -> None:
    print(msg)


def _load(cfg: Config, refresh: bool):
    _echo("Cargando datos de mercado...")
    ohlcv, closes, origins = load_universe(cfg, force_refresh=refresh)
    counts: dict[str, int] = {}
    for o in origins.values():
        counts[o] = counts.get(o, 0) + 1
    _echo(f"  {len(closes.columns)} activos | {len(closes)} barras "
          f"| {closes.index[0].date()} a {closes.index[-1].date()}")
    _echo(f"  Origen: {counts}")
    if counts.get("synthetic"):
        _echo("  AVISO: hay series sinteticas (sin red o simbolo no disponible).")
        _echo("         Los resultados NO son informativos sobre el mercado real.")
    _echo()
    return ohlcv, closes, origins


# --------------------------------------------------------------------------
def cmd_analyze(args, cfg: Config) -> int:
    ann = annualization_factor(cfg.data.timeframe)
    _, closes, _ = _load(cfg, args.refresh)

    _echo("Contrastes econometricos por activo")
    _echo("-" * 78)
    diag = diagnose_universe(closes, ann)
    view = diag[["annual_vol", "hurst", "adf_pvalue", "arch_pvalue", "jarque_bera_pvalue"]].copy()
    view.columns = ["Vol anual", "Hurst", "ADF p", "ARCH p", "JB p"]
    view["Vol anual"] = view["Vol anual"].map(lambda v: f"{v:.1%}")
    for c in ("Hurst", "ADF p", "ARCH p", "JB p"):
        view[c] = view[c].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "n/d")
    _echo(view.to_string())
    _echo()
    _echo("Lectura:")
    for sym, row in diag.iterrows():
        _echo(f"  {sym:6s} {row['verdict']}")
    _echo()

    _echo("Riesgo historico (retornos diarios)")
    _echo("-" * 78)
    rows = {}
    fallbacks = []
    rets = closes.pct_change().dropna()
    for sym in closes.columns:
        m = compute_risk(rets[sym], ann, cfg.risk.var_confidence)
        rows[sym] = {
            "VaR 95% hist": m.var_historical,
            "VaR 95% normal": m.var_parametric,
            "VaR 95% C-F": m.var_cornish_fisher,
            "CVaR 95%": m.cvar_historical,
            "Max DD": m.max_drawdown,
        }
        if m.cf_used_fallback:
            fallbacks.append(sym)
    risk_df = pd.DataFrame(rows).T
    _echo(risk_df.map(lambda v: f"{v:.2%}" if np.isfinite(v) else "n/d").to_string())
    _echo()
    _echo("El VaR normal casi siempre es el mas benigno: subestima las colas.")
    _echo("La diferencia con el historico y el de Cornish-Fisher es la medida")
    _echo("de cuanto riesgo esconde el supuesto de normalidad.")
    if fallbacks:
        _echo()
        _echo(f"Cornish-Fisher fuera de dominio en: {', '.join(fallbacks)}")
        _echo("Sus momentos son tan extremos que la expansion deja de ser una")
        _echo("funcion cuantil valida; para esos activos se muestra el VaR")
        _echo("historico. Use el CVaR, que no depende de ningun supuesto.")
    _echo()

    _echo("Prevision de volatilidad a 1 dia (GJR-GARCH)")
    _echo("-" * 78)
    vf = forecast_universe(np.log(closes).diff().dropna(), ann, use_garch=not args.fast)
    vf["sigma_ann"] = vf["sigma_ann"].map(lambda v: f"{float(v):.1%}" if np.isfinite(float(v)) else "n/d")
    vf["persistence"] = vf["persistence"].map(lambda v: f"{float(v):.3f}" if np.isfinite(float(v)) else "n/d")
    _echo(vf.to_string())
    _echo()
    _echo("Persistencia cerca de 1: los shocks de volatilidad tardan en disiparse,")
    _echo("asi que tras un dia malo conviene reducir tamano, no promediar a la baja.")
    return 0


# --------------------------------------------------------------------------
def cmd_pairs(args, cfg: Config) -> int:
    _, closes, _ = _load(cfg, args.refresh)
    _echo("Cointegracion (Engle-Granger) sobre log-precios")
    _echo("-" * 78)
    pairs = engle_granger_pairs(closes)
    if pairs.empty:
        _echo("Sin resultados.")
        return 0
    view = pairs.head(args.top).copy()
    view["pvalue"] = view["pvalue"].map(lambda v: f"{v:.4f}")
    view["hedge_ratio"] = view["hedge_ratio"].map(lambda v: f"{v:.3f}")
    view["half_life_days"] = view["half_life_days"].map(
        lambda v: f"{v:.1f}" if np.isfinite(v) else "inf")
    view["spread_z"] = view["spread_z"].map(lambda v: f"{v:+.2f}")
    _echo(view.to_string(index=False))
    _echo()
    sig = int(pairs["significant"].sum())
    sig_b = int(pairs["significant_bonferroni"].sum())
    _echo(f"Pares significativos al 5%: {sig} de {len(pairs)}")
    _echo(f"Tras correccion de Bonferroni: {sig_b}")
    _echo()
    _echo("Solo los que sobreviven a Bonferroni son creibles: al contrastar todos")
    _echo("los pares, algunos salen significativos por puro azar.")
    _echo("Un spread_z por encima de +2 sugiere vender A y comprar B; por debajo")
    _echo("de -2, lo contrario. Vidas medias por encima de ~60 dias no son operables.")
    return 0


# --------------------------------------------------------------------------
def cmd_backtest(args, cfg: Config) -> int:
    ann = annualization_factor(cfg.data.timeframe)
    ohlcv, closes, _ = _load(cfg, args.refresh)

    strategies: dict[str, pd.Series] = {}
    results = {}

    variants = [("Sistema completo", args.method, True)]
    if args.compare:
        variants += [
            ("Sin ML (solo econometria)", args.method, False),
            ("Equiponderado + riesgo", "equal", True),
        ]

    for name, method, use_ml in variants:
        _echo(f"Ejecutando: {name} (cartera={method}, ML={'si' if use_ml else 'no'})...")
        bt = WalkForwardBacktest(cfg, portfolio_method=method, use_ml=use_ml)
        res = bt.run(ohlcv, closes)
        strategies[name] = res.returns
        results[name] = res

    main = results["Sistema completo"]
    strategies["Buy & hold"] = main.benchmark_returns

    _echo()
    rp.print_metrics_table(compare(strategies, ann), "Comparativa walk-forward (neto de costes)")

    # --- Significancia estadistica ---------------------------------------
    _echo()
    _echo("Significancia estadistica del resultado principal")
    _echo("-" * 78)
    s = summarize(main.returns, ann)
    # Cota inferior honesta: 4 metodos de cartera x 3 variantes de senal, mas
    # el tanteo de parametros. Subirlo solo puede empeorar el veredicto, que es
    # justo el sentido de la correccion.
    n_trials = 12
    psr = probabilistic_sharpe_ratio(s["sharpe"], s["n_periods"], 0.0,
                                     s["skew"], s["kurtosis"] + 3, ann)
    dsr = deflated_sharpe_ratio(s["sharpe"], s["n_periods"], n_trials,
                                s["skew"], s["kurtosis"] + 3, ann)
    _echo(f"  Sharpe observado            : {s['sharpe']:.2f}")
    _echo(f"  P(Sharpe real > 0)          : {psr:.1%}" if np.isfinite(psr) else "  PSR: n/d")
    _echo(f"  Sharpe deflactado ({n_trials} pruebas): {dsr:.1%}" if np.isfinite(dsr) else "  DSR: n/d")
    _echo(f"  Observaciones               : {s['n_periods']}")
    _echo()
    if np.isfinite(dsr) and dsr < 0.95:
        _echo("  El Sharpe deflactado esta por debajo del 95%: el resultado no se")
        _echo("  distingue de la suerte una vez descontado el sesgo de seleccion.")
        _echo("  No desplegar capital sobre esta base.")
    else:
        _echo("  El resultado supera el umbral habitual de deflactacion.")

    # --- Costes y rotacion ------------------------------------------------
    _echo()
    _echo("Costes y rotacion")
    _echo("-" * 78)
    total_cost = float(main.costs.sum())
    _echo(f"  Coste total acumulado       : {total_cost:.2%} del capital")
    _echo(f"  Rotacion media por barra    : {float(main.turnover.mean()):.2%}")
    _echo(f"  Exposicion media al mercado : {float(main.gross_exposure.mean()):.1%}")
    _echo(f"  Barras totalmente en cash   : {float((main.gross_exposure < 0.01).mean()):.1%}")

    # --- Se cumple el objetivo de riesgo? --------------------------------
    _echo()
    _echo("Control de volatilidad: se cumple lo pedido?")
    _echo("-" * 78)
    invested = main.gross_exposure >= 0.01
    vol_all = float(main.returns.std(ddof=1) * np.sqrt(ann))
    vol_in = (float(main.returns[invested].std(ddof=1) * np.sqrt(ann))
              if invested.sum() > 30 else float("nan"))
    _echo(f"  Objetivo                    : {cfg.risk.target_volatility:.1%}")
    _echo(f"  Realizada (todo el periodo) : {vol_all:.1%}")
    _echo(f"  Realizada (solo invertido)  : {vol_in:.1%}")
    _echo()
    _echo("  El objetivo se aplica MIENTRAS se esta invertido. Como el sistema")
    _echo("  pasa mucho tiempo fuera del mercado, la volatilidad medida sobre")
    _echo("  todo el periodo sale por debajo. Si quiere acercar ambas cifras,")
    _echo("  relaje `min_probability` o `kelly_full_size_level` en config.yaml.")

    if main.model_reports:
        mr = pd.DataFrame(main.model_reports)
        _echo()
        _echo("Modelo ML (metricas dentro de muestra de cada reentrenamiento)")
        _echo("-" * 78)
        _echo(f"  Reentrenamientos            : {len(mr)}")
        _echo(f"  AUC medio (in-sample)       : {mr['auc'].mean():.3f}")
        _echo(f"  Brier medio (in-sample)     : {mr['brier'].mean():.3f}")
        _echo(f"  Tasa base media             : {mr['base_rate'].mean():.3f}")
        _echo("  Son metricas de entrenamiento, no de validacion: la validacion")
        _echo("  real es la curva walk-forward de arriba.")

    # --- Motivos de las decisiones ---------------------------------------
    if not main.diagnostics.empty:
        _echo()
        _echo("Motivos de dimensionamiento (frecuencia)")
        _echo("-" * 78)
        for reason, cnt in main.diagnostics["reason"].value_counts().items():
            _echo(f"  {cnt:5d}  {reason}")

    # --- Salidas ----------------------------------------------------------
    if not args.no_plots:
        _echo()
        try:
            p1 = rp.plot_backtest(main, ann)
            p2 = rp.plot_allocation(main.weights)
            _echo(f"Graficos: {p1}")
            _echo(f"          {p2}")
        except Exception as exc:
            _echo(f"No se pudieron generar los graficos: {exc}")
    for p in rp.export_csv(main):
        _echo(f"CSV:      {p}")
    return 0


# --------------------------------------------------------------------------
def cmd_recommend(args, cfg: Config) -> int:
    ann = annualization_factor(cfg.data.timeframe)
    ohlcv, closes, _ = _load(cfg, args.refresh)

    _echo("Calculando estado del sistema a fecha de hoy...")
    bt = WalkForwardBacktest(cfg, portfolio_method=args.method, use_ml=True)
    res = bt.run(ohlcv, closes)

    latest = res.weights.iloc[-1]
    prices = closes.iloc[-1]
    capital = args.capital or cfg.backtest.initial_capital

    _echo()
    rp.print_allocation(latest, prices, capital)

    if not res.diagnostics.empty:
        d = res.diagnostics.iloc[-1]
        _echo()
        _echo(f"Fecha de la ultima decision : {res.diagnostics.index[-1].date()}")
        _echo(f"Activos elegibles           : {int(d['n_eligible'])}")
        _echo(f"Vol prevista de la cartera  : "
              f"{d['port_vol_forecast']:.1%}" if np.isfinite(d["port_vol_forecast"]) else "n/d")
        _echo(f"Objetivo de volatilidad     : {cfg.risk.target_volatility:.1%}")
        _echo(f"Escalar por volatilidad     : {d['vol_scalar']:.2f}")
        _echo(f"Escalar por drawdown        : {d['dd_scalar']:.2f}")
        _echo(f"Kelly bruto                 : {d['kelly_raw']:.3f} "
              f"(tamano pleno a partir de {cfg.risk.kelly_full_size_level})")
        _echo(f"Factor de conviccion        : {d['conviction']:.2f}")
        _echo(f"Motivo                      : {d['reason']}")
        _echo(f"Detalle                     : {d['detail']}")

    m = summarize(res.returns, ann)
    _echo()
    _echo("Comportamiento historico de esta configuracion (walk-forward)")
    _echo("-" * 78)
    _echo(f"  CAGR {m['cagr']:.1%} | Vol {m['volatility']:.1%} | "
          f"Sharpe {m['sharpe']:.2f} | Max DD {m['max_drawdown']:.1%}")
    _echo()
    _echo("Esto es una salida de un modelo estadistico, no una recomendacion")
    _echo("de inversion. El rendimiento pasado no predice el futuro.")
    return 0


# --------------------------------------------------------------------------
# Forward test
# --------------------------------------------------------------------------
def cmd_forward_init(args, cfg: Config) -> int:
    from . import __version__
    from .forward import journal as jn

    _, closes, _ = _load(cfg, args.refresh)
    try:
        prereg = jn.write_preregistration(cfg, list(closes.columns), __version__,
                                          overwrite=args.overwrite)
    except FileExistsError as exc:
        _echo(f"ERROR: {exc}")
        _echo("Use --overwrite solo si esta empezando de cero a proposito.")
        return 1

    _echo(f"Pre-registro creado: {jn.PREREG_PATH}")
    _echo(f"Hash: {prereg.short_hash}")
    _echo()
    _echo("Hipotesis fijadas ANTES de tener datos")
    _echo("=" * 78)
    for h in prereg.hypotheses:
        mark = "CONTRASTABLE" if h["testable"] else "NO RESOLUBLE"
        _echo(f"\n{h['id']} [{h['rank']}] -- {mark}")
        _echo(f"  Afirmacion : {h['claim']}")
        _echo(f"  Metrica    : {h['metric']}")
        _echo(f"  Exito si   : {h['success']}")
        _echo(f"  Necesita   : {h['min_observations']:,} observaciones")
        for line in _wrap(h["note"], 72):
            _echo(f"  {line}")
    _echo()
    _echo("=" * 78)
    _echo("A partir de aqui las hipotesis no se tocan. Cambiarlas despues de ver")
    _echo("los datos es como se fabrican los resultados que no se replican.")
    _echo()
    _echo("Siguiente paso: ejecute `forward record` una vez al dia.")
    return 0


def cmd_forward_amend(args, cfg: Config) -> int:
    """Registra la enmienda E1 antes de la primera anotacion a la que afecta."""
    from . import __version__
    from .forward import journal as jn

    if jn.read_preregistration() is None:
        _echo("No hay pre-registro. Ejecute primero: forward init")
        return 1
    status = jn.verify_chain()
    if not status.ok:
        _echo("ERROR: el diario no esta intacto; no se enmienda un experimento roto.")
        return 1
    _, closes, origins = _load(cfg, args.refresh)
    if {o for o in origins.values() if o in ("synthetic", "cache-stale")}:
        _echo("ERROR: los datos no estan al dia; el ancla debe ser una barra real.")
        return 1

    anchor = str(closes.index[-1].date())
    try:
        a = jn.write_amendment(jn.AMENDMENT_E1, anchor, cfg, __version__)
    except (FileExistsError, ValueError) as exc:
        _echo(f"ERROR: {exc}")
        return 1

    _echo(f"Enmienda {a['id']} registrada: {a['title']}")
    _echo(f"  Hash          : {a['hash'][:12]}")
    _echo(f"  Rige desde    : {a['effective_from']} (ancla del calendario)")
    head = a["journal_head"]
    _echo(f"  Tras anotacion: #{head['seq']}" if head else "  Tras anotacion: ninguna")
    _echo()
    for line in _wrap(a["reason"], 76):
        _echo(f"  {line}")
    _echo()
    for c in a["changes"]:
        for n, line in enumerate(_wrap(c, 74)):
            _echo(("  - " if n == 0 else "    ") + line)
    _echo()
    _echo("Las hipotesis no cambian. H1 y H2 se cuentan desde la fecha de arriba.")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


def cmd_forward_record(args, cfg: Config) -> int:
    from . import __version__
    from .forward import journal as jn

    if jn.read_preregistration() is None:
        _echo("No hay pre-registro. Ejecute primero: forward init")
        return 1

    status = jn.verify_chain()
    if not status.ok:
        _echo("ERROR: el diario esta corrupto. No se anade nada mas hasta resolverlo.")
        for p in status.problems:
            _echo(f"  - {p}")
        return 1

    ohlcv, closes, origins = _load(cfg, args.refresh)
    bad = {o for o in origins.values() if o in ("synthetic", "cache-stale")}
    if bad:
        _echo("ERROR: los datos no estan al dia.")
        if "synthetic" in bad:
            _echo("       Hay series SINTETICAS. Registrar decisiones sobre datos")
            _echo("       inventados destruye el experimento.")
        if "cache-stale" in bad:
            _echo("       Hay series de CACHE ATRASADA: no se pudo descargar y la")
            _echo("       copia local no cubre la ultima barra cerrada.")
        _echo("       Revise la conexion. La tarea reintentara en unas horas;")
        _echo("       no se registra nada hasta que los datos sean correctos.")
        return 1

    from .backtest.engine import bar_position
    from .forward.evaluate import portfolio_state_at

    records = jn.read_journal()
    amendment = jn.current_amendment()
    active = jn.active_records(records, amendment)

    i = bar_position(closes.index, args.date)
    decision_date = str(closes.index[i].date())
    prices = closes.iloc[i]

    # Modo desatendido: si la barra de hoy ya esta registrada no es un error,
    # es lo normal cuando la tarea programada corre varias veces al dia.
    if args.if_new and any(r["decision_date"] == decision_date for r in records):
        _echo(f"Ya existe la anotacion de {decision_date}. Nada que hacer.")
        return 0

    # Estado de la cartera segun el propio diario, movido por el precio hasta
    # hoy: es lo que ve el backtest en la barra en que decide.
    prev = active[-1] if active else None
    prev_w, dd_now = portfolio_state_at(
        active, decision_date, {k: float(v) for k, v in prices.items()},
        cfg.backtest.fee, cfg.backtest.slippage)
    last_reb = next((r["decision_date"] for r in reversed(active)
                     if str((r.get("diagnostics") or {}).get("action", "")).startswith("rebalanceo")),
                    None)
    prev_target = dict(prev["weights"]) if prev else {}

    _echo("Calculando la decision de hoy (mismo codigo que el backtest)...")
    bt = WalkForwardBacktest(cfg, portfolio_method=args.method, use_ml=True)
    weights, diag = bt.decide_at(
        ohlcv, closes, when=closes.index[i], current_drawdown=dd_now,
        current_weights=pd.Series(prev_w, dtype=float) if prev_w else None,
        anchor=amendment["anchor"] if amendment else None,
        last_rebalance=last_reb,
        previous_target=pd.Series(prev_target, dtype=float) if prev else None,
    )
    if diag.get("action") == "mantener":
        # Entre rebalanceos la explicacion sigue siendo la del ultimo.
        carried = {k: v for k, v in (prev.get("diagnostics") or {}).items() if k not in diag}
        diag = {**carried, **diag}

    if args.dry_run:
        _echo("\n--- SIMULACION, no se escribe nada ---")
        rp.print_allocation(weights, prices, args.capital or cfg.backtest.initial_capital)
        _echo(f"\nAccion: {diag.get('action', 'rebalanceo')}")
        _echo(f"Motivo: {diag.get('reason')}  ({diag.get('detail')})")
        return 0

    try:
        rec = jn.append_record(
            decision_date=decision_date,
            prices={k: float(v) for k, v in prices.items()},
            weights={k: float(v) for k, v in weights.items()},
            gross_exposure=float(weights.sum()),
            diagnostics={k: (float(v) if isinstance(v, (int, float, np.floating))
                             else str(v))
                         for k, v in diag.items() if k != "date"},
            cfg=cfg, code_version=__version__, note=args.note,
            state_in={
                "prev_weights": {k: float(v) for k, v in prev_w.items()},
                "prev_target": {k: float(v) for k, v in prev_target.items()},
                "drawdown": float(dd_now),
                "last_rebalance": last_reb,
            },
        )
    except ValueError as exc:
        _echo(f"ERROR: {exc}")
        return 1

    _echo()
    rp.print_allocation(weights, prices, args.capital or cfg.backtest.initial_capital)
    _echo()
    _echo(f"Anotacion #{rec['seq']} registrada para {decision_date}")
    _echo(f"  Hash          : {rec['hash'][:16]}")
    _echo(f"  Enlaza con    : {rec['prev_hash'][:16]}")
    _echo(f"  Drawdown usado: {dd_now:.2%}")
    if "action" in diag:
        _echo(f"  Accion        : {diag['action']} (proximo rebalanceo: {diag['next_rebalance']})")
    _echo(f"  Motivo        : {diag.get('reason')}")
    _echo(f"  Detalle       : {diag.get('detail')}")
    _echo()
    _echo(f"Diario: {jn.JOURNAL_PATH}")
    return 0


def cmd_forward_status(args, cfg: Config) -> int:
    """Salud del experimento: sirve para comprobar de un vistazo que la
    automatizacion sigue viva y que no se estan perdiendo dias."""
    from datetime import date, datetime, timezone

    from .forward import journal as jn

    status = jn.verify_chain()
    prereg = jn.read_preregistration()
    amendment = jn.current_amendment()
    all_records = jn.read_journal() if status.n_records or status.ok else []
    records = jn.active_records(all_records, amendment)

    _echo("Estado del forward test")
    _echo("=" * 78)
    if prereg is None:
        _echo("  Sin pre-registro. Ejecute: forward init")
        return 1
    _echo(f"  Pre-registro   : {prereg.short_hash} ({prereg.created_utc[:10]})")
    if amendment:
        _echo(f"  Protocolo      : enmienda {amendment['id']} desde {amendment['effective_from']} "
              f"({len(all_records) - len(records)} anotaciones anteriores archivadas)")
    _echo(f"  Integridad     : {'OK' if status.ok else 'ROTA'}")
    for p in status.problems:
        _echo(f"    - {p}")
    _echo(f"  Anotaciones    : {len(records)}")
    if not records:
        _echo()
        _echo("  Aun no hay ninguna. Ejecute: forward record")
        return 0

    first = date.fromisoformat(records[0]["decision_date"])
    last = date.fromisoformat(records[-1]["decision_date"])
    today = datetime.now(timezone.utc).date()
    stale = (today - last).days

    _echo(f"  Primera        : {first}")
    _echo(f"  Ultima         : {last}  (hace {stale} dias)")

    # Huecos: dias naturales del periodo que no tienen anotacion.
    span = (last - first).days + 1
    missing = span - len(records)
    _echo(f"  Cobertura      : {len(records)}/{span} dias del periodo "
          f"({len(records) / span:.0%})")
    if missing > 0:
        _echo(f"  Dias sin anotar: {missing}")
        _echo("    No se rellenan a posteriori: un backfill es un backtest, no")
        _echo("    evidencia prospectiva. La posicion simplemente se mantuvo.")

    invested = sum(1 for r in records if r["gross_exposure"] >= 0.01)
    _echo(f"  Con posicion   : {invested}")
    nxt = (records[-1].get("diagnostics") or {}).get("next_rebalance")
    if nxt:
        _echo(f"  Prox. rebalanceo: {nxt}")

    _echo()
    _echo("Progreso de las hipotesis contrastables")
    _echo("-" * 78)
    for h in prereg.hypotheses:
        if not h["testable"]:
            continue
        have = invested if h["id"] == "H1" else len(records)
        need = int(h["min_observations"])
        pct = min(have / need, 1.0)
        bar = "#" * int(pct * 30) + "." * (30 - int(pct * 30))
        eta = max(need - have, 0)
        _echo(f"  {h['id']}  [{bar}] {have}/{need}"
              + (f"  faltan ~{eta} dias" if eta else "  COMPLETA"))

    if stale >= 3:
        _echo()
        _echo(f"  AVISO: la ultima anotacion es de hace {stale} dias.")
        _echo("  Compruebe la tarea programada: scripts\\install_task.ps1 -Status")

    log = jn.FORWARD_DIR / "record.log"
    if log.exists() and args.log:
        _echo()
        _echo("Ultimas lineas del registro de la automatizacion")
        _echo("-" * 78)
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
            _echo(f"  {line}")
    return 0


def cmd_forward_verify(args, cfg: Config) -> int:
    from .forward import journal as jn

    status = jn.verify_chain()
    _echo("Integridad del diario")
    _echo("-" * 78)
    _echo(f"  Anotaciones          : {status.n_records}")
    _echo(f"  Pre-registro intacto : "
          f"{'si' if jn.preregistration_is_intact() else 'NO / ausente'}")
    if status.ok:
        _echo("  Cadena de hashes     : intacta")
        _echo()
        _echo("Ninguna decision pasada ha sido modificada desde que se escribio.")
        return 0
    _echo(f"  Cadena de hashes     : ROTA en el registro {status.first_break}")
    _echo()
    for p in status.problems:
        _echo(f"  - {p}")
    _echo()
    _echo("El forward test queda invalidado a partir de ese punto.")
    return 1


def cmd_forward_report(args, cfg: Config) -> int:
    from .forward import journal as jn
    from .forward.evaluate import (
        annualised_volatility,
        block_bootstrap_ci,
        evaluate_hypotheses,
        newey_west_tstat,
        observations_required,
        realised_returns,
        volatility_precision,
    )

    ann = annualization_factor(cfg.data.timeframe)
    prereg = jn.read_preregistration()
    if prereg is None:
        _echo("No hay pre-registro. Ejecute primero: forward init")
        return 1

    status = jn.verify_chain()
    if status.n_records == 0 and not status.ok:
        _echo("ERROR: el diario no se puede leer.")
        for p in status.problems:
            _echo(f"  - {p}")
        return 1
    amendment = jn.current_amendment()
    all_records = jn.read_journal()
    records = jn.active_records(all_records, amendment)
    _echo("Forward test -- informe")
    _echo("=" * 78)
    _echo(f"  Pre-registro    : {prereg.short_hash} ({prereg.created_utc[:10]})")
    if amendment:
        _echo(f"  Protocolo       : enmienda {amendment['id']} desde "
              f"{amendment['effective_from']}; se evaluan solo sus anotaciones "
              f"({len(all_records) - len(records)} anteriores archivadas)")
    _echo(f"  Integridad      : {'OK' if status.ok else 'ROTA -- resultados no fiables'}")
    _echo(f"  Anotaciones     : {len(records)}")
    if not status.ok:
        for p in status.problems:
            _echo(f"    - {p}")
    if len(records) < 2:
        _echo()
        _echo("Aun no hay tramos que medir. Vuelva cuando haya al menos 2 anotaciones.")
        return 0

    _, closes, _ = _load(cfg, args.refresh)
    res = realised_returns(records, closes, cfg.backtest.fee, cfg.backtest.slippage)
    for n in res.notes:
        _echo(f"  AVISO: {n}")

    eq = (1 + res.returns).cumprod()
    _echo()
    _echo(f"Periodo cubierto  : {records[0]['decision_date']} a "
          f"{records[-1]['decision_date']} ({res.span_days} dias, {res.n_gaps} tramos)")
    _echo()
    _echo("Resultado realizado")
    _echo("-" * 78)
    _echo(f"  Retorno acumulado           : {float(eq.iloc[-1] - 1):+.2%}")
    _echo(f"  Buy & hold en el mismo plazo: {float((1 + res.benchmark).prod() - 1):+.2%}")
    # Con un solo tramo la desviacion tipica no existe: decirlo, no imprimir nan.
    if len(res.returns) >= 2:
        _echo(f"  Volatilidad anualizada      : "
              f"{annualised_volatility(res.returns, res.days, ann):.2%}"
              "  (cada tramo normalizado por sus dias)")
    else:
        _echo("  Volatilidad anualizada      : n/d (hacen falta al menos 2 tramos)")
    _echo(f"  Exposicion media            : {float(res.exposure.mean()):.1%}")
    _echo(f"  Costes acumulados           : {float(res.costs.sum()):.2%}")

    _echo()
    _echo("Contraste de las hipotesis pre-registradas")
    _echo("=" * 78)
    for v in evaluate_hypotheses(res, cfg.risk.target_volatility,
                                 prereg.hypotheses, ann):
        _echo(f"\n{v.hid}  [{v.status.upper()}]")
        for line in _wrap(v.claim, 72):
            _echo(f"  {line}")
        _echo(f"  Observado : {v.observed}")
        _echo(f"  Necesita  : {v.required}")
        if v.detail:
            _echo(f"  Nota      : {v.detail}")

    # --- Cuanto falta para poder concluir algo ---------------------------
    if res.n_gaps >= 30:
        d = (res.returns - res.benchmark_scaled).dropna().to_numpy()
        excess = float(d.mean() * ann)
        te = float(d.std(ddof=1) * np.sqrt(ann))
        req = observations_required(excess, te, periods_per_year=ann)
        _echo()
        _echo("Cuanto falta para que la ventaja sea concluyente")
        _echo("-" * 78)
        _echo(f"  Ventaja observada           : {excess:+.2%} anual")
        _echo(f"  Tracking error              : {te:.2%} anual")
        if np.isfinite(req["years_significance"]):
            _echo(f"  Information ratio           : {req['information_ratio']:.3f}")
            _echo(f"  Anos para p<0.05            : {req['years_significance']:.1f}")
            _echo(f"  Anos con 80% de potencia    : {req['years_powered']:.1f}")
        else:
            _echo("  La ventaja observada es negativa. No hay nada que demostrar")
            _echo("  todavia; siga registrando y vuelva a mirar mas adelante.")
        _echo()
        _echo("  Si esa cifra son decadas, ningun forward test la va a resolver.")
        _echo("  Es el resultado honesto, no un fallo del experimento.")

    inv = int((res.exposure >= 0.01).sum())
    _echo()
    _echo(f"Precision actual del estimador de volatilidad: "
          f"+-{volatility_precision(inv):.1%} ({inv} tramos invertido)")
    return 0


def cmd_forward_reproduce(args, cfg: Config) -> int:
    """Recomputa decisiones pasadas y las compara con lo registrado (H2)."""
    from .forward import journal as jn

    records = jn.read_journal()
    if not records:
        _echo("Diario vacio.")
        return 0

    ohlcv, closes, _ = _load(cfg, args.refresh)
    bt = WalkForwardBacktest(cfg, portfolio_method=args.method, use_ml=True)

    subset = records[-args.last:] if args.last else records
    _echo("Reproduccion de decisiones pasadas")
    _echo("-" * 78)
    _echo("Todo el sistema es causal, asi que recalcular hoy la decision de una")
    _echo("fecha pasada debe dar el mismo resultado que se registro entonces.")
    _echo("Si difiere: o el exchange reviso el historico, o hay fuga de futuro.")
    _echo()

    worst = 0.0
    failures = 0
    n_scheduled = 0
    for rec in subset:
        state = rec.get("state_in") or {}
        prev = state.get("prev_weights") or {}
        target = state.get("prev_target") or {}
        amendment = jn.amendment_by_hash(rec.get("amendment_hash"))
        try:
            w, diag = bt.decide_at(
                ohlcv, closes, when=rec["decision_date"],
                current_drawdown=float(state.get("drawdown", 0.0)),
                current_weights=pd.Series(prev, dtype=float) if prev else None,
                anchor=amendment["anchor"] if amendment else None,
                last_rebalance=state.get("last_rebalance"),
                previous_target=pd.Series(target, dtype=float) if amendment else None,
            )
        except Exception as exc:
            _echo(f"  {rec['decision_date']}  no reproducible: {exc}")
            failures += 1
            continue
        recorded = pd.Series(rec["weights"], dtype=float)
        diff = (w.reindex(recorded.index).fillna(0.0) - recorded).abs()
        dmax = float(diff.max()) if len(diff) else 0.0
        worst = max(worst, dmax)
        problems = []
        if dmax >= 0.01:
            problems.append(f"dif. max {dmax:.4f} ({diff.idxmax()})")
        if amendment:
            n_scheduled += 1
            rd = rec.get("diagnostics") or {}
            for key in ("action", "model_upto"):
                if rd.get(key) != diag.get(key):
                    problems.append(f"{key}: registrado {rd.get(key)}, recalculado {diag.get(key)}")
        if problems or args.verbose:
            _echo(f"  {rec['decision_date']}  {'DIFIERE' if problems else 'OK'}  "
                  + ("; ".join(problems) if problems else f"dif. max {dmax:.4f}"))
        if problems:
            failures += 1

    _echo()
    _echo(f"  Comprobadas    : {len(subset)} "
          f"({n_scheduled} con el calendario del backtest)")
    _echo(f"  Diferencia max : {worst:.4f} ({worst * 100:.2f} pp)")
    if failures == 0:
        # Recalcular prueba reproducibilidad y causalidad. Que el calendario sea
        # el del backtest lo comprueba ademas un test automatico
        # (tests/test_forward.py::test_live_schedule_matches_the_backtest).
        _echo("  H2 CUMPLE: las decisiones son reproducibles"
              + (" y siguen el calendario del backtest." if n_scheduled == len(subset)
                 else "; las de la fase 1 no seguian el calendario del backtest."))
        return 0
    _echo(f"  H2 NO CUMPLE: {failures} decisiones no reproducibles.")
    return 1


# --------------------------------------------------------------------------
def cmd_piloto(args, cfg: Config) -> int:
    """Abre el piloto de riesgo en el navegador, o imprime el plan de hoy (--texto).

    Solo escucha en este equipo salvo `--red`: la app muestra la cartera real.
    Se arranca en modo headless (sin la pregunta del correo que hace Streamlit
    la primera vez) y el navegador se abre cuando el servidor ya responde.
    """
    if args.texto:
        return _piloto_texto()
    import subprocess
    import threading
    from pathlib import Path

    app = Path(__file__).resolve().parent / "piloto" / "app.py"
    host = "0.0.0.0" if args.red else "127.0.0.1"
    puerto = _puerto_libre(args.puerto, host)
    url = f"http://127.0.0.1:{puerto}"
    orden = [sys.executable, "-m", "streamlit", "run", str(app),
             "--server.address", host, "--server.port", str(puerto),
             "--server.headless", "true", "--browser.gatherUsageStats", "false",
             "--client.toolbarMode", "minimal"]
    if args.red:
        _echo("AVISO: cualquiera en su red local podra abrir el piloto y ver su cartera.")
    _echo(f"Piloto de riesgo en {url}  (Ctrl+C en esta ventana para cerrarlo)")
    sys.stdout.flush()
    if not args.sin_navegador:
        threading.Thread(target=_abrir_cuando_responda, args=(url, puerto), daemon=True).start()
    try:
        return subprocess.call(orden)
    except KeyboardInterrupt:
        return 0


def _puerto_libre(inicio: int, host: str) -> int:
    import socket

    for puerto in range(inicio, min(inicio + 50, 65536)):
        with socket.socket() as c:  # alguien ya responde ahi?
            c.settimeout(0.2)
            if c.connect_ex(("127.0.0.1", puerto)) == 0:
                continue
        with socket.socket() as s:
            try:
                s.bind((host, puerto))
            except OSError:
                continue
        return puerto
    raise ValueError(f"no hay puertos libres entre {inicio} y {inicio + 49}; use --puerto")


def _abrir_cuando_responda(url: str, puerto: int, espera: float = 60.0) -> None:
    import socket
    import time
    import webbrowser

    fin = time.monotonic() + espera
    while time.monotonic() < fin:
        with socket.socket() as c:
            c.settimeout(0.5)
            if c.connect_ex(("127.0.0.1", puerto)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.3)


def _piloto_texto() -> int:
    from .piloto import diario, motor, sesion

    guardada, aviso = diario.leer_cartera()
    fotos, avisos = diario.leer_fotos()
    for a in [aviso, *avisos]:
        if a:
            _echo(f"AVISO: {a}")
    if not guardada.get("tenencias"):
        _echo("No hay cartera guardada. Abra la app (python -m cryptoquant piloto), anote lo que")
        _echo("tiene y pulse Guardar.")
        return 2
    aj = motor.Ajustes.desde(guardada.get("ajustes"))
    reparto = (guardada.get("ajustes") or {}).get("reparto", "actual")
    reparto = reparto if reparto in motor.REPARTOS else "actual"
    s = sesion.calcular(guardada["tenencias"], reparto, guardada.get("mezcla"), fotos, aj)
    p = s.plan
    for a in s.avisos:
        _echo(f"AVISO: {a}")
    if s.sin_precio_ahora:
        _echo(f"AVISO: sin precio de ahora para {', '.join(s.sin_precio_ahora)}; se usa el ultimo cierre")

    _echo(f"Piloto de riesgo  (cierres hasta el {p.fecha_datos.date()} UTC, precios de ahora)")
    _echo("-" * 78)
    _echo(f"  Valor de la cartera   : {p.total:,.2f} USDT")
    _echo(f"  En cripto ahora       : {p.exposicion_actual:.1%}  ({p.valor_cripto:,.2f} USDT)")
    _echo(f"  Volatilidad prevista  : {p.sigma_prevista:.1%} anual de la mezcla "
          f"(objetivo de la cartera {aj.objetivo:.0%})")
    if not aj.freno:
        freno = "desactivado"
    elif p.factor_freno < 1:
        freno = f"ACTIVO: caida {p.caida:.1%}, se invierte el {p.factor_freno:.0%} de lo normal"
    else:
        freno = f"inactivo (caida {p.caida:.1%}; empieza en -{aj.freno_inicio:.0%})"
    _echo(f"  Freno                 : {freno}")
    _echo(f"  Recomendado en cripto : {p.exposicion_objetivo:.1%}  ({p.cripto_objetivo:,.2f} USDT)")
    _echo()
    if not p.actuar:
        _echo(f"Nada que hacer: la diferencia ({abs(p.mover):,.2f} USDT, {abs(p.mover) / p.total:.1%}) "
              f"es menor que la banda ({aj.banda:.0%}) o que el minimo por orden "
              f"({aj.minimo_orden:,.2f} USDT).")
    else:
        verbo = "pasar a estables" if p.mover < 0 else "invertir en cripto"
        _echo(f"ACCION: {verbo} {abs(p.mover):,.2f} USDT")
        o = p.ordenes
        for a, fila in o[o["ejecutar"]].iterrows():
            _echo(f"  {'Comprar' if fila['importe_orden'] > 0 else 'Vender ':7s} "
                  f"{abs(fila['cantidad_orden']):>16.8f} {a:6s} {abs(fila['importe_orden']):>12,.2f} USDT"
                  f"  a {_precio(fila['precio'])}")
        if not o["ejecutar"].any():
            _echo("  Hoy no hay ninguna orden que se pueda ejecutar.")
        pequenas = o.index[~o["ejecutar"] & ~o["recortada"] & (o["importe"].abs() >= 0.01)]
        if len(pequenas):
            _echo(f"  Se omiten ordenes de menos de {aj.minimo_orden:,.2f} USDT: {', '.join(pequenas)}.")
        if o["recortada"].any():
            _echo("  Compras recortadas a lo que pagan su efectivo y las ventas de arriba: "
                  f"{', '.join(o.index[o['recortada']])}.")
    _echo()
    _echo("Solo recomienda: las ordenes las decide y ejecuta usted. Guarde la cartera al terminar.")
    return 0


# --------------------------------------------------------------------------
def cmd_dashboard(args, cfg: Config) -> int:
    """Genera el panel HTML. Solo lee: no descarga datos ni toca el diario."""
    from pathlib import Path

    from .dashboard.render import build_dashboard

    out = build_dashboard(cfg, Path(args.out) if args.out else None)
    _echo(f"Panel generado: {out}")
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    return 0


# --------------------------------------------------------------------------
def cmd_laboratorio(args, cfg: Config) -> int:
    """Recorre de las velas al modelo y deja CSV y graficos para Python y R."""
    from .laboratorio import informe

    _echo(f"Laboratorio: {args.simbolo} {args.temporalidad} desde {args.desde}")
    inf = informe.ejecutar(args.simbolo, args.temporalidad, args.desde, args.descargar,
                           args.acierto, args.rb, args.riesgo, args.capital,
                           graficos=not args.no_plots, semilla=cfg.seed,
                           objetivo_anual=args.objetivo)
    R = inf.resultados
    _echo(f"  Origen: {inf.origen} | {len(inf.velas)} velas, {len(inf.diario)} dias "
          f"| {inf.diario.index[0].date()} a {inf.diario.index[-1].date()}")
    _echo()

    d, colas = R["distribucion"], R["colas"]
    _echo("Distribucion de retornos diarios")
    _echo("-" * 78)
    _echo(f"  Curtosis en exceso {d['curtosis_exceso']:.1f} (normal: 0) | "
          f"t de Student con {d['t_grados_libertad']:.1f} grados de libertad")
    for k, fila in colas.iterrows():
        _echo(f"  Movimientos > {k} sigmas: {fila['observados']:4.0f} observados, "
              f"{fila['esperados_normal']:8.3f} esperados si fuera normal")
    _echo()

    s = R["senales"]
    _echo("Senales tecnicas (a 5 dias, en la direccion de la senal)")
    _echo("-" * 78)
    for _, f in s[s["horizonte"] == 5].iterrows():
        _echo(f"  {f['senal']:26s} n={f['eventos']:3d}  acierto {f['acierto']:.0%} "
              f"(base {f['acierto_base']:.0%})  p={f['p_valor']:.2f}")
    umbral = 0.05 / len(s)
    _echo(f"  Significativas tras Bonferroni (p < {umbral:.4f}): "
          f"{int((s['p_valor'] < umbral).sum())} de {len(s)}")
    _echo()

    lb, a = R["ljung_box"]["ljung_box_p_10"], R["arima"]
    _echo("Series de tiempo")
    _echo("-" * 78)
    est = R["estacionariedad"]["adf_p"]
    _echo(f"  ADF p: log-precio {est['log_precio']:.3f}, retorno {est['retorno_log']:.3f}")
    _echo(f"  Ljung-Box p (10 rezagos): retorno {lb['retorno']:.3f}, "
          f"retorno^2 {lb['retorno_cuadrado']:.2g}")
    for k, t in R["calendario"].items():
        _echo(f"  Por {k}: Kruskal p retorno {t.attrs['kruskal_p_retorno']:.3f}, "
              f"volatilidad {t.attrs['kruskal_p_volatilidad']:.2g}")
    _echo(f"  ARIMA{a['orden']}: RMSE {a['rmse_arima']:.3f} frente a {a['rmse_cero']:.3f} "
          f"de predecir 0 ({a['mejora_sobre_cero']:+.2%})")
    _echo()

    g, ge, p = R["garch"], R["garch_eval"], R["plan"]
    _echo("Volatilidad: GARCH(1,1)")
    _echo("-" * 78)
    _echo(f"  alpha {g['alpha']:.3f}  beta {g['beta']:.3f}  persistencia {g['persistencia']:.3f} "
          f"(vida media de un shock: {g['vida_media_dias']:.0f} dias)")
    _echo(f"  Sigma prevista para manana: {g['sigma_manana']:.2%} diaria, "
          f"{g['sigma_manana_anual']:.0%} anual")
    _echo(f"  Fuera de muestra, QLIKE (menor es mejor): GARCH {ge['qlike_garch']:.3f}, "
          f"movil 20d {ge['qlike_movil']:.3f}")
    _echo(f"  Compra a {p['precio']:,.2f}: stop {p['stop']:,.2f} ({p['distancia_stop_pct']:.1%}), "
          f"objetivo {p['objetivo']:,.2f}")
    _echo(f"  Tamano para arriesgar {args.riesgo:.1%} de {args.capital:,.0f}: "
          f"{p['nominal']:,.2f} ({p['fraccion_capital']:.0%} del capital)")
    _echo()

    rf = R["bosque"]
    _echo(f"Random forest (prueba desde {rf.inicio_prueba.date()})")
    _echo("-" * 78)
    for nombre, f in rf.metricas.iterrows():
        _echo(f"  {nombre:18s} acierto {f['acierto']:.1%}  AUC {f['auc']:.3f}")
    _echo(f"  p-valor frente a la mejor base: {rf.p_valor_vs_base:.3f}")
    _echo(f"  Variables con mas importancia: {', '.join(rf.importancia.index[:3])}")
    _echo()

    mp, mc, par = R["mc_precio"], R["mc_cuenta"], R["mc_parametros"]
    _echo("Monte Carlo")
    _echo("-" * 78)
    _echo(f"  Precio a 90 dias: mediana {mp['precio_p50']:,.0f}, "
          f"90% entre {mp['precio_p5']:,.0f} y {mp['precio_p95']:,.0f}; "
          f"prob. de tocar -20%: {mp['prob_toca_menos_20']:.0%}")
    _echo(f"  Cuenta con acierto {par['acierto']:.0%}, R:B 1:{par['riesgo_beneficio']:g}, "
          f"riesgo {par['riesgo']:.1%} por operacion, 200 operaciones:")
    _echo(f"    esperanza {mc['esperanza_por_operacion_R']:+.2f} R por operacion "
          f"(acierto minimo rentable {mc['acierto_minimo_rentable']:.0%})")
    _echo(f"    capital final mediano {mc['capital_mediano_final']:,.0f} | "
          f"prob. de perder {mc['prob_perdida']:.1%} | drawdown mediano {mc['drawdown_mediano']:.0%}")
    _echo(f"    racha perdedora mediana: {mc['racha_perdedora_mediana']:.0f} operaciones seguidas")
    _echo()
    ctl = R["control"]
    _echo(f"Control de volatilidad (objetivo {args.objetivo:.0%} anual): la idea de la tesis")
    _echo("-" * 78)
    nombres = {"estrategia": "control de volatilidad", "pasiva": "comprar y mantener",
               "pasiva_igual_vol": "mantener a igual vol (ex post)"}
    for fila, f in ctl.iterrows():
        _echo(f"  {nombres[fila]:31s} {f['rentabilidad_anual']:+6.1%} anual  vol {f['volatilidad']:5.1%}"
              f"  caida max {f['caida_maxima']:6.1%}  capital {f['capital_final']:,.0f}")
    _echo(f"  Exposicion media {ctl.attrs['exposicion_media']:.0%}. La referencia a igual vol necesito")
    _echo("  conocer la volatilidad futura; el control llego a ese riesgo sin conocerla.")
    _echo()
    _echo(f"Archivos en {inf.archivos[0].parent}")
    return 0


# --------------------------------------------------------------------------
# Riesgo
# --------------------------------------------------------------------------
_NOMBRE_METODO = {"historico": "Historico", "normal": "Normal", "t": "t de Student",
                  "garch_t": "GARCH-t"}


def cmd_riesgo(args, cfg: Config) -> int:
    """Riesgo del modelo frente a la pasiva, y si el VaR con que se mide es de fiar."""
    from .riesgo import informe

    ohlcv, closes, _ = _load(cfg, args.refresh)
    _echo("Recalculando el backtest y midiendo el riesgo (alrededor de un minuto)...")
    inf = informe.ejecutar(ohlcv, closes, cfg, n_sim=args.simulaciones)
    _echo()

    v = inf.validacion
    for alpha in sorted(v["alpha"].unique()):
        _echo(f"Backtest del VaR al {alpha:.0%} a un dia (dias invertidos)")
        _echo("-" * 78)
        _echo(f"  {'':15s}{'Metodo':14s}{'Excep.':>8s}{'Tasa':>8s}{'Kupiec p':>10s}"
              f"{'Indep. p':>10s}{'Cola':>7s}  Veredicto")
        for _, f in v[v["alpha"] == alpha].iterrows():
            _echo(f"  {f['cartera']:15s}{_NOMBRE_METODO[f['metodo']]:14s}{f['excepciones']:8d}"
                  f"{f['tasa']:8.1%}{f['p_kupiec']:10.3f}{f['p_independencia']:10.3f}"
                  f"{f['cola_ratio']:7.2f}  {_veredicto(f)}")
        _echo(f"  Tasa esperada: {1 - alpha:.0%}. Cola: perdida media en las excepciones / CVaR "
              f"previsto (1 = exacto).")
        _echo()

    a0 = v["alpha"].min()
    ok = v[(v["alpha"] == a0)].groupby("metodo")["aceptado"].all()
    fiables = [_NOMBRE_METODO[m] for m, b in ok.items() if b]
    _echo("Lectura:")
    _echo(f"  Metodos aceptados al {a0:.0%} en todas las carteras: "
          f"{', '.join(fiables) if fiables else 'ninguno'}.")
    _echo("  Un VaR rechazado por independencia falla en rachas: acierta en calma")
    _echo("  y se queda corto justo cuando el mercado cambia de regimen.")
    _echo()

    _echo(f"VaR para manana al {a0:.0%} (exposicion actual del modelo: {inf.exposicion_hoy:.0%})")
    _echo("-" * 78)
    hoy = inf.var_hoy.pivot(index="cartera", columns="metodo", values="var")
    for cartera, fila in hoy.iterrows():
        _echo(f"  {cartera:15s}" + "  ".join(f"{_NOMBRE_METODO[m]} {fila[m]:.2%}"
                                             for m in ("historico", "garch_t")))
    cap = args.capital or cfg.backtest.initial_capital
    g = hoy.loc["modelo", "garch_t"]
    _echo(f"  Con {cap:,.0f} en el modelo: en 1 de cada 20 dias se perderian mas de "
          f"{g * cap:,.0f}.")
    _echo()

    _echo("Riesgo a futuro (Monte Carlo por bloques de la historia 2021-hoy)")
    _echo("-" * 78)
    _echo(f"  {'':26s}{'Dias':>5s}{'P(perder)':>11s}{'Peor 5%':>10s}{'P(caida>10%)':>14s}"
          f"{'P(caida>20%)':>14s}")
    for _, f in inf.futuro.iterrows():
        _echo(f"  {f['serie']:26s}{f['horizonte_dias']:5d}{f['prob_perdida']:11.0%}"
              f"{f['retorno_p5']:10.1%}{f['prob_caida_10']:14.1%}{f['prob_caida_20']:14.1%}")
    _echo("  'igual_vol': la pasiva reducida hasta la volatilidad del modelo. Es la")
    _echo("  comparacion justa: cualquiera puede bajar el riesgo invirtiendo menos.")
    _echo()
    _echo(f"Archivos en {inf.archivos[0].parent}  (el panel los muestra: dashboard)")
    return 0


def _precios_cartera(cfg: Config, activos: list[str], refresh: bool) -> tuple[pd.DataFrame, str]:
    """Cierres diarios: del universo, de la cache; el resto, descargados en memoria."""
    from .laboratorio.datos import a_diario, descargar_velas

    try:
        _, closes, _ = load_universe(cfg, force_refresh=refresh)
    except Exception:  # noqa: BLE001 - sin cache ni red, se intenta activo a activo
        closes = pd.DataFrame()
    fuera = [a for a in activos if a not in closes.columns]
    extra = {}
    for a in fuera:
        v = descargar_velas(f"{a}/{cfg.data.quote}", "1d", "2021-01-01")
        extra[a] = a_diario(v)["close"] if len(v) else v["close"]
    if extra:
        closes = pd.concat([closes, pd.DataFrame(extra)], axis=1)
    return closes, ", ".join(fuera)


def cmd_cartera(args, cfg: Config) -> int:
    from .riesgo import cartera as rc

    try:
        tenencias = rc.parsear(args.tengo)
    except ValueError as exc:
        _echo(f"ERROR: {exc}")
        return 2
    activos = [a for a, v in tenencias.items() if a not in rc.EFECTIVO and v > 0]
    if not activos:
        raise ValueError("la cartera solo tiene efectivo: no hay riesgo de mercado que medir")
    closes, descargados = _precios_cartera(cfg, activos, args.refresh)
    if descargados:
        _echo(f"Fuera del universo del sistema, descargados de Binance: {descargados}")
    r = rc.analizar(tenencias, closes, alpha=args.confianza, horizonte=args.dias,
                    riesgo_por_operacion=args.riesgo)
    _echo(f"Precios de cierre del {r.fecha.date()} (UTC)")
    ultimo = closes.index.max()
    if r.fecha < ultimo:
        _echo(f"  AVISO: algun activo no tiene cierre despues del {r.fecha.date()}; hay datos "
              f"hasta el {ultimo.date()}. Se usa la ultima fecha comun a todos.")
    _echo()

    _echo(f"Su cartera: {r.total:,.2f} {cfg.data.quote}")
    _echo("-" * 78)
    for a, f in r.posiciones.iterrows():
        _echo(f"  {a:9s}{f['valor']:14,.2f}  {f['peso']:6.1%} del dinero  "
              f"{f['contribucion_riesgo']:6.1%} del riesgo")
    _echo("  Un activo puede ser el 30% del dinero y el 50% del riesgo: pesa su")
    _echo("  volatilidad y cuanto se mueve con el resto.")
    _echo()

    a = f"{args.confianza:.0%}"
    _echo(f"Un mal dia: perdida que solo se supera 1 de cada {1 / (1 - args.confianza):.0f} dias ({a})")
    _echo("-" * 78)
    val = r.validacion
    for m, f in r.var_manana.iterrows():
        ok = val.loc[m]
        if not ok["suficiente"]:
            juicio = f"sin datos suficientes: {int(ok['n'])} dias, hacen falta 250"
        elif ok["aceptado"]:
            juicio = f"fiable: fallo el {ok['tasa']:.1%}"
        elif ok["p_kupiec"] <= 0.05:
            juicio = f"NO fiable: fallo el {ok['tasa']:.1%}"
        else:
            juicio = f"NO fiable: fallo el {ok['tasa']:.1%}, pero en rachas"
        _echo(f"  {_NOMBRE_METODO[m]:14s} VaR {f['var']:10,.2f} ({f['var_pct']:5.2%})   "
              f"si se supera, de media {f['cvar']:10,.2f}   [{juicio}]")
    _echo("  'fiable': con esta misma cartera, ese metodo habria acertado en el")
    _echo(f"  pasado (Kupiec y Christoffersen, p > 0.05). Se esperaba fallar el {1 - args.confianza:.0%}.")
    _echo()

    m = r.mes
    _echo(f"Proximos {args.dias} dias (Monte Carlo, 10 000 escenarios)")
    _echo("-" * 78)
    _echo(f"  Resultado mediano     : {m['retorno_mediano']:+.1%} ({m['retorno_mediano'] * r.total:+,.2f})")
    _echo(f"  Peor 5% de escenarios : {m['retorno_p5']:+.1%} ({m['retorno_p5_dinero']:+,.2f}) o peor")
    _echo(f"  Media de ese peor 5%  : {m['cvar_5']:+.1%} ({m['cvar_5_dinero']:+,.2f})")
    _echo(f"  Probabilidad de perder: {m['prob_perdida']:.0%}")
    _echo(f"  Caida de mas del 10%  : {m['prob_caida_10']:.0%}   de mas del 20%: {m['prob_caida_20']:.0%}")
    _echo("  Las colas son lo fiable. El mediano hereda la tendencia de la historia")
    _echo("  remuestreada; no es una prevision del precio.")
    _echo()

    _echo(f"Por activo (entrada nueva arriesgando {args.riesgo:.1%} del total)")
    _echo("-" * 78)
    for act, f in r.por_activo.iterrows():
        _echo(f"  {act:6s} vol. prevista {f['sigma_anual']:5.0%} anual | mal dia (t) "
              f"{f['var_1d_posicion']:9,.2f} | stop {_precio(f['stop'])} (-{f['distancia_stop_pct']:.1%})"
              f" | tamano {f['tamano_entrada']:,.2f}")
    _echo()
    _echo("Mide riesgo; no predice precios ni es una recomendacion de inversion.")
    return 0


def cmd_paquete(args, cfg: Config) -> int:
    """CSV de respaldo para una clase: que ninguna clase dependa de una API."""
    from pathlib import Path

    from .laboratorio import paquete

    simbolos = [s for s in args.simbolos.split(",") if s.strip()]
    _echo(f"Descargando {', '.join(simbolos)} (1d desde {args.desde_diario}, "
          f"1h desde {args.desde_horario})...")
    ficha = paquete.preparar(Path(args.destino), simbolos,
                             {"1d": args.desde_diario, "1h": args.desde_horario})
    for f in ficha:
        _echo(f"  {f['archivo']:22s} {f['filas']:7d} filas  {f['desde']} a {f['hasta']}  ({f['fuente']})")
    _echo(f"Listo en {Path(args.destino).resolve()} (con LEEME.txt y huellas SHA-256).")
    return 0


def _veredicto(f) -> str:
    if not f["suficiente"]:
        return f"sin datos ({int(f['n'])} dias)"
    return "aceptado" if f["aceptado"] else "RECHAZADO"


def _precio(x: float) -> str:
    return f"{x:,.2f}" if x >= 1 else f"{x:.4g}"


def cmd_evidencia(args, cfg: Config) -> int:
    from .laboratorio import cuantitativo, variables
    from .riesgo import evidencia as ev

    if args.registro:
        regs = ev._leer_registro()
        umbral = 0.05 / max(1, ev.ideas_evaluadas())
        _echo(f"{ev.ideas_evaluadas()} ideas distintas evaluadas ({len(regs)} evaluaciones); "
              f"umbral de hoy p < {umbral:.5f}")
        for r in regs[-args.ultimas:]:
            p = r["p_valor"]
            hoy = "pasaria hoy" if r["pasa"] and p is not None and p < umbral else ""
            _echo(f"  {r['fecha_utc'][:10]}  {r['senal']:28s} h={r['parametros']['horizonte']:<3d}"
                  f" {'PASO' if r['pasa'] else 'no paso':8s} p={p if p is None else round(p, 4)}"
                  f"  {hoy}")
        _echo("Una idea que paso con el umbral de entonces debe volver a pasar con el de hoy.")
        return 0

    catalogo = list(cuantitativo.senales(variables.preparar_variables(_demo_velas())).columns)
    nombres = catalogo if args.senal == "todas" else [args.senal]
    if any(n not in catalogo for n in nombres):
        _echo(f"Senales disponibles: {', '.join(catalogo)}, o 'todas'.")
        _echo("Para una idea propia, use riesgo.evidencia.evaluar_senal desde Python.")
        return 2

    ohlcv, _, _ = _load(cfg, args.refresh)
    pasan = 0
    for nombre in nombres:
        gen = (lambda n: lambda df: cuantitativo.senales(variables.preparar_variables(df))[n])(nombre)
        v = ev.evaluar_senal(nombre, gen, ohlcv, horizonte=args.horizonte)
        pasan += v.pasa
        _echo(f"{nombre} (a {v.horizonte} dias, fuera de muestra desde {v.desde})")
        _echo(f"  {v.eventos} eventos | acierto {v.acierto:.1%} frente a {v.acierto_base:.1%} de base"
              f" | neto {v.retorno_neto_medio:+.2%} por operacion")
        _echo(f"  {'PASA' if v.pasa else 'NO PASA'}" +
              ("" if v.pasa else ": " + "; ".join(v.motivos)))
        _echo()
    _echo(f"{pasan} de {len(nombres)} pasan. Ideas distintas evaluadas hasta hoy: "
          f"{ev.ideas_evaluadas()} (umbral actual p < {0.05 / max(1, ev.ideas_evaluadas()):.5f}).")
    if pasan:
        _echo("Pasar el filtro la hace candidata, no parte del sistema: incorporarla")
        _echo("cambiaria la estrategia del forward test y exige una enmienda registrada.")
    return 0


def _en(lo: float, hi: float, con_lo: bool = False, con_hi: bool = False):
    """Tipo de argparse: numero finito entre lo y hi (limites excluidos salvo
    que se indique). Rechaza nan e inf, que float() acepta sin rechistar."""
    def tipo(texto: str) -> float:
        try:
            x = float(texto)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{texto!r} no es un numero") from None
        ok_lo = x >= lo if con_lo else x > lo
        ok_hi = x <= hi if con_hi else x < hi
        if not (np.isfinite(x) and ok_lo and ok_hi):
            raise argparse.ArgumentTypeError(
                f"{texto} fuera de rango: debe estar en {'[' if con_lo else '('}{lo:g}, "
                f"{hi:g}{']' if con_hi else ')'}")
        return x
    return tipo


def _entero(lo: int, hi: int):
    def tipo(texto: str) -> int:
        try:
            x = int(texto)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{texto!r} no es un entero") from None
        if not lo <= x <= hi:
            raise argparse.ArgumentTypeError(f"{x} fuera de rango: debe estar entre {lo} y {hi}")
        return x
    return tipo


def _sin_traceback(cmd):
    """Errores de uso y de datos como un mensaje, no como un volcado de pila.

    Solo para los comandos de laboratorio y riesgo: el resto del sistema deja
    que sus fallos lleguen enteros al log de la tarea programada.
    """
    from functools import wraps

    from .riesgo.evidencia import RegistroCorrupto

    @wraps(cmd)
    def envuelto(args, cfg):
        try:
            return cmd(args, cfg)
        except (ValueError, KeyError, FileNotFoundError, RegistroCorrupto) as exc:
            _echo(f"ERROR: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}")
            return 2
        except Exception as exc:
            if type(exc).__module__.split(".")[0] == "ccxt":  # red, simbolo inexistente...
                _echo(f"ERROR de Binance: {type(exc).__name__}: {str(exc)[:200]}")
                return 2
            raise
    return envuelto


def _demo_velas() -> pd.DataFrame:
    """Velas minimas solo para listar el catalogo de senales."""
    from .data.sources import _synthetic

    return _synthetic("DEMO/USDT", "1d", "2020-01-01", 120, seed=0)


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cryptoquant",
        description="Sistema cuantitativo de inversion en criptoactivos con control de riesgo.",
    )
    p.add_argument("-c", "--config", help="ruta a config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--refresh", action="store_true", help="ignorar cache y redescargar")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="diagnostico econometrico del universo")
    a.add_argument("--fast", action="store_true", help="EWMA en lugar de GARCH")
    a.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("pairs", help="pares cointegrados")
    pr.add_argument("--top", type=int, default=10)
    pr.set_defaults(func=cmd_pairs)

    b = sub.add_parser("backtest", help="backtest walk-forward")
    b.add_argument("--method", default="hrp",
                   choices=["hrp", "risk_parity", "min_variance", "equal"])
    b.add_argument("--compare", action="store_true", help="comparar contra variantes")
    b.add_argument("--no-plots", action="store_true")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("recommend", help="asignacion recomendada actual")
    r.add_argument("--method", default="hrp",
                   choices=["hrp", "risk_parity", "min_variance", "equal"])
    r.add_argument("--capital", type=float, help="capital a asignar")
    r.set_defaults(func=cmd_recommend)

    pl = sub.add_parser("piloto", help="piloto de riesgo: cuanto tener en cripto (app en el navegador)")
    pl.add_argument("--texto", action="store_true",
                    help="imprimir el plan de hoy en la consola, sin abrir la app")
    pl.add_argument("--puerto", type=_entero(1024, 65535), default=8501)
    pl.add_argument("--red", action="store_true",
                    help="aceptar conexiones de la red local (por defecto, solo este equipo)")
    pl.add_argument("--sin-navegador", action="store_true", help="no abrir el navegador")
    pl.set_defaults(func=_sin_traceback(cmd_piloto))

    d = sub.add_parser("dashboard", help="generar el panel visual (HTML)")
    d.add_argument("--open", action="store_true", help="abrirlo en el navegador")
    d.add_argument("--out", help="ruta de salida (por defecto reports/dashboard.html)")
    d.set_defaults(func=cmd_dashboard)

    lab = sub.add_parser("laboratorio",
                         help="de las velas al modelo, paso a paso (CSV para la version en R)")
    lab.add_argument("--simbolo", default="BTC")
    lab.add_argument("--temporalidad", default="1h", choices=["1h", "4h", "1d"],
                     help="intradia permite ver patrones por hora")
    lab.add_argument("--desde", default="2022-01-01")
    lab.add_argument("--descargar", action="store_true",
                     help="volver a descargar aunque ya exista el CSV")
    lab.add_argument("--acierto", type=_en(0, 1, con_lo=True, con_hi=True), default=0.45,
                     help="Monte Carlo de la cuenta")
    lab.add_argument("--rb", type=_en(0, 100), default=2.0, help="relacion riesgo-beneficio 1:rb")
    lab.add_argument("--riesgo", type=_en(0, 1), default=0.01,
                     help="fraccion arriesgada por operacion")
    lab.add_argument("--capital", type=_en(0, 1e15), default=10_000.0)
    lab.add_argument("--objetivo", type=_en(0, 5), default=0.15,
                     help="volatilidad anual objetivo del control de volatilidad")
    lab.add_argument("--no-plots", action="store_true")
    lab.set_defaults(func=_sin_traceback(cmd_laboratorio))

    pq = sub.add_parser("paquete", help="CSV de respaldo para una clase (Drive, Posit Cloud)")
    pq.add_argument("--destino", default="clase/01_primera_clase/datos")
    pq.add_argument("--simbolos", default="BTC,ETH,SOL")
    pq.add_argument("--desde-diario", default="2020-01-01")
    pq.add_argument("--desde-horario", default="2025-01-01")
    pq.set_defaults(func=_sin_traceback(cmd_paquete))

    rg = sub.add_parser("riesgo", help="riesgo del modelo frente a la pasiva y backtest del VaR")
    rg.add_argument("--capital", type=_en(0, 1e15), help="para expresar el VaR en dinero")
    rg.add_argument("--simulaciones", type=_entero(100, 1_000_000), default=10_000)
    rg.set_defaults(func=_sin_traceback(cmd_riesgo))

    ca = sub.add_parser("cartera", help="riesgo de su cartera: VaR, escenarios, stop y tamano")
    ca.add_argument("--tengo", required=True,
                    help="unidades por activo, p. ej. 'BTC=0.05,ETH=1.2,USDT=500'")
    ca.add_argument("--confianza", type=_en(0.5, 1), default=0.95)
    ca.add_argument("--dias", type=_entero(1, 3650), default=21, help="horizonte del Monte Carlo")
    ca.add_argument("--riesgo", type=_en(0, 1, con_hi=True), default=0.01,
                    help="fraccion del total arriesgada por entrada nueva")
    ca.set_defaults(func=_sin_traceback(cmd_cartera))

    ev = sub.add_parser("evidencia", help="filtro que toda senal debe pasar antes de proponerse")
    ev.add_argument("--senal", default="todas", help="nombre de la senal del laboratorio, o 'todas'")
    ev.add_argument("--horizonte", type=_entero(1, 365), default=5)
    ev.add_argument("--registro", action="store_true", help="ver las ideas ya evaluadas")
    ev.add_argument("--ultimas", type=_entero(1, 100_000), default=20)
    ev.set_defaults(func=_sin_traceback(cmd_evidencia))

    # --- forward test -----------------------------------------------------
    f = sub.add_parser("forward", help="test prospectivo con diario sellado")
    fsub = f.add_subparsers(dest="forward_command", required=True)

    fi = fsub.add_parser("init", help="pre-registrar las hipotesis (una sola vez)")
    fi.add_argument("--overwrite", action="store_true",
                    help="reescribir el pre-registro; invalida el experimento")
    fi.set_defaults(func=cmd_forward_init)

    fa = fsub.add_parser("amend", help="registrar la enmienda E1 (calendario del backtest)")
    fa.set_defaults(func=cmd_forward_amend)

    fr = fsub.add_parser("record", help="registrar la decision de hoy")
    fr.add_argument("--method", default="hrp",
                    choices=["hrp", "risk_parity", "min_variance", "equal"])
    fr.add_argument("--capital", type=float)
    fr.add_argument("--date", help="fecha de decision (por defecto, la ultima barra)")
    fr.add_argument("--note", default="", help="anotacion libre")
    fr.add_argument("--dry-run", action="store_true", help="calcular sin escribir")
    fr.add_argument("--if-new", action="store_true",
                    help="modo desatendido: salir sin error si ya esta registrada")
    fr.set_defaults(func=cmd_forward_record)

    fs = fsub.add_parser("status", help="salud del experimento y de la automatizacion")
    fs.add_argument("--log", action="store_true", help="mostrar el log de la tarea")
    fs.set_defaults(func=cmd_forward_status)

    fv = fsub.add_parser("verify", help="comprobar que nadie ha tocado el pasado")
    fv.set_defaults(func=cmd_forward_verify)

    fp = fsub.add_parser("report", help="contrastar las hipotesis pre-registradas")
    fp.set_defaults(func=cmd_forward_report)

    fx = fsub.add_parser("reproduce", help="recomputar decisiones pasadas (H2)")
    fx.add_argument("--method", default="hrp",
                    choices=["hrp", "risk_parity", "min_variance", "equal"])
    fx.add_argument("--last", type=int, default=10,
                    help="cuantas anotaciones recientes comprobar (0 = todas)")
    fx.set_defaults(func=cmd_forward_reproduce)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = Config.load(args.config)
    np.random.seed(cfg.seed)
    from .forward.journal import WorkingCopyError

    try:
        return args.func(args, cfg)
    except WorkingCopyError as exc:
        _echo(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        _echo("\nInterrumpido.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
