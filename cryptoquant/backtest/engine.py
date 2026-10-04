"""Motor de backtest walk-forward.

Principio rector: en la barra `t` el sistema solo puede ver informacion
disponible al cierre de `t`, y los retornos se aplican en `t+1`. El modelo se
reentrena periodicamente usando exclusivamente datos anteriores, purgados por
el horizonte de etiquetado. Los costes de transaccion se cobran sobre la
rotacion efectiva.

Un backtest sin estas tres cosas produce curvas espectaculares que no se
reproducen con dinero real.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, annualization_factor
from ..econometrics.volatility import ledoit_wolf_cov
from ..features.technical import build_features, log_returns
from ..models.labeling import binary_target, sample_weights_by_uniqueness, triple_barrier_labels
from ..models.predictor import ProbabilityModel
from ..portfolio.optimizer import build_weights
from ..portfolio.sizing import decide_exposure, portfolio_volatility

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    gross_exposure: pd.Series
    benchmark_returns: pd.Series
    diagnostics: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    model_reports: list[dict] = field(default_factory=list)
    # Pesos objetivo fijados en cada barra de rebalanceo (antes de la deriva).
    targets: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------
# Calendario (compartido por el backtest y el modo en vivo)
# --------------------------------------------------------------------------
def effective_retrain_every(bt) -> int:
    """Cada cuantas barras reentrena de verdad el backtest.

    `run` solo reentrena en barras de rebalanceo, asi que con retrain_every=63
    y rebalance_every=5 lo hace cada 65. El modo en vivo usa esta cifra, no la
    nominal, para seguir exactamente el mismo calendario.
    """
    k = bt.rebalance_every
    return -(-bt.retrain_every // k) * k


def scheduled_bars(anchor_pos: int, i: int, bt) -> tuple[int, int]:
    """Ultima barra de rebalanceo y ultima de reentrenamiento programadas <= i.

    El calendario se ancla en `anchor_pos`, igual que `run` lo ancla en su
    primera barra operable.
    """
    if i < anchor_pos:
        raise ValueError("la fecha de decision es anterior al ancla del calendario")
    off = i - anchor_pos
    per = effective_retrain_every(bt)
    return (anchor_pos + off // bt.rebalance_every * bt.rebalance_every,
            anchor_pos + off // per * per)


def bar_position(index: pd.Index, when) -> int:
    """Posicion de la ultima barra con fecha <= `when` (None: la ultima)."""
    if when is None:
        return len(index) - 1
    ts = pd.Timestamp(when)
    if ts.tz is None and index.tz is not None:
        ts = ts.tz_localize(index.tz)
    pos = int(index.searchsorted(ts, side="right") - 1)
    if pos < 0:
        raise ValueError(f"{when} es anterior al inicio del historico")
    return pos


class WalkForwardBacktest:
    def __init__(self, cfg: Config, portfolio_method: str = "hrp", use_ml: bool = True):
        self.cfg = cfg
        self.ann = annualization_factor(cfg.data.timeframe)
        self.portfolio_method = portfolio_method
        self.use_ml = use_ml

    # ------------------------------------------------------------------
    # Preparacion
    # ------------------------------------------------------------------
    def prepare(self, ohlcv: dict[str, pd.DataFrame], closes: pd.DataFrame):
        """Precalcula features y etiquetas por activo.

        Las etiquetas miran al futuro por construccion. Eso es legitimo porque
        solo se usan para entrenar sobre periodos ya cerrados y purgados; nunca
        entran en la decision de la barra actual.
        """
        feats: dict[str, pd.DataFrame] = {}
        labels: dict[str, pd.DataFrame] = {}
        rets = log_returns(closes)

        for sym, df in ohlcv.items():
            feats[sym] = build_features(df, self.ann, self.cfg.signal)
            # Volatilidad diaria causal para escalar las barreras.
            sigma_d = rets[sym].rolling(20, min_periods=10).std()
            labels[sym] = triple_barrier_labels(
                df["close"], sigma_d,
                horizon=self.cfg.signal.label_horizon,
                profit_mult=self.cfg.signal.label_profit_mult,
                stop_mult=self.cfg.signal.label_stop_mult,
            )
        return feats, labels, rets

    def _pooled_training_set(self, feats, labels, upto: int, index: pd.Index):
        """Apila las observaciones de todos los activos en un unico panel.

        Entrenar un modelo por activo desperdicia datos: con 2000 barras y 8
        activos, el panel agrupado da ~16000 observaciones y el modelo aprende
        relaciones transversales (que configuracion de momentum/vol funciona)
        en lugar de memorizar el historico de un solo simbolo.
        """
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
        X = pd.concat(X_parts, axis=0).reset_index(drop=True)
        y = pd.concat(y_parts, axis=0).reset_index(drop=True)
        w = pd.concat(w_parts, axis=0).reset_index(drop=True)
        return X, y, w

    # ------------------------------------------------------------------
    # Senal
    # ------------------------------------------------------------------
    def _econometric_score(self, feats: dict[str, pd.DataFrame], t_label) -> pd.Series:
        """Puntuacion transversal basada en momentum ajustado por riesgo.

        Es la senal clasica, sin machine learning, que se combina con la del
        modelo. Tener dos fuentes independientes reduce la dependencia de que
        el ML acierte.
        """
        scores = {}
        for sym, f in feats.items():
            if t_label not in f.index:
                continue
            row = f.loc[t_label]
            mom = np.nanmean([row.get(f"mom_{w}", np.nan) for w in self.cfg.signal.momentum_windows])
            if not np.isfinite(mom):
                scores[sym] = 0.5
                continue
            # Compresion logistica a (0,1). La escala 1.5 hace que un momentum
            # de +1 sigma se traduzca en una probabilidad implicita de ~0.65.
            scores[sym] = float(1.0 / (1.0 + np.exp(-mom / 1.5)))
        return pd.Series(scores)

    # ------------------------------------------------------------------
    # Decision (compartida por el backtest y el modo en vivo)
    # ------------------------------------------------------------------
    def _decide(self, feats, log_ret, i: int, index, symbols: list[str],
                feature_cols: list[str], model, current_drawdown: float,
                current_w: pd.Series) -> tuple[pd.Series, dict]:
        """Calcula los pesos objetivo para la barra `i`.

        Este metodo es el unico sitio donde se decide. El backtest y el registro
        en vivo lo llaman igual, de modo que no pueden divergir: si la decision
        de hoy difiere de la que el backtest reproduce para hoy, el problema
        esta en los datos, no en dos implementaciones distintas.
        """
        cfg, bt, risk = self.cfg, self.cfg.backtest, self.cfg.risk
        t = index[i]

        live = pd.DataFrame({s: feats[s].loc[t] for s in symbols}).T
        live = live.reindex(columns=feature_cols)
        valid = live.notna().all(axis=1)

        p_ml = pd.Series(0.5, index=symbols)
        if self.use_ml and model is not None and model.model is not None and bool(valid.any()):
            p_ml.loc[valid] = model.predict_proba(live.loc[valid])

        p_econ = self._econometric_score(feats, t).reindex(symbols).fillna(0.5)
        use_ml_now = self.use_ml and model is not None and model.model is not None
        wgt = cfg.signal.ml_weight if use_ml_now else 0.0
        prob = wgt * p_ml + (1 - wgt) * p_econ

        # --- Filtro de regimen por activo -------------------------------
        regime = live["regime_up"].fillna(0.0) > 0.5
        eligible = [s for s in symbols
                    if prob[s] >= cfg.signal.min_probability and bool(regime.get(s, False))]

        decision = None
        port_vol = np.nan
        if not eligible:
            target_w = pd.Series(0.0, index=symbols)
        else:
            hist = log_ret[eligible].iloc[max(0, i - risk.covariance_lookback):i + 1]
            cov = ledoit_wolf_cov(hist)
            base_w = build_weights(cov, eligible, self.portfolio_method,
                                   risk.max_weight_per_asset)
            # Inclinacion por conviccion: el peso base de HRP se ajusta segun la
            # probabilidad relativa, acotado para que la senal no destruya la
            # diversificacion que aporta HRP.
            tilt = (prob[eligible] / prob[eligible].mean()).clip(0.5, 1.5)
            tilted = base_w * tilt
            tilted = tilted / tilted.sum()

            port_vol = portfolio_volatility(tilted.to_numpy(dtype=float), cov, self.ann)
            payoff = cfg.signal.label_profit_mult / cfg.signal.label_stop_mult

            decision = decide_exposure(
                prob_win=float(prob[eligible].mean()),
                payoff_ratio=payoff,
                forecast_vol=port_vol,
                current_drawdown=current_drawdown,
                # Amplitud de mercado: un activo puede estar sobre su media
                # mientras el conjunto se hunde. Si menos de la mitad del
                # universo esta en tendencia alcista, se recorta aunque los
                # elegidos pinten bien.
                regime_ok=bool(regime.mean() >= 0.5),
                risk_cfg=risk,
            )
            target_w = pd.Series(0.0, index=symbols)
            target_w.loc[eligible] = tilted * decision.gross_exposure
            # Tope de concentracion sobre el peso absoluto. Con pocos activos
            # elegibles el reparto relativo no puede respetarlo (3 activos no
            # suman 1 con tope 0.30), pero el absoluto si. El exceso recortado
            # se queda en efectivo, que es la decision conservadora.
            target_w = target_w.clip(upper=risk.max_weight_per_asset)

        # Filtro de rotacion: no mover pesos por cambios triviales.
        delta = (target_w - current_w).abs()
        target_w = target_w.where(delta >= bt.min_trade_threshold, current_w)

        diag = {
            "date": t,
            "n_eligible": len(eligible),
            "mean_prob": float(prob.mean()),
            "port_vol_forecast": port_vol,
            "gross_target": float(target_w.sum()),
            "kelly_raw": decision.kelly_raw if decision else 0.0,
            "conviction": decision.conviction if decision else 0.0,
            "regime_scalar": decision.regime_scalar if decision else 0.0,
            "vol_scalar": decision.vol_scalar if decision else 0.0,
            "dd_scalar": decision.drawdown_scalar if decision else 1.0,
            "reason": decision.reason if decision else "ningun activo elegible",
            "detail": decision.detail if decision else "",
        }
        return target_w, diag

    def decide_at(self, ohlcv: dict[str, pd.DataFrame], closes: pd.DataFrame,
                  when: pd.Timestamp | str | None = None,
                  current_drawdown: float = 0.0,
                  current_weights: pd.Series | None = None,
                  anchor: pd.Timestamp | str | None = None,
                  last_rebalance: pd.Timestamp | str | None = None,
                  previous_target: pd.Series | None = None) -> tuple[pd.Series, dict]:
        """Decision para una fecha concreta, sin simular todo el historico.

        Es lo que usa el registro diario del forward test, y tambien lo que
        permite recomputar una decision pasada para comprobar que sigue
        saliendo igual.

        Con `anchor` sigue el calendario de `run`: rebalancea cada
        `rebalance_every` barras contadas desde el ancla y usa el modelo del
        ultimo reentrenamiento programado (cada `effective_retrain_every`). Los
        dias intermedios devuelve `previous_target`, como hace el backtest. Si
        un rebalanceo programado cayo en un dia sin anotar, se hace en la
        primera llamada posterior ("rebalanceo tardio"): el pasado no se opera.

        Sin `anchor` reentrena con todo lo disponible y decide siempre. Asi
        funcionaba el protocolo original del forward test, y se conserva para
        poder reproducir aquellas anotaciones.

        Args:
            when: fecha de decision. Por defecto, la ultima barra disponible.
            current_drawdown: drawdown vigente de la cartera real (<= 0).
            current_weights: pesos actuales ya derivados por precio, para el
                filtro de rotacion (el backtest usa los derivados).
            anchor: primera barra del calendario.
            last_rebalance: fecha del ultimo rebalanceo hecho, o None.
            previous_target: pesos objetivo vigentes, para los dias sin rebalanceo.
        """
        index = closes.index
        symbols = list(closes.columns)
        i = bar_position(index, when)

        min_bars = max(self.cfg.backtest.warmup, self.cfg.signal.regime_ma + 20)
        if i < min_bars:
            raise ValueError(
                f"historico insuficiente antes de {index[i].date()}: "
                f"{i} barras, se requieren {min_bars}"
            )

        info: dict = {"decision_date": str(index[i].date()), "bars_available": i + 1}
        train_upto = i
        if anchor is not None:
            a = bar_position(index, anchor)
            if str(index[a].date()) != str(pd.Timestamp(anchor).date()):
                raise ValueError(f"el ancla {anchor} no es una barra del historico")
            sched, train_upto = scheduled_bars(a, i, self.cfg.backtest)
            last = None if last_rebalance is None else bar_position(index, last_rebalance)
            step = pd.Timedelta(self.cfg.data.timeframe) * self.cfg.backtest.rebalance_every
            info.update({
                "scheduled_rebalance": str(index[sched].date()),
                "next_rebalance": str((index[sched] + step).date()),
                "model_upto": str(index[train_upto].date()),
            })
            if last is not None and last >= sched:
                if previous_target is None:
                    raise ValueError("un dia sin rebalanceo necesita los pesos objetivo vigentes")
                info["action"] = "mantener"
                return previous_target.reindex(symbols).fillna(0.0), info
            info["action"] = "rebalanceo" if i == sched else "rebalanceo tardio"

        feats, labels, log_ret = self.prepare(ohlcv, closes)
        feature_cols = list(feats[symbols[0]].columns)
        model = ProbabilityModel(seed=self.cfg.seed)
        if self.use_ml:
            X, y, w = self._pooled_training_set(feats, labels, train_upto, index)
            if X is not None:
                model.fit(X, y, w)

        cw = (pd.Series(0.0, index=symbols) if current_weights is None
              else current_weights.reindex(symbols).fillna(0.0))
        target_w, diag = self._decide(feats, log_ret, i, index, symbols,
                                      feature_cols, model, current_drawdown, cw)
        diag.update(info)
        return target_w, diag

    # ------------------------------------------------------------------
    # Ejecucion
    # ------------------------------------------------------------------
    def run(self, ohlcv: dict[str, pd.DataFrame], closes: pd.DataFrame) -> BacktestResult:
        cfg, bt, risk = self.cfg, self.cfg.backtest, self.cfg.risk
        feats, labels, log_ret = self.prepare(ohlcv, closes)
        simple_ret = closes.pct_change().fillna(0.0)

        index = closes.index
        symbols = list(closes.columns)
        feature_cols = list(feats[symbols[0]].columns)
        n = len(index)
        start = max(bt.warmup, cfg.signal.regime_ma + 20)
        if n <= start + 60:
            raise ValueError(f"historico insuficiente: {n} barras, se requieren > {start + 60}")

        model = ProbabilityModel(seed=cfg.seed)
        model_reports: list[dict] = []
        last_train = -10**9

        equity = [bt.initial_capital]
        eq_index = [index[start]]
        current_w = pd.Series(0.0, index=symbols)
        target_w = pd.Series(0.0, index=symbols)
        rows_w, rows_diag, rows_target = [], [], []
        turnover_l, costs_l, gross_l, ret_l = [], [], [], []
        peak = bt.initial_capital

        for i in range(start, n - 1):
            t = index[i]
            rebalance = (i - start) % bt.rebalance_every == 0

            if rebalance:
                # --- Reentrenamiento walk-forward -------------------------
                if self.use_ml and (i - last_train) >= bt.retrain_every:
                    X, y, w = self._pooled_training_set(feats, labels, i, index)
                    if X is not None:
                        model.fit(X, y, w)
                        last_train = i
                        if model.report:
                            model_reports.append({
                                "date": t,
                                "auc": model.report.auc_train,
                                "brier": model.report.brier_train,
                                "n_train": model.report.n_train,
                                "base_rate": model.report.base_rate,
                            })

                dd_now = equity[-1] / peak - 1.0
                target_w, diag = self._decide(
                    feats, log_ret, i, index, symbols, feature_cols,
                    model, dd_now, current_w,
                )
                rows_diag.append(diag)
                rows_target.append(pd.Series(target_w, name=t))

            # --- Costes de rebalanceo (se cobran en t) --------------------
            trade = float((target_w - current_w).abs().sum())
            cost = trade * (bt.fee + bt.slippage)
            current_w = target_w.copy()

            # --- Retorno del periodo t -> t+1 -----------------------------
            r_next = simple_ret.iloc[i + 1]
            port_ret = float((current_w * r_next).sum()) - cost

            new_eq = equity[-1] * (1 + port_ret)
            equity.append(new_eq)
            eq_index.append(index[i + 1])
            peak = max(peak, new_eq)

            # Los pesos derivan con el precio hasta el proximo rebalanceo.
            grown = current_w * (1 + r_next)
            denom = 1 + port_ret
            current_w = grown / denom if abs(denom) > 1e-9 else grown

            rows_w.append(pd.Series(current_w, name=index[i + 1]))
            turnover_l.append(trade)
            costs_l.append(cost)
            gross_l.append(float(current_w.sum()))
            ret_l.append(port_ret)

        eq = pd.Series(equity, index=pd.Index(eq_index, name="date"))
        rets = pd.Series(ret_l, index=pd.Index(eq_index[1:], name="date"))
        bench = simple_ret.loc[rets.index].mean(axis=1)  # equiponderada buy&hold

        return BacktestResult(
            equity=eq,
            returns=rets,
            weights=pd.DataFrame(rows_w),
            gross_exposure=pd.Series(gross_l, index=rets.index),
            benchmark_returns=bench,
            diagnostics=pd.DataFrame(rows_diag).set_index("date") if rows_diag else pd.DataFrame(),
            turnover=pd.Series(turnover_l, index=rets.index),
            costs=pd.Series(costs_l, index=rets.index),
            model_reports=model_reports,
            targets=pd.DataFrame(rows_target),
        )


def buy_and_hold(closes: pd.DataFrame, start_idx: int = 0) -> pd.Series:
    """Referencia: cartera equiponderada comprada y mantenida, sin rebalanceo."""
    sub = closes.iloc[start_idx:]
    return (sub / sub.iloc[0]).mean(axis=1)
