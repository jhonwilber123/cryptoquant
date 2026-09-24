"""Modelo predictivo de probabilidad de exito.

El modelo NO predice el precio. Predice P(el trade sea rentable | features),
que es la magnitud que necesita el dimensionamiento por Kelly. Predecir precios
es un problema mal condicionado; estimar una probabilidad calibrada y dimensionar
en consecuencia es tratable.

Decisiones deliberadas:
- Gradient boosting sobre arboles: robusto a features en escalas distintas, no
  extrapola fuera del rango visto (deseable aqui) y tolera no linealidades.
- Calibracion isotonica: un modelo con buen AUC pero probabilidades mal
  calibradas destruye el sizing de Kelly. La calibracion se ajusta en un bloque
  temporal separado, nunca con CV aleatoria.
- Pesos por unicidad: corrige el solapamiento de horizontes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# sklearn avisa sobre `delayed`/`Parallel` desde sus propias internas al
# entrenar en bucle. Es ruido de la libreria, no de este codigo, y al
# reentrenar decenas de veces inunda la salida del backtest.
warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning
)


@dataclass
class ModelReport:
    n_train: int
    n_features: int
    auc_train: float
    brier_train: float
    base_rate: float
    feature_importance: dict[str, float] = field(default_factory=dict)


class ProbabilityModel:
    """Envoltorio sobre gradient boosting + calibracion isotonica."""

    def __init__(self, seed: int = 42, calibrate: bool = True):
        self.seed = seed
        self.calibrate = calibrate
        self.model = None
        self.calibrator = None
        self.columns: list[str] = []
        self.report: ModelReport | None = None
        self._base_rate: float = 0.5

    def _build(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=3,          # arboles poco profundos: menos sobreajuste
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.seed,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None
            ) -> "ProbabilityModel":
        from sklearn.metrics import brier_score_loss, roc_auc_score

        mask = y.notna() & X.notna().any(axis=1)
        Xc, yc = X.loc[mask], y.loc[mask].astype(int)
        if len(Xc) < 100 or yc.nunique() < 2:
            self.model = None
            self._base_rate = float(yc.mean()) if len(yc) else 0.5
            return self

        self.columns = list(Xc.columns)
        w = None
        if sample_weight is not None:
            w = sample_weight.loc[mask].fillna(1.0).to_numpy(dtype=float)

        # La calibracion necesita datos que el modelo base no haya visto. Se
        # reserva el ultimo 20% temporal (nunca una muestra aleatoria).
        if self.calibrate and len(Xc) >= 300:
            cut = int(len(Xc) * 0.8)
            X_fit, y_fit = Xc.iloc[:cut], yc.iloc[:cut]
            X_cal, y_cal = Xc.iloc[cut:], yc.iloc[cut:]
            w_fit = w[:cut] if w is not None else None
        else:
            X_fit, y_fit, w_fit = Xc, yc, w
            X_cal = y_cal = None

        self.model = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.model.fit(X_fit.to_numpy(dtype=float), y_fit.to_numpy(),
                           sample_weight=w_fit)

        if X_cal is not None and len(X_cal) >= 50 and y_cal.nunique() >= 2:
            from sklearn.isotonic import IsotonicRegression

            raw = self.model.predict_proba(X_cal.to_numpy(dtype=float))[:, 1]
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
            self.calibrator.fit(raw, y_cal.to_numpy())
        else:
            self.calibrator = None

        self._base_rate = float(yc.mean())
        p_train = self.predict_proba(Xc)
        try:
            auc = float(roc_auc_score(yc, p_train))
            brier = float(brier_score_loss(yc, p_train))
        except ValueError:
            auc, brier = np.nan, np.nan

        self.report = ModelReport(
            n_train=len(Xc),
            n_features=len(self.columns),
            auc_train=auc,
            brier_train=brier,
            base_rate=self._base_rate,
            feature_importance={},
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(trade rentable). Si el modelo no pudo ajustarse, devuelve la tasa
        base, que equivale a no tener opinion."""
        if self.model is None:
            return np.full(len(X), self._base_rate)
        Xa = X.reindex(columns=self.columns).to_numpy(dtype=float)
        p = self.model.predict_proba(Xa)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.01, 0.99)

    def permutation_importance(self, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5
                               ) -> pd.Series:
        """Importancia por permutacion sobre datos no vistos.

        Se prefiere a la importancia interna del arbol, que sobrevalora las
        features de alta cardinalidad.
        """
        from sklearn.inspection import permutation_importance as _pi

        if self.model is None:
            return pd.Series(dtype=float)
        mask = y.notna()
        Xc, yc = X.loc[mask].reindex(columns=self.columns), y.loc[mask].astype(int)
        if len(Xc) < 50 or yc.nunique() < 2:
            return pd.Series(dtype=float)
        res = _pi(
            self.model, Xc.to_numpy(dtype=float), yc.to_numpy(),
            n_repeats=n_repeats, random_state=self.seed, scoring="roc_auc",
        )
        return pd.Series(res.importances_mean, index=self.columns).sort_values(ascending=False)
