"""Machine learning: random forest que clasifica si la vela siguiente sube.

Tres reglas que separan un resultado creible de uno que no se reproduce:

1. Entrenar con el pasado y probar con el futuro. Un `train_test_split`
   aleatorio mete dias de 2026 en el entrenamiento y evalua sobre 2023: el
   modelo "adivina" un pasado que ya ha visto rodeado por su futuro.
2. Dejar un hueco (embargo) entre ambos tramos. El objetivo de la ultima fila
   de entrenamiento se calcula con el cierre de la primera de prueba.
3. Compararse con una linea base. Si el mercado subio el 53% de los dias,
   decir siempre "sube" acierta el 53%. Un modelo con 54% no ha aprendido
   casi nada; uno con 51% es peor que no hacer nada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .variables import matriz_modelo


@dataclass
class ResultadoBosque:
    modelo: RandomForestClassifier
    metricas: pd.DataFrame        # modelo y lineas base, sobre el tramo de prueba
    importancia: pd.DataFrame     # por impureza y por permutacion
    prueba: pd.DataFrame          # real, probabilidad y prediccion por fecha
    inicio_prueba: pd.Timestamp
    p_valor_vs_base: float        # binomial: acierto del modelo > el de la mejor base


def dividir_temporal(X: pd.DataFrame, y: pd.Series, frac_entrenamiento: float = 0.7,
                     embargo: int = 1):
    n = int(len(X) * frac_entrenamiento)
    return (X.iloc[:n], y.iloc[:n]), (X.iloc[n + embargo:], y.iloc[n + embargo:])


def entrenar_bosque(variables: pd.DataFrame, columnas: list[str] | None = None,
                    frac_entrenamiento: float = 0.7, n_arboles: int = 400,
                    min_hoja: int = 25, semilla: int = 42) -> ResultadoBosque:
    X, y = matriz_modelo(variables, columnas)
    if len(X) < 200:
        raise ValueError(f"solo {len(X)} filas completas tras el arranque de los indicadores; "
                         "hacen falta al menos 200 (amplie --desde)")
    (Xtr, ytr), (Xte, yte) = dividir_temporal(X, y, frac_entrenamiento)
    if ytr.nunique() < 2 or yte.nunique() < 2:
        raise ValueError("el entrenamiento o la prueba contienen una sola clase (solo sube o "
                         "solo baja): no hay nada que clasificar")

    # min_samples_leaf alto y max_features bajo: con senal debil, un arbol
    # profundo memoriza ruido del entrenamiento.
    rf = RandomForestClassifier(n_estimators=n_arboles, min_samples_leaf=min_hoja,
                                max_features="sqrt", random_state=semilla)
    rf.fit(Xtr, ytr)
    prob = rf.predict_proba(Xte)[:, 1]
    pred = (prob > 0.5).astype(int)

    mayoritaria = int(ytr.mean() >= 0.5)
    base_mayoritaria = np.full(len(yte), mayoritaria)
    # Persistencia: manana hace lo mismo que hoy.
    base_persistencia = (Xte["retorno"] > 0).astype(int).to_numpy()

    def fila(p, s=None):
        return {"acierto": accuracy_score(yte, p),
                "acierto_equilibrado": balanced_accuracy_score(yte, p),
                "auc": roc_auc_score(yte, s) if s is not None else 0.5}

    metricas = pd.DataFrame({
        "random_forest": fila(pred, prob),
        f"siempre_{'sube' if mayoritaria else 'baja'}": fila(base_mayoritaria),
        "persistencia": fila(base_persistencia),
    }).T

    mejor_base = metricas.drop(index="random_forest")["acierto"].max()
    aciertos = int((pred == yte.to_numpy()).sum())
    p_valor = float(stats.binomtest(aciertos, len(yte), mejor_base, alternative="greater").pvalue)

    perm = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=semilla,
                                  scoring="roc_auc")
    importancia = pd.DataFrame({
        "impureza": rf.feature_importances_,
        "permutacion_auc": perm.importances_mean,
        "permutacion_sd": perm.importances_std,
    }, index=X.columns).sort_values("permutacion_auc", ascending=False)

    prueba = pd.DataFrame({"sube": yte, "probabilidad": prob, "prediccion": pred}, index=yte.index)
    return ResultadoBosque(rf, metricas, importancia, prueba, yte.index[0], p_valor)
