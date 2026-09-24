"""Validacion temporal con purga y embargo.

La validacion cruzada aleatoria de sklearn es invalida en series temporales:
mezcla pasado y futuro y produce metricas infladas que no se reproducen en
real. Aqui se usa validacion walk-forward con dos protecciones adicionales:

- Purga: se eliminan del entrenamiento las muestras cuyo horizonte de
  etiquetado se solapa con el periodo de test.
- Embargo: se descarta un margen adicional tras el test para neutralizar la
  autocorrelacion serial residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def purged_walk_forward(
    index: pd.Index,
    n_splits: int = 5,
    train_window: int | None = None,
    test_size: int | None = None,
    horizon: int = 10,
    embargo: float = 0.01,
) -> Iterator[WalkForwardSplit]:
    """Genera particiones walk-forward purgadas.

    Args:
        index: indice temporal completo.
        n_splits: numero de pliegues.
        train_window: si se indica, ventana deslizante; si no, expansiva.
        test_size: tamano de cada bloque de test.
        horizon: horizonte del etiquetado, en barras (define la purga).
        embargo: fraccion del total a embargar tras el test.
    """
    n = len(index)
    if test_size is None:
        test_size = max(20, n // (n_splits + 1))
    embargo_bars = int(n * embargo)

    first_test = n - n_splits * test_size
    if first_test <= horizon + 50:
        raise ValueError(
            f"historico insuficiente: {n} barras para {n_splits} pliegues de {test_size}"
        )

    for k in range(n_splits):
        test_start = first_test + k * test_size
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 5:
            continue

        train_hi = test_start - horizon - embargo_bars  # purga + embargo
        train_lo = 0 if train_window is None else max(0, train_hi - train_window)
        if train_hi - train_lo < 100:
            continue

        yield WalkForwardSplit(
            train_idx=np.arange(train_lo, train_hi),
            test_idx=np.arange(test_start, test_end),
            train_end=index[train_hi - 1],
            test_start=index[test_start],
            test_end=index[test_end - 1],
        )


def _sharpe_estimation_error(sharpe_per_period: float, n_obs: int,
                             skew: float, kurtosis: float) -> float:
    """Error estandar del estimador del Sharpe, corregido por los momentos.

    Con retornos asimetricos y leptocurticos -- lo normal en cripto -- el
    Sharpe muestral es MENOS preciso de lo que sugiere la formula gaussiana, y
    esta correccion lo refleja.
    """
    var = (1 - skew * sharpe_per_period
           + (kurtosis - 1) / 4 * sharpe_per_period**2) / (n_obs - 1)
    return float(np.sqrt(var)) if var > 0 else np.nan


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int,
                          skew: float = 0.0, kurtosis: float = 3.0,
                          periods_per_year: float = 1.0) -> float:
    """Sharpe deflactado (Bailey & Lopez de Prado, 2014).

    Si se prueban 50 estrategias y se elige la mejor, su Sharpe esta sesgado al
    alza por pura seleccion. Devuelve la probabilidad de que el Sharpe
    verdadero sea > 0 una vez descontado ese sesgo. Por debajo de 0.95 el
    resultado no es distinguible de la suerte.

    IMPORTANTE sobre las escalas: la formula solo es valida con el Sharpe
    expresado POR PERIODO. Pasar el anualizado y compararlo contra un umbral
    por periodo mezcla dos escalas que difieren en un factor sqrt(365) y
    devuelve 0 para cualquier estrategia razonable. Por eso esta funcion pide
    `periods_per_year` de forma explicita y hace la conversion internamente.

    Args:
        sharpe: Sharpe anualizado con `periods_per_year` (o por periodo si
            `periods_per_year` es 1.0).
        n_obs: numero de observaciones.
        n_trials: numero de configuraciones probadas. Es una cota inferior
            honesta: cada decision de diseno tanteada cuenta como una prueba.
        skew, kurtosis: asimetria y curtosis TOTAL (no exceso) de los retornos.
        periods_per_year: factor con el que se anualizo `sharpe`.
    """
    from scipy import stats

    if n_obs < 10 or not np.isfinite(sharpe):
        return np.nan
    sr = sharpe / np.sqrt(periods_per_year)

    euler = 0.5772156649
    n = max(int(n_trials), 2)
    # Valor esperado del maximo de n Sharpes bajo la hipotesis de que ninguna
    # estrategia tiene habilidad. Es un valor ESTANDARIZADO: hay que escalarlo
    # por la dispersion del estimador para llevarlo a unidades de Sharpe.
    z_max = ((1 - euler) * stats.norm.ppf(1 - 1 / n)
             + euler * stats.norm.ppf(1 - 1 / (n * np.e)))
    sr_star = z_max / np.sqrt(n_obs - 1)

    sr_std = _sharpe_estimation_error(sr, n_obs, skew, kurtosis)
    if not np.isfinite(sr_std) or sr_std <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - sr_star) / sr_std))


def probabilistic_sharpe_ratio(sharpe: float, n_obs: int, benchmark: float = 0.0,
                               skew: float = 0.0, kurtosis: float = 3.0,
                               periods_per_year: float = 1.0) -> float:
    """Probabilidad de que el Sharpe real supere `benchmark`.

    Corrige por asimetria y curtosis. Misma advertencia de escalas que
    `deflated_sharpe_ratio`: `sharpe` y `benchmark` deben venir anualizados con
    el mismo `periods_per_year`.
    """
    from scipy import stats

    if n_obs < 10 or not np.isfinite(sharpe):
        return np.nan
    sr = sharpe / np.sqrt(periods_per_year)
    bench = benchmark / np.sqrt(periods_per_year)

    denom = _sharpe_estimation_error(sr, n_obs, skew, kurtosis)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - bench) / denom))
