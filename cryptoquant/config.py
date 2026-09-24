"""Configuracion central del sistema.

Toda la parametrizacion vive aqui o en `config.yaml`. Nada de numeros magicos
dispersos por el codigo: si un parametro afecta al riesgo, tiene que ser
visible y auditable desde un solo sitio.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = ROOT / "reports"
# Si existe, esta carpeta es la copia de trabajo (USB) y no la que recolecta:
# puede leer el diario, pero nunca escribir en el ni instalar la tarea.
WORKING_COPY_MARKER = ROOT / "COPIA_DE_TRABAJO.txt"

for _d in (DATA_DIR, CACHE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    exchange: str = "binance"
    quote: str = "USDT"
    symbols: list[str] = field(
        default_factory=lambda: ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "LINK", "AVAX"]
    )
    timeframe: str = "1d"
    start: str = "2019-01-01"
    max_bars: int = 3000
    cache: bool = True
    # Si no hay red disponible se generan series sinteticas reproducibles.
    allow_synthetic_fallback: bool = True


@dataclass
class RiskConfig:
    # Volatilidad anualizada objetivo de la cartera. Es la palanca principal
    # para "invertir sin asumir el riesgo pleno" del mercado cripto.
    target_volatility: float = 0.15
    # Exposicion bruta maxima. <1.0 significa que parte del capital queda en
    # stablecoin/efectivo de forma permanente.
    max_gross_exposure: float = 1.0
    max_weight_per_asset: float = 0.30
    # Nivel de Kelly bruto a partir del cual se despliega el presupuesto de
    # riesgo completo. NO es un multiplicador de la exposicion: el tamano lo
    # fija `target_volatility`. Subirlo es mas exigente, bajarlo mas agresivo.
    kelly_full_size_level: float = 0.25
    # Cortacircuitos: si el drawdown supera este umbral, se desapalanca.
    max_drawdown_stop: float = 0.25
    drawdown_derisk_start: float = 0.10
    # Nivel de confianza para VaR/CVaR.
    var_confidence: float = 0.95
    # Ventana para la estimacion de covarianzas.
    covariance_lookback: int = 180
    vol_lookback: int = 60


@dataclass
class SignalConfig:
    # Filtro de regimen: solo se toma exposicion larga si el precio esta por
    # encima de esta media movil. Evita comprar en mercados bajistas.
    regime_ma: int = 100
    momentum_windows: list[int] = field(default_factory=lambda: [21, 63, 126])
    # Umbral de probabilidad del modelo ML para abrir posicion.
    min_probability: float = 0.55
    # Etiquetado triple-barrera.
    label_horizon: int = 10
    label_profit_mult: float = 2.0
    label_stop_mult: float = 1.5
    # Peso relativo entre la senal del modelo ML y la senal econometrica.
    ml_weight: float = 0.5


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    # Comision por lado (0.001 = 10 pb, tipico en exchange spot retail).
    fee: float = 0.001
    # Slippage estimado por lado.
    slippage: float = 0.0005
    # Rebalanceo: cada cuantas barras se recalculan los pesos.
    rebalance_every: int = 5
    # No rebalancear si el cambio de peso es menor que esto (ahorra costes).
    min_trade_threshold: float = 0.02
    # Barras iniciales reservadas para calentar indicadores/modelos.
    warmup: int = 250
    # Reentrenamiento walk-forward cada N barras.
    retrain_every: int = 63
    train_window: int = 500


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    seed: int = 42

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Carga config.yaml si existe; en caso contrario usa los defaults."""
        path = Path(path) if path else ROOT / "config.yaml"
        cfg = cls()
        if not path.exists():
            return cfg
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for section in ("data", "risk", "signal", "backtest"):
            if section in raw and isinstance(raw[section], dict):
                current = getattr(cfg, section)
                valid = {f.name for f in dataclasses.fields(current)}
                unknown = set(raw[section]) - valid
                if unknown:
                    raise ValueError(
                        f"config.yaml: claves desconocidas en '{section}': {sorted(unknown)}"
                    )
                setattr(cfg, section, type(current)(**{**dataclasses.asdict(current), **raw[section]}))
        if "seed" in raw:
            cfg.seed = int(raw["seed"])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


PERIODS_PER_YEAR = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24, "1w": 52}


def annualization_factor(timeframe: str) -> float:
    """Cripto cotiza 24/7, por eso 365 y no 252 como en renta variable."""
    if timeframe not in PERIODS_PER_YEAR:
        raise ValueError(f"timeframe no soportado: {timeframe}")
    return float(PERIODS_PER_YEAR[timeframe])
