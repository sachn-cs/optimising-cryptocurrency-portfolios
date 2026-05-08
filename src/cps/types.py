from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    name: str
    use_prediction: bool
    use_shifts: bool


@dataclass
class PortfolioResult:
    strategy: str
    horizon_days: int
    rebalance_date: pd.Timestamp
    exit_date: pd.Timestamp
    selected_assets: list[str]
    weights: dict[str, float]
    turnover: float
    gross_return: float
    net_return: float


@dataclass
class EvaluationSummary:
    strategy: str
    horizon_days: int
    average_trade: float
    win_rate: float
    profit_factor: float
    var_95: float
    mes_95: float
    omega_0: float
    trade_count: int


@dataclass
class RunArtifacts:
    returns: pd.DataFrame
    market_returns: pd.Series
    trades: list[PortfolioResult]
    summary: list[EvaluationSummary]
    similarity_matrices: dict[str, np.ndarray]
