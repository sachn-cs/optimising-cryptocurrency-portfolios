from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DataValidationConfig, clean_price_data, log_returns, market_proxy
from .execution import ExecutionCostConfig, apply_execution_costs, compute_total_cost_rate
from .forecast import forecast_matrix
from .governance import ForecastGovernance
from .metrics import summarize_strategy
from .networking import (
    build_weighted_graph_from_distance,
    consensus_similarity_matrix,
    correlation_distance_matrix,
    louvain_partition,
    stable_clusters_from_similarity,
)
from .observability import MetricsRegistry, StructuredLogger, Timer
from .portfolio import (
    compute_ledoit_wolf_constant_variance_covariance,
    compute_portfolio_simple_return,
    optimize_maximum_sharpe_ratio,
)
from .risk import RiskLimits, apply_weight_cap, validate_trade_risk
from .types import PortfolioResult, RunArtifacts, StrategySpec


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for portfolio construction and evaluation pipeline."""

    train_window_days: int = 180
    correlation_window_days: int = 60
    rebalance_step_days: int = 30
    horizons_days: list[int] = field(default_factory=lambda: [1, 3, 7, 14])
    consensus_runs: int = 20
    majority_threshold: float = 0.5
    risk_free_rate_annual: float = 0.045
    forecast_method: str = "arima"
    random_seed: int = 42
    weight_cap: float = 0.35
    max_assets: int = 25
    min_assets: int = 2
    max_volatility_annual: float = 1.2
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0


def build_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec("baseline", use_prediction=False, use_shifts=False),
        StrategySpec("s", use_prediction=False, use_shifts=True),
        StrategySpec("p", use_prediction=True, use_shifts=False),
        StrategySpec("p-s", use_prediction=True, use_shifts=True),
    ]


def compute_daily_risk_free_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 365.0) - 1.0


def build_consensus_partitions(
    train_segment: pd.DataFrame,
    strategy: StrategySpec,
    config: PipelineConfig,
    random_generator: np.random.Generator,
) -> tuple[list[list[set[str]]], np.ndarray]:
    assets = list(train_segment.columns)
    partitions: list[list[set[str]]] = []
    forecast_steps = config.correlation_window_days if strategy.use_prediction else 0
    prediction = (
        forecast_matrix(train_segment, forecast_steps, config.forecast_method)
        if forecast_steps > 0
        else None
    )

    for run_index in range(config.consensus_runs):
        shift = run_index if strategy.use_shifts else 0
        end_index = len(train_segment) - shift
        start_index = end_index - config.correlation_window_days
        if start_index < 0:
            continue
        window = train_segment.iloc[start_index:end_index].copy()
        if strategy.use_prediction and prediction is not None:
            window = pd.concat([window, prediction], axis=0, ignore_index=True)
        distance = correlation_distance_matrix(window)
        graph = build_weighted_graph_from_distance(distance)
        partition = louvain_partition(graph, seed=int(random_generator.integers(0, 1_000_000)))
        partitions.append(partition)

    similarity = consensus_similarity_matrix(partitions, assets)
    return partitions, similarity


def run_pipeline(
    prices: pd.DataFrame,
    config: PipelineConfig,
    logger: StructuredLogger | None = None,
    metrics_registry: MetricsRegistry | None = None,
) -> RunArtifacts:
    pipeline_timer = Timer()
    cleaned_prices = clean_price_data(prices, DataValidationConfig(min_assets=config.min_assets))
    returns = log_returns(cleaned_prices)
    market_returns = market_proxy(returns)
    random_generator = np.random.default_rng(config.random_seed)
    governance = ForecastGovernance()

    risk_limits = RiskLimits(
        max_assets=config.max_assets,
        min_assets=config.min_assets,
        max_weight_per_asset=config.weight_cap,
        max_volatility_annual=config.max_volatility_annual,
    )
    cost_config = ExecutionCostConfig(
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
    )

    all_trades: list[PortfolioResult] = []
    all_summaries = []
    similarity_matrices: dict[str, np.ndarray] = {}
    daily_risk_free_rate = compute_daily_risk_free_rate(config.risk_free_rate_annual)
    strategy_specs = build_strategy_specs()

    if logger is not None:
        logger.log_event("pipeline_started", {"rows": len(returns), "assets": returns.shape[1]})

    for horizon_days in config.horizons_days:
        returns_by_strategy: dict[str, list[float]] = {spec.name: [] for spec in strategy_specs}
        market_by_strategy: dict[str, list[float]] = {spec.name: [] for spec in strategy_specs}

        rebalance_index = config.train_window_days
        while rebalance_index + horizon_days <= len(returns):
            train_returns = returns.iloc[rebalance_index - config.train_window_days: rebalance_index]
            future_returns = returns.iloc[rebalance_index: rebalance_index + horizon_days]

            for strategy in strategy_specs:
                partitions, similarity = build_consensus_partitions(train_returns, strategy, config, random_generator)
                del partitions
                similarity_key = f"{strategy.name}_h{horizon_days}_t{rebalance_index}"
                similarity_matrices[similarity_key] = similarity

                stable_clusters = stable_clusters_from_similarity(similarity, list(train_returns.columns), config.majority_threshold)
                selected_assets = [cluster[int(random_generator.integers(0, len(cluster)))] for cluster in stable_clusters if cluster]
                if not selected_assets:
                    continue

                if len(selected_assets) > config.max_assets:
                    selected_assets = selected_assets[: config.max_assets]
                if len(selected_assets) < config.min_assets:
                    continue

                selected_train_returns = train_returns[selected_assets]
                selected_future_returns = future_returns[selected_assets]
                expected_returns = selected_train_returns.mean(axis=0)
                covariance = compute_ledoit_wolf_constant_variance_covariance(selected_train_returns)
                weights = optimize_maximum_sharpe_ratio(expected_returns, covariance, daily_risk_free_rate)
                weights = apply_weight_cap(weights, config.weight_cap)
                validate_trade_risk(selected_assets, weights, covariance, risk_limits)

                mse_value = float(((selected_train_returns - expected_returns) ** 2).mean().mean())
                governance.record_error(mse_value)

                gross_trade_return = compute_portfolio_simple_return(selected_future_returns, weights)
                turnover = float(np.abs(weights).sum())
                cost_rate = compute_total_cost_rate(cost_config, turnover)
                net_trade_return = apply_execution_costs(gross_trade_return, cost_rate)

                market_trade_return = float(((1.0 + future_returns.mean(axis=1)).prod()) - 1.0)
                returns_by_strategy[strategy.name].append(net_trade_return)
                market_by_strategy[strategy.name].append(market_trade_return)

                all_trades.append(
                    PortfolioResult(
                        strategy=strategy.name,
                        horizon_days=horizon_days,
                        rebalance_date=returns.index[rebalance_index],
                        exit_date=returns.index[rebalance_index + horizon_days - 1],
                        selected_assets=selected_assets,
                        weights=weights.to_dict(),
                        turnover=turnover,
                        gross_return=gross_trade_return,
                        net_return=net_trade_return,
                    )
                )
                if metrics_registry is not None:
                    metrics_registry.increment("trades_executed")
            rebalance_index += config.rebalance_step_days

        for strategy in strategy_specs:
            all_summaries.append(
                summarize_strategy(
                    strategy=strategy.name,
                    horizon=horizon_days,
                    trade_returns=returns_by_strategy[strategy.name],
                    market_returns=market_by_strategy[strategy.name],
                )
            )

    if governance.is_drift_detected() and logger is not None:
        logger.log_event("forecast_drift_detected", {"history_points": len(governance.mse_history)})

    if metrics_registry is not None:
        metrics_registry.record_timing_millis("pipeline_duration_millis", pipeline_timer.elapsed_millis())

    if logger is not None:
        logger.log_event("pipeline_completed", {"trades": len(all_trades), "summaries": len(all_summaries)})

    return RunArtifacts(
        returns=returns,
        market_returns=market_returns,
        trades=all_trades,
        summary=all_summaries,
        similarity_matrices=similarity_matrices,
    )
