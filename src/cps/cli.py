from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_price_data
from .metrics import summaries_to_frame
from .observability import MetricsRegistry, StructuredLogger
from .pipeline import PipelineConfig, run_pipeline
from .resilience import RetryConfig, execute_with_retry
from .runner import build_run_id, ensure_idempotent_run, mark_run_complete


def generate_synthetic_prices(days: int = 500, assets: int = 12, seed: int = 7) -> pd.DataFrame:
    random_generator = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    factors = random_generator.normal(0.0005, 0.02, size=(days, 3))
    exposures = random_generator.normal(0, 1, size=(assets, 3))
    idiosyncratic = random_generator.normal(0, 0.015, size=(days, assets))
    returns = factors @ exposures.T + idiosyncratic
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    columns = [f"asset_{asset_index:02d}" for asset_index in range(assets)]
    return pd.DataFrame(prices, index=dates, columns=columns)


def parse_horizons(horizons_text: str) -> list[int]:
    horizons = [int(value.strip()) for value in horizons_text.split(",") if value.strip()]
    if not horizons:
        raise ValueError("At least one horizon value is required")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizon values must be positive integers")
    return horizons


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consensus-clustered crypto portfolio system")
    parser.add_argument("--prices-csv", type=str, default="", help="CSV with date column and asset price columns")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--train-window-days", type=int, default=180)
    parser.add_argument("--corr-window-days", type=int, default=60)
    parser.add_argument("--rebalance-step-days", type=int, default=30)
    parser.add_argument("--horizons", type=str, default="1,3,7,14")
    parser.add_argument("--consensus-runs", type=int, default=20)
    parser.add_argument("--majority-threshold", type=float, default=0.5)
    parser.add_argument("--rf-annual", type=float, default=0.045)
    parser.add_argument("--forecast-method", choices=["arima", "naive"], default="arima")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-cap", type=float, default=0.35)
    parser.add_argument("--max-assets", type=int, default=25)
    parser.add_argument("--min-assets", type=int, default=2)
    parser.add_argument("--max-volatility-annual", type=float, default=1.2)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_directory = Path(arguments.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    logger = StructuredLogger("crypto_portfolio", str(output_directory / "events.jsonl"))
    metrics_registry = MetricsRegistry()

    config = PipelineConfig(
        train_window_days=arguments.train_window_days,
        correlation_window_days=arguments.corr_window_days,
        rebalance_step_days=arguments.rebalance_step_days,
        horizons_days=parse_horizons(arguments.horizons),
        consensus_runs=arguments.consensus_runs,
        majority_threshold=arguments.majority_threshold,
        risk_free_rate_annual=arguments.rf_annual,
        forecast_method=arguments.forecast_method,
        random_seed=arguments.seed,
        weight_cap=arguments.weight_cap,
        max_assets=arguments.max_assets,
        min_assets=arguments.min_assets,
        max_volatility_annual=arguments.max_volatility_annual,
        transaction_cost_bps=arguments.transaction_cost_bps,
        slippage_bps=arguments.slippage_bps,
    )

    run_id = build_run_id(config)
    marker = ensure_idempotent_run(arguments.run_dir, run_id)

    retry_config = RetryConfig(max_attempts=3, initial_backoff_seconds=0.05)
    if arguments.prices_csv:
        prices = execute_with_retry(
            lambda: load_price_data(arguments.prices_csv, date_col=arguments.date_col), retry_config
        )
    else:
        prices = generate_synthetic_prices()

    artifacts = execute_with_retry(lambda: run_pipeline(prices, config, logger, metrics_registry), retry_config)

    trades_frame = pd.DataFrame(
        [
            {
                "strategy": trade.strategy,
                "horizon_days": trade.horizon_days,
                "rebalance_date": trade.rebalance_date,
                "exit_date": trade.exit_date,
                "selected_assets": ",".join(trade.selected_assets),
                "weights": trade.weights,
                "turnover": trade.turnover,
                "gross_return": trade.gross_return,
                "net_return": trade.net_return,
            }
            for trade in artifacts.trades
        ]
    )
    summary_frame = summaries_to_frame(artifacts.summary)

    trades_path = output_directory / "trades.csv"
    summary_path = output_directory / "summary.csv"
    returns_path = output_directory / "log_returns.csv"
    metrics_path = output_directory / "metrics.json"

    trades_frame.to_csv(trades_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    artifacts.returns.to_csv(returns_path, index=True)
    metrics_path.write_text(
        pd.Series({"counters": metrics_registry.counters, "timings_millis": metrics_registry.timings_millis}).to_json(),
        encoding="utf-8",
    )

    mark_run_complete(marker)
    print(f"Run id: {run_id}")
    print(f"Wrote {len(trades_frame)} trades to {trades_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
