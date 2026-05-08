# API Reference

## Core Entry Points
- `cps.pipeline.run_pipeline(prices, config)`
- `cps.cli.main()`

## Key Configuration
`PipelineConfig` fields:
- `train_window_days`
- `correlation_window_days`
- `rebalance_step_days`
- `horizons_days`
- `consensus_runs`
- `majority_threshold`
- `risk_free_rate_annual`
- `forecast_method`
- `random_seed`

## Output Artifacts
`RunArtifacts` includes:
- `returns`: cleaned log-returns time series.
- `market_returns`: equal-weight market proxy series.
- `trades`: per-rebalance trade records.
- `summary`: per-strategy and per-horizon aggregated metrics.
- `similarity_matrices`: consensus co-membership matrices by scenario.
