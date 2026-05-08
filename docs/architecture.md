# Architecture

## Design Goals
- Keep modules cohesive and single-purpose.
- Preserve loose coupling via explicit typed interfaces.
- Allow strategy, forecasting, and optimization components to evolve independently.

## Module Boundaries
- `data.py`: ingestion, validation, cleaning, and return-series transformation.
- `forecast.py`: return forecasting (naive and ARIMA) with failure-safe behavior.
- `networking.py`: correlation-distance graph construction and consensus clustering.
- `portfolio.py`: covariance regularization, optimization, and trade return computation.
- `metrics.py`: performance and downside-risk metrics plus tabular summaries.
- `pipeline.py`: orchestration and integration across modules.
- `cli.py`: runtime configuration, entrypoint execution, and artifact export.

## Scalability Considerations
- Forecasting is isolated, enabling model swaps without pipeline rewrites.
- Strategy specification is centralized in `build_strategy_specs()` for extension.
- All data contracts use pandas structures and dataclasses for predictable composition.
