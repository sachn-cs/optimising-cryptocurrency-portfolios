# Crypto Portfolio System

Production-hardened implementation of the framework in `arXiv:2505.24831v2` for cryptocurrency portfolio construction.

## Core Capabilities
- Return forecasting (`naive`, `arima`)
- Rolling correlation networks and Louvain clustering
- Consensus stable cluster extraction
- Sharpe-ratio portfolio optimization
- Downside-risk and profitability metrics

## Production Controls
- Retry and bounded backoff for critical operations
- Idempotent run execution with run markers
- Risk limits: asset count, per-asset cap, annualized volatility ceiling
- Execution costs: transaction cost and slippage applied to net returns
- Forecast-governance drift checks
- Structured event logging and metrics emission

See [docs/production-readiness.md](/Users/sachin/Research/math/optimising-cryptocurrency-portfolios/docs/production-readiness.md).

## Project Structure
- `src/cps/` core package
- `tests/` unit and integration tests
- `docs/architecture.md` architecture
- `docs/api.md` API contracts
- `docs/production-readiness.md` operational controls

## Installation
```bash
pip install -e .[dev]
```

## Run
Synthetic data:
```bash
crypto-portfolio --output-dir outputs --run-dir runs
```

CSV input:
```bash
crypto-portfolio --prices-csv /path/to/prices.csv --date-col date --output-dir outputs --run-dir runs
```

## Important CLI Flags
```bash
crypto-portfolio \
  --train-window-days 180 \
  --corr-window-days 60 \
  --rebalance-step-days 30 \
  --horizons 1,3,7,14 \
  --consensus-runs 20 \
  --majority-threshold 0.5 \
  --rf-annual 0.045 \
  --forecast-method arima \
  --weight-cap 0.35 \
  --max-assets 25 \
  --min-assets 2 \
  --max-volatility-annual 1.2 \
  --transaction-cost-bps 10 \
  --slippage-bps 5 \
  --seed 42
```

## Outputs
- `trades.csv` with gross and net returns
- `summary.csv` strategy-level metrics
- `log_returns.csv`
- `events.jsonl` structured runtime events
- `metrics.json` counters and timing metrics

## Test and Coverage
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m coverage run -m pytest -q && python -m coverage report -m
```
