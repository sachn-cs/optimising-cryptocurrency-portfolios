# Frequently Asked Questions

## General

### What is this project?

A production-hardened system for constructing cryptocurrency portfolios using consensus clustering, based on the research paper [arXiv:2505.24831v2](https://arxiv.org/abs/2505.24831v2).

### What cryptocurrencies are supported?

Any cryptocurrency with historical price data. The system is asset-agnostic and works with any set of price series.

### Is this financial advice?

No. This is a research and educational tool. Always consult a qualified financial advisor before making investment decisions.

## Data

### What format should my data be in?

CSV files with:
- A date column (configurable name)
- One column per asset with price values

### How much data do I need?

- Minimum: enough for the training window (default 180 days)
- Recommended: at least 1 year of daily data

### Can I use real-time data?

Currently, the system accepts CSV files. Real-time data ingestion is planned for future releases.

## Technical

### Why are there multiple consensus runs?

Multiple runs with different random seeds produce more stable cluster assignments. The majority threshold determines how consistently assets must cluster together.

### What is the difference between naive and ARIMA forecasting?

- **Naive**: Uses the most recent return as the forecast
- **ARIMA**: Uses autoregressive integrated moving average models for more sophisticated predictions

### What do the horizon parameters mean?

Horizons (e.g., 1,3,7,14) represent the number of days forward for return forecasting and portfolio evaluation.

### How are transaction costs applied?

Transaction costs and slippage are deducted from gross returns to produce net returns in the trade records.

## Troubleshooting

### Tests fail with import errors

Ensure you're running from the project root with PYTHONPATH set:

```bash
PYTHONPATH=src pytest -q
```

### Coverage is below 90%

Add tests for uncovered code paths. Check the coverage report for specific lines:

```bash
python -m coverage report -m
```
