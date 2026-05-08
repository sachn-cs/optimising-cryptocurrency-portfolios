# Production Readiness

This implementation now includes production controls across reliability, risk, observability, and release quality.

## Reliability
- Retry with bounded exponential backoff for critical operations via `resilience.execute_with_retry`.
- Idempotent run markers via `runner.ensure_idempotent_run` and `runner.mark_run_complete`.

## Risk and Execution
- Portfolio constraints in `risk.RiskLimits`:
  - minimum and maximum asset count
  - effective per-asset cap enforcement
  - annualized volatility ceiling
- Cost model in `execution`:
  - transaction cost (bps)
  - slippage (bps)
  - net return after costs

## Governance
- Forecast MSE tracking and drift detection in `governance.ForecastGovernance`.

## Observability
- Structured JSON event logging in `observability.StructuredLogger`.
- In-process counters and latency metrics via `observability.MetricsRegistry`.
- CLI emits:
  - `events.jsonl`
  - `metrics.json`

## CI Quality Gate
- GitHub Actions workflow runs tests on Python 3.10, 3.11, and 3.12.
- Coverage gate enforced at 90% minimum.
