from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cps.execution import (
    ExecutionCostConfig,
    apply_execution_costs,
    compute_total_cost_rate,
)
from cps.governance import ForecastGovernance
from cps.pipeline import PipelineConfig
from cps.resilience import RetryConfig, execute_with_retry
from cps.risk import RiskLimits, apply_weight_cap, validate_trade_risk
from cps.runner import (
    build_run_id,
    ensure_idempotent_run,
    mark_run_complete,
)


def test_execution_costs_reduce_return():
    cost = compute_total_cost_rate(ExecutionCostConfig(10.0, 5.0), turnover=1.0)
    net = apply_execution_costs(0.10, cost)
    assert cost > 0
    assert net < 0.10


def test_governance_drift_detection():
    governance = ForecastGovernance(drift_threshold_multiplier=1.5)
    for value in [0.1] * 9 + [0.3]:
        governance.record_error(value)
    assert governance.is_drift_detected()


def test_retry_executes_until_success():
    state = {"count": 0}

    def flaky() -> int:
        state["count"] += 1
        if state["count"] < 2:
            raise ValueError("fail")
        return 7

    value = execute_with_retry(flaky, RetryConfig(max_attempts=3, initial_backoff_seconds=0.0))
    assert value == 7


def test_risk_limits_and_weight_cap():
    weights = pd.Series([0.9, 0.1], index=["a", "b"])
    capped = apply_weight_cap(weights, 0.6)
    covariance = pd.DataFrame([[0.01, 0.0], [0.0, 0.01]], index=["a", "b"], columns=["a", "b"])
    limits = RiskLimits(max_assets=3, min_assets=2, max_weight_per_asset=0.7, max_volatility_annual=2.0)
    validate_trade_risk(["a", "b"], capped, covariance, limits)
    assert abs(float(capped.sum()) - 1.0) < 1e-8


def test_runner_idempotency(tmp_path: Path):
    config = PipelineConfig()
    run_id = build_run_id(config)
    marker = ensure_idempotent_run(str(tmp_path), run_id)
    mark_run_complete(marker)
    with pytest.raises(ValueError):
        ensure_idempotent_run(str(tmp_path), run_id)


def test_cli_metrics_file_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import sys

    from cps import cli

    out_dir = tmp_path / "out"
    run_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto-portfolio",
            "--output-dir",
            str(out_dir),
            "--run-dir",
            str(run_dir),
            "--forecast-method",
            "naive",
            "--horizons",
            "1",
            "--consensus-runs",
            "2",
        ],
    )
    cli.main()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "counters" in metrics
