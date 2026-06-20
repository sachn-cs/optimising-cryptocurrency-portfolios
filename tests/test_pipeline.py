from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cps.metrics import omega_ratio, profit_factor
from cps.pipeline import PipelineConfig, run_pipeline


def synthetic_prices(days: int = 320, assets: int = 8, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    factors = rng.normal(0.0002, 0.015, size=(days, 2))
    exposures = rng.normal(0, 1, size=(assets, 2))
    idio = rng.normal(0, 0.012, size=(days, assets))
    ret = factors @ exposures.T + idio
    px = 50 * np.exp(np.cumsum(ret, axis=0))
    return pd.DataFrame(px, index=dates, columns=[f"c{i}" for i in range(assets)])


def test_pipeline_end_to_end():
    prices = synthetic_prices()
    cfg = PipelineConfig(
        train_window_days=120,
        correlation_window_days=40,
        rebalance_step_days=30,
        horizons_days=(1, 3),
        consensus_runs=8,
        forecast_method="naive",
        random_seed=11,
    )
    artifacts = run_pipeline(prices, cfg)
    assert not artifacts.returns.empty
    assert len(artifacts.trades) > 0
    assert len(artifacts.summary) == 8
    for t in artifacts.trades:
        assert abs(sum(t.weights.values()) - 1.0) < 1e-6
        assert all(w >= -1e-10 for w in t.weights.values())


def test_metrics_edge_cases():
    t = np.array([0.1, -0.05, 0.03, -0.02])
    assert profit_factor(t) > 1.0
    assert omega_ratio(t, 0.0) > 1.0


def test_cli_smoke(tmp_path: Path):
    out = tmp_path / "o"
    cmd = [
        sys.executable,
        "-m",
        "cps.cli",
        "--output-dir",
        str(out),
        "--run-dir",
        str(out / "runs"),
        "--forecast-method",
        "naive",
        "--horizons",
        "1",
        "--consensus-runs",
        "4",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (out / "trades.csv").exists()
    assert (out / "summary.csv").exists()
