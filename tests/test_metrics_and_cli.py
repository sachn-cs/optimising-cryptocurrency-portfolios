from __future__ import annotations

import numpy as np
import pytest

from cps.cli import parse_horizons
from cps.metrics import (
    average_trade,
    mes,
    omega_ratio,
    profit_factor,
    summaries_to_frame,
    summarize_strategy,
    var_quantile,
    win_rate,
)


def test_metrics_core_values():
    trades = np.array([0.10, -0.05, 0.03, -0.02])
    market = np.array([0.01, -0.03, 0.02, -0.04])
    assert average_trade(trades) > 0
    assert 0.0 <= win_rate(trades) <= 1.0
    assert profit_factor(trades) > 1.0
    assert var_quantile(trades, 0.95) <= 0.10
    assert mes(trades, market, 0.95) <= 0.10
    assert omega_ratio(trades, 0.0) > 1.0


def test_strategy_summary_and_frame():
    summary = summarize_strategy("baseline", 1, [0.1, -0.1], [0.05, -0.05])
    frame = summaries_to_frame([summary])
    assert frame.shape[0] == 1
    assert frame.loc[0, "strategy"] == "baseline"


def test_parse_horizons_validation():
    assert parse_horizons("1,3,7") == [1, 3, 7]
    with pytest.raises(ValueError):
        parse_horizons("")
    with pytest.raises(ValueError):
        parse_horizons("1,0")


def test_metrics_zero_loss_and_empty_inputs():
    positive_trades = np.array([0.1, 0.2])
    assert profit_factor(positive_trades) == float("inf")
    assert omega_ratio(positive_trades, 0.0) == float("inf")
    assert average_trade(np.array([])) == 0.0
    assert win_rate(np.array([])) == 0.0
