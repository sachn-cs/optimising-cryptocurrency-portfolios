from __future__ import annotations

import pandas as pd
import pytest

from cps.data import (
    DataValidationConfig,
    clean_price_data,
    load_price_data,
    log_returns,
    market_proxy,
)


def test_load_price_data_and_date_index(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,a,b\n2024-01-01,10,20\n2024-01-02,11,19\n", encoding="utf-8")
    frame = load_price_data(str(path), date_col="date")
    assert list(frame.columns) == ["a", "b"]
    assert frame.index.name == "date"


def test_clean_price_data_filters_and_interpolates():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "a": [10.0, None, 12.0, 13.0, 14.0],
            "b": [20.0, 20.5, None, 21.0, 22.0],
            "c": [5.0, 5.1, 5.2, 5.3, 5.4],
            "d": [8.0, 8.1, 8.2, 8.3, 8.4],
        },
        index=index,
    )
    cleaned = clean_price_data(prices, DataValidationConfig(max_missing_days=2, min_assets=4))
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned > 0).all().all()


def test_log_returns_and_market_proxy():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"a": [10.0, 11.0, 12.1, 13.31], "b": [20.0, 19.0, 20.9, 21.945]}, index=index)
    returns = log_returns(prices)
    market = market_proxy(returns)
    assert len(returns) == 3
    assert len(market) == 3


def test_clean_price_data_rejects_non_positive_values():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {"a": [10.0, 0.0, 11.0, 12.0], "b": [10.0, 11.0, 12.0, 13.0], "c": [1.0, 1.1, 1.2, 1.3], "d": [2.0, 2.1, 2.2, 2.3]},
        index=index,
    )
    with pytest.raises(ValueError):
        clean_price_data(prices, DataValidationConfig())
