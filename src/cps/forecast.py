from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def naive_forecast(train_returns: pd.Series, steps: int) -> pd.Series:
    if train_returns.empty:
        raise ValueError("Train return series is empty")
    return pd.Series([train_returns.iloc[-1]] * steps, index=range(steps), dtype=float)


def arima_forecast(train_returns: pd.Series, steps: int, order: tuple[int, int, int] = (1, 0, 1)) -> pd.Series:
    if train_returns.nunique() < 2:
        return naive_forecast(train_returns, steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(train_returns, order=order)
            fit = model.fit()
            pred = fit.forecast(steps=steps)
            return pd.Series(pred, index=range(steps), dtype=float)
        except Exception:
            return naive_forecast(train_returns, steps)


def forecast_matrix(train_returns: pd.DataFrame, steps: int, method: str) -> pd.DataFrame:
    cols = {}
    for col in train_returns.columns:
        s = train_returns[col].astype(float)
        if method == "naive":
            cols[col] = naive_forecast(s, steps)
        elif method == "arima":
            cols[col] = arima_forecast(s, steps)
        else:
            raise ValueError(f"Unknown forecast method: {method}")
    return pd.DataFrame(cols)
