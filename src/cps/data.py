from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataValidationConfig:
    max_missing_days: int = 10
    min_assets: int = 4


def load_price_data(csv_path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {csv_path}")
    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.sort_values(date_col).set_index(date_col)
    if df.empty:
        raise ValueError("Price data is empty")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate asset columns detected")
    if not np.issubdtype(df.to_numpy().dtype, np.number):
        raise ValueError("Price matrix contains non-numeric values")
    return df.astype(float)


def clean_price_data(prices: pd.DataFrame, cfg: DataValidationConfig) -> pd.DataFrame:
    if prices.shape[1] < cfg.min_assets:
        raise ValueError("Insufficient number of assets before filtering")
    missing_counts = prices.isna().sum(axis=0)
    keep_cols = missing_counts[missing_counts <= cfg.max_missing_days].index
    filtered = prices[keep_cols].copy()
    if filtered.empty:
        raise ValueError("No assets remaining after missing-value filtering")
    filtered = filtered.interpolate(method="time").ffill().bfill()
    if filtered.isna().any().any():
        raise ValueError("NaN values remain after interpolation")
    if (filtered <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    if filtered.shape[1] < cfg.min_assets:
        raise ValueError("Insufficient number of assets after filtering")
    return filtered


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(prices / prices.shift(1)).dropna(how="all")
    if lr.empty:
        raise ValueError("No log-returns could be computed")
    return lr.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def market_proxy(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1)
