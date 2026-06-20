from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_assets: int = 25
    min_assets: int = 2
    max_weight_per_asset: float = 0.35
    max_volatility_annual: float = 1.2


def compute_effective_weight_cap(configured_cap: float, assets_count: int) -> float:
    if assets_count <= 0:
        raise ValueError("assets_count must be positive")
    if configured_cap <= 0:
        raise ValueError("configured_cap must be positive")
    return min(1.0, max(configured_cap, 1.0 / assets_count))


def apply_weight_cap(weights: pd.Series, cap: float) -> pd.Series:
    effective_cap = compute_effective_weight_cap(cap, len(weights))
    current = weights.clip(lower=0.0)
    if current.sum() <= 0:
        return pd.Series(1.0 / len(weights), index=weights.index)
    current = current / current.sum()

    for _ in range(100):
        over_cap = current > effective_cap
        if not over_cap.any():
            break
        excess = float((current[over_cap] - effective_cap).sum())
        current[over_cap] = effective_cap
        under_cap = current < effective_cap
        room = float((effective_cap - current[under_cap]).sum())
        if room <= 1e-12:
            break
        increment = (effective_cap - current[under_cap]) / room * excess
        current[under_cap] = current[under_cap] + increment
        current = current / current.sum()

    current = current.clip(lower=0.0)
    current = current / current.sum()
    return current


def validate_trade_risk(
    selected_assets: list[str],
    weights: pd.Series,
    covariance: pd.DataFrame,
    limits: RiskLimits,
) -> None:
    if len(selected_assets) < limits.min_assets:
        raise ValueError("Selected assets below minimum risk limit")
    if len(selected_assets) > limits.max_assets:
        raise ValueError("Selected assets above maximum risk limit")
    effective_cap = compute_effective_weight_cap(limits.max_weight_per_asset, len(selected_assets))
    if float(weights.max()) > effective_cap + 1e-8:
        raise ValueError("Per-asset weight exceeds configured cap")
    annual_volatility = float((weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy()) ** 0.5) * (365.0**0.5)
    if annual_volatility > limits.max_volatility_annual:
        raise ValueError("Portfolio annualized volatility exceeds configured maximum")
