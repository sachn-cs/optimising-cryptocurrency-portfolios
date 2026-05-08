from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCostConfig:
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0


def compute_total_cost_rate(cost_config: ExecutionCostConfig, turnover: float) -> float:
    bps_total = (cost_config.transaction_cost_bps + cost_config.slippage_bps) * max(turnover, 0.0)
    return bps_total / 10000.0


def apply_execution_costs(gross_return: float, cost_rate: float) -> float:
    return (1.0 + gross_return) * (1.0 - cost_rate) - 1.0
