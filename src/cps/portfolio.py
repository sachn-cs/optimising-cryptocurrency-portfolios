from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ledoit_wolf_constant_variance_covariance(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Computes a Ledoit-Wolf constant-variance shrinkage covariance matrix."""
    matrix = returns.to_numpy(dtype=float)
    observations_count, assets_count = matrix.shape
    if assets_count == 1:
        variance = float(np.var(matrix[:, 0], ddof=1)) if observations_count > 1 else 1e-8
        return pd.DataFrame(
            [[max(variance, 1e-8)]],
            index=returns.columns,
            columns=returns.columns,
        )

    sample_covariance = np.cov(matrix, rowvar=False, ddof=1)
    average_variance = np.trace(sample_covariance) / assets_count
    target_covariance = np.eye(assets_count) * average_variance

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    squared = centered**2
    phi_matrix = (
        (squared.T @ squared) / observations_count
        - 2 * (centered.T @ centered) * sample_covariance / observations_count
        + sample_covariance**2
    )
    phi = np.sum(phi_matrix)

    gamma = np.linalg.norm(sample_covariance - target_covariance, ord="fro") ** 2
    kappa = phi / gamma if gamma > 0 else 0.0
    shrinkage = max(0.0, min(1.0, kappa / observations_count))
    shrunk_covariance = shrinkage * target_covariance + (1 - shrinkage) * sample_covariance
    return pd.DataFrame(
        shrunk_covariance,
        index=returns.columns,
        columns=returns.columns,
    )


def project_weights_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Projects unconstrained weights onto the long-only unit simplex."""
    if np.isclose(weights.sum(), 1.0) and np.all(weights >= 0):
        return weights
    sorted_weights = np.sort(weights)[::-1]
    cumulative_sum = np.cumsum(sorted_weights)
    rho = np.nonzero(sorted_weights * np.arange(1, len(weights) + 1) > (cumulative_sum - 1))[0][-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1.0)
    projected = np.maximum(weights - theta, 0)
    return np.asarray(projected, dtype=float)


def optimize_maximum_sharpe_ratio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    daily_risk_free_rate: float,
    max_iterations: int = 2000,
    learning_step: float = 0.05,
) -> pd.Series:
    """Optimizes long-only weights that maximize Sharpe ratio."""
    mean_returns = expected_returns.to_numpy(dtype=float)
    covariance_matrix = covariance.to_numpy(dtype=float)
    assets_count = len(mean_returns)
    if assets_count == 1:
        return pd.Series([1.0], index=expected_returns.index)

    weights = np.ones(assets_count, dtype=float) / assets_count
    for iteration_index in range(max_iterations):
        del iteration_index
        portfolio_return = float(weights @ mean_returns)
        portfolio_variance = float(weights @ covariance_matrix @ weights)
        portfolio_std = np.sqrt(max(portfolio_variance, 1e-12))
        gradient = (
            mean_returns * portfolio_std
            - (portfolio_return - daily_risk_free_rate) * (covariance_matrix @ weights) / portfolio_std
        ) / max(portfolio_variance, 1e-12)
        weights = project_weights_to_simplex(weights + learning_step * gradient)
    return pd.Series(weights, index=expected_returns.index)


def compute_portfolio_simple_return(
    future_returns: pd.DataFrame,
    weights: pd.Series,
) -> float:
    """Computes simple return over a holding period from compounded asset returns."""
    aligned_returns = future_returns[weights.index]
    compounded_returns = (1.0 + aligned_returns).prod(axis=0) - 1.0
    return float(
        np.dot(
            compounded_returns.to_numpy(dtype=float),
            weights.to_numpy(dtype=float),
        )
    )
