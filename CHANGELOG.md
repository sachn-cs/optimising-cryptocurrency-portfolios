# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Type checking with mypy
- Linting and formatting with ruff
- Pre-commit hooks for automated code quality
- Makefile for common development commands
- GitHub Actions release workflow for PyPI publishing
- Expanded test coverage for governance and resilience modules

## [0.1.0] - 2026-05-19

### Added

- Return forecasting with naive and ARIMA methods
- Rolling correlation networks and Louvain clustering
- Consensus stable cluster extraction
- Sharpe-ratio portfolio optimization
- Downside-risk and profitability metrics
- Retry and bounded backoff for critical operations
- Idempotent run execution with run markers
- Risk limits: asset count, per-asset cap, annualized volatility ceiling
- Execution costs: transaction cost and slippage applied to net returns
- Forecast-governance drift checks
- Structured event logging and metrics emission
- CLI with full parameter configuration
- CSV input support for custom price data
- Synthetic data generation for testing
- GitHub Actions CI with Python 3.10, 3.11, 3.12
- Coverage gate at 90% minimum
