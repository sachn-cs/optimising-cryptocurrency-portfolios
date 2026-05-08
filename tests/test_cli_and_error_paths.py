from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from cps import cli
from cps.data import load_price_data


def test_generate_synthetic_prices_shape():
    frame = cli.generate_synthetic_prices(days=20, assets=5, seed=3)
    assert frame.shape == (20, 5)
    assert (frame > 0).all().all()


def test_main_runs_with_synthetic_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto-portfolio",
            "--output-dir",
            str(output_dir),
            "--run-dir",
            str(output_dir / "runs"),
            "--forecast-method",
            "naive",
            "--horizons",
            "1",
            "--consensus-runs",
            "3",
        ],
    )
    cli.main()
    assert (output_dir / "trades.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "log_returns.csv").exists()


def test_main_runs_with_csv_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,a,b,c,d\n"
        "2024-01-01,10,20,30,40\n"
        "2024-01-02,11,21,29,41\n"
        "2024-01-03,12,20,28,42\n"
        "2024-01-04,13,22,27,43\n"
        "2024-01-05,14,23,26,44\n"
        "2024-01-06,15,24,25,45\n"
        "2024-01-07,16,25,24,46\n"
        "2024-01-08,17,26,23,47\n"
        "2024-01-09,18,27,22,48\n"
        "2024-01-10,19,28,21,49\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out_csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto-portfolio",
            "--prices-csv",
            str(csv_path),
            "--date-col",
            "date",
            "--output-dir",
            str(output_dir),
            "--run-dir",
            str(output_dir / "runs"),
            "--train-window-days",
            "5",
            "--corr-window-days",
            "3",
            "--rebalance-step-days",
            "2",
            "--horizons",
            "1",
            "--forecast-method",
            "naive",
            "--consensus-runs",
            "2",
        ],
    )
    cli.main()
    assert (output_dir / "summary.csv").exists()


def test_load_price_data_missing_date_column(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("x,a\n1,10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_price_data(str(csv_path), date_col="date")


def test_load_price_data_non_numeric_values(tmp_path: Path):
    csv_path = tmp_path / "bad_numeric.csv"
    csv_path.write_text("date,a\n2024-01-01,ten\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_price_data(str(csv_path), date_col="date")


def test_parse_horizons_rejects_non_integer():
    with pytest.raises(ValueError):
        cli.parse_horizons("1,two,3")
