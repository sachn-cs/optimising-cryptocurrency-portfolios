from __future__ import annotations

from cps.governance import ForecastGovernance


class TestForecastGovernance:
    def test_record_error_stores_values(self):
        gov = ForecastGovernance()
        gov.record_error(0.1)
        gov.record_error(0.2)
        assert gov.mse_history == [0.1, 0.2]

    def test_drift_not_detected_with_few_samples(self):
        gov = ForecastGovernance(drift_threshold_multiplier=1.0)
        for _ in range(9):
            gov.record_error(0.1)
        assert not gov.is_drift_detected()

    def test_drift_not_detected_when_stable(self):
        gov = ForecastGovernance(drift_threshold_multiplier=2.0)
        for _ in range(15):
            gov.record_error(0.1)
        assert not gov.is_drift_detected()

    def test_drift_detected_when_spike(self):
        gov = ForecastGovernance(drift_threshold_multiplier=1.5)
        for _ in range(9):
            gov.record_error(0.1)
        gov.record_error(0.5)
        assert gov.is_drift_detected()

    def test_drift_not_detected_below_threshold(self):
        gov = ForecastGovernance(drift_threshold_multiplier=3.0)
        for _ in range(9):
            gov.record_error(0.1)
        gov.record_error(0.25)
        assert not gov.is_drift_detected()

    def test_custom_threshold_multiplier(self):
        gov = ForecastGovernance(drift_threshold_multiplier=1.0)
        for _ in range(9):
            gov.record_error(0.1)
        gov.record_error(0.15)
        assert gov.is_drift_detected()

    def test_empty_history(self):
        gov = ForecastGovernance()
        assert not gov.is_drift_detected()
        assert gov.mse_history == []

    def test_single_error(self):
        gov = ForecastGovernance()
        gov.record_error(0.5)
        assert len(gov.mse_history) == 1
        assert not gov.is_drift_detected()

    def test_error_values_are_float(self):
        gov = ForecastGovernance()
        gov.record_error(1)
        gov.record_error(2.0)
        assert all(isinstance(v, float) for v in gov.mse_history)

    def test_baseline_excludes_latest(self):
        gov = ForecastGovernance(drift_threshold_multiplier=1.0)
        for _ in range(9):
            gov.record_error(0.1)
        gov.record_error(0.1)
        assert not gov.is_drift_detected()
