from __future__ import annotations

import pytest

from cps.resilience import RetryConfig, execute_with_retry


class TestRetryConfig:
    def test_default_values(self):
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_backoff_seconds == 0.1
        assert config.backoff_multiplier == 2.0

    def test_custom_values(self):
        config = RetryConfig(max_attempts=5, initial_backoff_seconds=0.5, backoff_multiplier=3.0)
        assert config.max_attempts == 5
        assert config.initial_backoff_seconds == 0.5
        assert config.backoff_multiplier == 3.0

    def test_frozen(self):
        config = RetryConfig()
        with pytest.raises(AttributeError):
            config.max_attempts = 10


class TestExecuteWithRetry:
    def test_succeeds_first_attempt(self):
        config = RetryConfig(max_attempts=3, initial_backoff_seconds=0.0)
        result = execute_with_retry(lambda: 42, config)
        assert result == 42

    def test_succeeds_after_retries(self):
        state = {"count": 0}

        def flaky() -> int:
            state["count"] += 1
            if state["count"] < 3:
                raise ValueError("not yet")
            return 99

        config = RetryConfig(max_attempts=5, initial_backoff_seconds=0.0)
        result = execute_with_retry(flaky, config)
        assert result == 99
        assert state["count"] == 3

    def test_exhausts_retries_and_raises(self):
        def always_fail() -> int:
            raise RuntimeError("permanent failure")

        config = RetryConfig(max_attempts=3, initial_backoff_seconds=0.0)
        with pytest.raises(RuntimeError, match="permanent failure"):
            execute_with_retry(always_fail, config)

    def test_max_attempts_must_be_positive(self):
        config = RetryConfig(max_attempts=0)
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            execute_with_retry(lambda: None, config)

    def test_negative_max_attempts(self):
        config = RetryConfig(max_attempts=-1)
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            execute_with_retry(lambda: None, config)

    def test_single_attempt_succeeds(self):
        config = RetryConfig(max_attempts=1, initial_backoff_seconds=0.0)
        result = execute_with_retry(lambda: "ok", config)
        assert result == "ok"

    def test_single_attempt_fails(self):
        def fail() -> str:
            raise ValueError("no retries")

        config = RetryConfig(max_attempts=1, initial_backoff_seconds=0.0)
        with pytest.raises(ValueError, match="no retries"):
            execute_with_retry(fail, config)

    def test_returns_complex_type(self):
        config = RetryConfig(max_attempts=1, initial_backoff_seconds=0.0)
        result = execute_with_retry(lambda: {"key": [1, 2, 3]}, config)
        assert result == {"key": [1, 2, 3]}

    def test_different_exception_types(self):
        call_count = {"n": 0}

        def mixed_errors() -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TypeError("type error")
            if call_count["n"] == 2:
                raise KeyError("key error")
            return "recovered"

        config = RetryConfig(max_attempts=5, initial_backoff_seconds=0.0)
        result = execute_with_retry(mixed_errors, config)
        assert result == "recovered"

    def test_backoff_increases(self):
        config = RetryConfig(max_attempts=3, initial_backoff_seconds=0.01, backoff_multiplier=2.0)
        delays = []
        original_sleep = __import__("time").sleep

        def mock_sleep(d):
            delays.append(d)

        __import__("time").sleep = mock_sleep
        try:
            execute_with_retry(lambda: (_ for _ in ()).throw(ValueError("test")), config)
        except ValueError:
            pass
        finally:
            __import__("time").sleep = original_sleep

        assert len(delays) == 2
        assert delays[1] == pytest.approx(delays[0] * 2.0)
