from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ForecastGovernance:
    mse_history: list[float] = field(default_factory=list)
    drift_threshold_multiplier: float = 2.0

    def record_error(self, mse_value: float) -> None:
        self.mse_history.append(float(mse_value))

    def is_drift_detected(self) -> bool:
        if len(self.mse_history) < 10:
            return False
        baseline = np.mean(self.mse_history[:-1])
        latest = self.mse_history[-1]
        return latest > baseline * self.drift_threshold_multiplier
