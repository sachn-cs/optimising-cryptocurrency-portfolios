from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .pipeline import PipelineConfig


def build_run_id(config: PipelineConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ensure_idempotent_run(run_directory: str, run_id: str) -> Path:
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{run_id}.done"
    if marker.exists():
        raise ValueError(f"Run {run_id} has already completed in {directory}")
    return marker


def mark_run_complete(marker_path: Path) -> None:
    marker_path.write_text("completed\n", encoding="utf-8")
