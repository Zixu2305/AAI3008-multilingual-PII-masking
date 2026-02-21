from __future__ import annotations

from pathlib import Path

from src.utils.io import PROJECT_ROOT, ensure_dir


VALID_STAGES = {"asr", "pii", "masked", "eval", "asr_eval", "pii_eval"}


def artifact_dir(stage: str, run_id: str) -> Path:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of: {sorted(VALID_STAGES)}")
    return ensure_dir(PROJECT_ROOT / "data" / "runs" / stage / run_id)
