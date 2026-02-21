from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.utils.artifacts import artifact_dir
from src.utils.io import read_jsonl, write_json


def _write_csv(path: Path, headers: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def run_eval(cfg: dict) -> Path:
    run_id = cfg.get("run", {}).get("run_id", "eval_run")
    out_dir = artifact_dir("eval", run_id)

    asr_dir = Path(cfg.get("inputs", {}).get("asr_dir", ""))
    pii_dir = Path(cfg.get("inputs", {}).get("pii_dir", ""))

    segments = read_jsonl(asr_dir / "segments.jsonl") if asr_dir else []
    spans_rows = read_jsonl(pii_dir / "spans.jsonl") if pii_dir else []

    pii_counter: Counter[str] = Counter()
    lang_counter: Counter[str] = Counter()

    for seg in segments:
        lang_counter[seg.get("language", "unknown")] += 1

    for row in spans_rows:
        for span in row.get("spans", []):
            pii_counter[span.get("type", "UNKNOWN")] += 1

    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_segments": len(segments),
        "num_records_with_spans": len(spans_rows),
        "total_detected_spans": int(sum(pii_counter.values())),
        "types_detected": dict(pii_counter),
        "languages_seen": dict(lang_counter),
    }

    write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "per_type.csv", ["pii_type", "count"], [[k, v] for k, v in sorted(pii_counter.items())])
    _write_csv(out_dir / "per_lang.csv", ["language", "count"], [[k, v] for k, v in sorted(lang_counter.items())])
    write_json(
        out_dir / "run_meta.json",
        {
            "stage": "eval",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "asr_dir": str(asr_dir),
                "pii_dir": str(pii_dir),
                "masked_dir": cfg.get("inputs", {}).get("masked_dir"),
                "gold_dir": cfg.get("inputs", {}).get("gold_dir"),
            },
        },
    )

    return out_dir
