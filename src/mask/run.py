from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.utils.artifacts import artifact_dir
from src.utils.io import PROJECT_ROOT, read_jsonl, write_json, write_jsonl


def _resolve_audio_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def run_mask(cfg: dict) -> Path:
    run_id = cfg.get("run", {}).get("run_id", "masked_run")
    out_dir = artifact_dir("masked", run_id)
    masked_audio_dir = out_dir / "masked_audio"
    masked_audio_dir.mkdir(parents=True, exist_ok=True)

    pii_input = Path(cfg.get("run", {}).get("input_pii_dir", f"data/runs/pii/{run_id}"))
    spans_rows = read_jsonl(pii_input / "spans.jsonl")

    by_source: dict[str, dict] = defaultdict(
        lambda: {
            "sample_segment_ids": [],
            "first_segment_id": None,
            "last_segment_id": None,
            "segments_count": 0,
            "masked_span_count": 0,
        }
    )

    for row in spans_rows:
        source_audio = row.get("audio_path")
        if not source_audio:
            continue

        entry = by_source[source_audio]
        seg_id = row.get("segment_id")
        if seg_id:
            if entry["first_segment_id"] is None:
                entry["first_segment_id"] = seg_id
            entry["last_segment_id"] = seg_id
            if len(entry["sample_segment_ids"]) < 10:
                entry["sample_segment_ids"].append(seg_id)
        entry["segments_count"] += 1
        entry["masked_span_count"] += len(row.get("spans", []))

    audit_rows = []
    files_copied = 0
    files_missing = 0
    for source_audio, stats in sorted(by_source.items()):
        src = _resolve_audio_path(source_audio)
        output_audio = None
        if src.exists() and src.is_file():
            try:
                rel = src.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = Path(src.name)
            dst = masked_audio_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            output_audio = str(dst)
            files_copied += 1
        else:
            files_missing += 1

        audit_rows.append(
            {
                "source_audio": source_audio,
                "masked_audio": output_audio,
                "segments_count": stats["segments_count"],
                "first_segment_id": stats["first_segment_id"],
                "last_segment_id": stats["last_segment_id"],
                "sample_segment_ids": stats["sample_segment_ids"],
                "masked_span_count": stats["masked_span_count"],
                "method": "placeholder_copy_dedup_by_source_audio",
            }
        )

    write_jsonl(out_dir / "audit.jsonl", audit_rows)
    write_json(
        out_dir / "run_meta.json",
        {
            "stage": "masked",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_pii_dir": str(pii_input),
            "items": len(audit_rows),
            "source_rows": len(by_source),
            "input_rows": len(spans_rows),
            "files_copied": files_copied,
            "files_missing": files_missing,
        },
    )

    return out_dir
