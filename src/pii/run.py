from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pii.ner import load_ner_pipeline, predict_ner_spans
from src.utils.artifacts import artifact_dir
from src.utils.io import read_jsonl, write_json, write_jsonl


PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{7,}\d)(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _rule_spans_for_text(text: str, rules_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict] = []
    if bool(rules_cfg.get("phone", True)):
        for m in PHONE_RE.finditer(text):
            spans.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "type": "PHONE",
                    "text": m.group(0),
                    "source": "rule",
                }
            )
    if bool(rules_cfg.get("email", True)):
        for m in EMAIL_RE.finditer(text):
            spans.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "type": "EMAIL",
                    "text": m.group(0),
                    "source": "rule",
                }
            )
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans


def _clean_and_merge_spans(
    text: str,
    spans: list[dict[str, Any]],
    merge_gap_chars: int,
    min_span_chars: int,
) -> list[dict[str, Any]]:
    unique = {}
    for s in spans:
        start = int(s.get("start", -1))
        end = int(s.get("end", -1))
        label = str(s.get("type", "")).strip()
        if start < 0 or end <= start or not label:
            continue
        if (end - start) < min_span_chars:
            continue
        key = (start, end, label)
        if key not in unique:
            out = dict(s)
            out["start"] = start
            out["end"] = end
            out["type"] = label
            out["text"] = text[start:end]
            unique[key] = out

    ordered = sorted(unique.values(), key=lambda s: (int(s["start"]), int(s["end"]), str(s["type"])))
    if not ordered:
        return []

    merged: list[dict[str, Any]] = []
    for span in ordered:
        if not merged:
            merged.append(span)
            continue

        prev = merged[-1]
        prev_start, prev_end = int(prev["start"]), int(prev["end"])
        cur_start, cur_end = int(span["start"]), int(span["end"])
        if prev.get("type") == span.get("type") and cur_start <= (prev_end + merge_gap_chars):
            prev["end"] = max(prev_end, cur_end)
            prev["text"] = text[int(prev["start"]) : int(prev["end"])]
            continue
        merged.append(span)

    return merged


def run_pii(cfg: dict) -> Path:
    run_id = cfg.get("run", {}).get("run_id", "pii_run")
    out_dir = artifact_dir("pii", run_id)

    asr_input = Path(cfg.get("run", {}).get("input_asr_dir", f"data/runs/asr/{run_id}"))
    segments = read_jsonl(asr_input / "segments.jsonl")
    pii_cfg = cfg.get("pii", {})
    enable_rules = bool(pii_cfg.get("enable_rules", True))
    enable_ner = bool(pii_cfg.get("enable_ner", False))
    rules_cfg = pii_cfg.get("rules", {})
    ner_cfg = pii_cfg.get("ner", {})
    post_cfg = cfg.get("postprocess", {})
    merge_gap_chars = int(post_cfg.get("merge_gap_chars", 0) or 0)
    min_span_chars = int(post_cfg.get("min_span_chars", 1) or 1)

    ner_pipe = None
    if enable_ner:
        model_name = str(ner_cfg.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("pii.ner.model_name is required when pii.enable_ner=true")
        ner_pipe = load_ner_pipeline(model_name=model_name, device_cfg=ner_cfg.get("device", "cpu"))

    total_spans_counter: Counter[str] = Counter()

    rows = []
    for seg in segments:
        text = seg.get("text", "")
        spans: list[dict[str, Any]] = []
        if enable_rules:
            spans.extend(_rule_spans_for_text(text, rules_cfg=rules_cfg))
        if ner_pipe is not None:
            spans.extend(
                predict_ner_spans(
                    text=text,
                    ner_pipeline=ner_pipe,
                    label_map=ner_cfg.get("label_map"),
                    max_length=ner_cfg.get("max_length"),
                    score_threshold=ner_cfg.get("score_threshold"),
                )
            )

        spans = _clean_and_merge_spans(
            text=text,
            spans=spans,
            merge_gap_chars=merge_gap_chars,
            min_span_chars=min_span_chars,
        )
        total_spans_counter.update(str(s.get("type", "UNKNOWN")) for s in spans)

        rows.append(
            {
                "segment_id": seg.get("segment_id"),
                "audio_path": seg.get("audio_path"),
                "text": text,
                "spans": spans,
            }
        )

    write_jsonl(out_dir / "spans.jsonl", rows)
    write_json(
        out_dir / "run_meta.json",
        {
            "stage": "pii",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_asr_dir": str(asr_input),
            "items": len(rows),
            "total_detected_spans": int(sum(total_spans_counter.values())),
            "types_detected": dict(total_spans_counter),
            "detectors": {
                "rules": enable_rules,
                "ner": enable_ner,
            },
            "ner_model_name": ner_cfg.get("model_name") if enable_ner else None,
        },
    )

    return out_dir
