from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pii.ner import load_ner_pipeline, predict_ner_spans
from src.utils.artifacts import artifact_dir
from src.utils.io import read_jsonl, write_json, write_jsonl


PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{6,}\d)(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# --- Enhanced regex patterns ---

# Chinese numeral phone: 4+4 separated by comma/pause, or 8+ consecutive
_ZH_DIGIT = "[零一二三四五六七八九〇○]"
ZH_PHONE_RE = re.compile(
    rf"(?:{_ZH_DIGIT}{{4}}[，、,\s]{_ZH_DIGIT}{{4}})"
    rf"|(?:{_ZH_DIGIT}{{8,}})",
)

# English spoken-form phone: 8+ consecutive spoken digit words
_EN_DIGIT_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"
EN_SPOKEN_PHONE_RE = re.compile(
    rf"(?:{_EN_DIGIT_WORD}(?:\s+{_EN_DIGIT_WORD}){{7,}})",
    re.IGNORECASE,
)

# Cue-word-gated partial phone: catches 4+ digit phones preceded by phone-related words
EN_PARTIAL_PHONE_RE = re.compile(
    r"(?:phone|number|call|dial|contact|hp|handphone)"
    r"[\s:]*"
    r"(\d{4,})",
    re.IGNORECASE,
)
ZH_PARTIAL_PHONE_RE = re.compile(
    r"(?:电话|手机|联系|号码|拨打)"
    r"[是为：:]*\s*"
    rf"({_ZH_DIGIT}{{3,}})",
)

# Address patterns
# Chinese: 大牌 + Chinese numerals, XX路/街/道/园/苑 patterns
# Also handles mixed: 大牌XXX，English Road Name
_ROAD_SUFFIX = r"(?:Road|Street|Avenue|Drive|Lane|View\s+Road)"
ZH_ADDRESS_RE = re.compile(
    # Pattern 1: 大牌 + numerals + optional Chinese/English road name
    rf"(?:[\u4e00-\u9fff]{{0,6}})?大牌{_ZH_DIGIT}{{2,}}"
    rf"(?:[，、,\s]*门牌{_ZH_DIGIT}+)?"
    rf"(?:[，、,\s]*[\u4e00-\u9fff]{{2,6}}(?:路|街|道|桥))?"
    rf"(?:[，、,\s]+[A-Za-z][\w\s]*?{_ROAD_SUFFIX})?"
    # Pattern 2: XX路/街/道/园/苑/花园 + optional 大牌
    rf"|[\u4e00-\u9fff]{{2,6}}(?:路|街|道|园|苑|花园|公寓|Residence)"
    rf"(?:[，、,.\s]*大牌{_ZH_DIGIT}{{2,}})?",
)

# English: Block/Blk + number + road-type, OR number + street name + road-type,
# OR street name + road-type + number
_ROAD_TYPE = r"(?:Road|Street|Avenue|Drive|Lane|Way|Crescent|Place|Boulevard|View\s+Road|Grove\s+Road)"
EN_ADDRESS_RE = re.compile(
    rf"(?:Block|Blk)\s+\d+[\s,]*[\w\s]*?{_ROAD_TYPE}"
    rf"|"
    rf"\d{{1,4}}\s+[\w\s]*?{_ROAD_TYPE}(?:\s+\d{{1,4}})?"
    rf"|"
    rf"[\w]+(?:\s+[\w]+){{0,1}}\s+{_ROAD_TYPE}\s+\d{{1,4}}",
    re.IGNORECASE,
)

# Singapore NRIC/FIN: [STFGM] + 7 digits + 1 check letter (e.g. S1234567A)
SG_NRIC_RE = re.compile(r"(?<![A-Za-z])[STFGM]\d{7}[A-Za-z](?![A-Za-z])", re.IGNORECASE)

# Singapore postal code: "S" or "Singapore" prefix + 6 digits (e.g. S609690)
SG_POSTAL_RE = re.compile(r"(?:Singapore\s*|S)\d{6}(?!\d)", re.IGNORECASE)

# Singapore postal code after address: road-type + optional street number + separator + 6 digits
SG_POSTAL_AFTER_ADDR_RE = re.compile(
    rf"{_ROAD_TYPE}"
    r"(?:\s+\d{1,4})?"     # optional street number
    r"[,\s]+"               # separator (comma, space)
    r"(\d{6})(?!\d)",
    re.IGNORECASE,
)


def _rule_spans_for_text(text: str, rules_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict] = []

    def _add(pattern, pii_type: str, group: int = 0) -> None:
        for m in pattern.finditer(text):
            spans.append({
                "start": m.start(group),
                "end": m.end(group),
                "type": pii_type,
                "text": m.group(group),
                "source": "rule",
            })

    # Phone: digit-based
    if bool(rules_cfg.get("phone", True)):
        _add(PHONE_RE, "PHONE")

    # Phone: Chinese numeral
    if bool(rules_cfg.get("zh_phone", True)):
        _add(ZH_PHONE_RE, "PHONE")

    # Phone: English spoken-form
    if bool(rules_cfg.get("en_spoken_phone", True)):
        _add(EN_SPOKEN_PHONE_RE, "PHONE")

    # Phone: cue-word-gated partial (4+ digits after "phone", "contact", etc.)
    if bool(rules_cfg.get("partial_phone", True)):
        _add(EN_PARTIAL_PHONE_RE, "PHONE", group=1)
        _add(ZH_PARTIAL_PHONE_RE, "PHONE", group=1)

    # Email
    if bool(rules_cfg.get("email", True)):
        _add(EMAIL_RE, "EMAIL")

    # Address: Chinese
    if bool(rules_cfg.get("address", True)):
        _add(ZH_ADDRESS_RE, "ADDRESS")
        _add(EN_ADDRESS_RE, "ADDRESS")

    # Singapore NRIC/FIN
    if bool(rules_cfg.get("nric", True)):
        _add(SG_NRIC_RE, "ID")

    # Singapore postal code
    if bool(rules_cfg.get("postal_code", True)):
        _add(SG_POSTAL_RE, "ADDRESS")
        _add(SG_POSTAL_AFTER_ADDR_RE, "ADDRESS", group=1)

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


def _detect_pii_for_text(
    text: str,
    enable_rules: bool,
    rules_cfg: dict,
    ner_pipe: Any | None,
    ner_cfg: dict,
    enable_llm: bool,
    llm_cfg: dict,
    merge_gap_chars: int,
    min_span_chars: int,
) -> list[dict[str, Any]]:
    """Run all enabled detectors on a text and return merged spans."""
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

    if enable_llm:
        try:
            from src.pii.llm_detect import detect_pii_llm
            llm_spans = detect_pii_llm(text, llm_cfg)
            spans.extend(llm_spans)
        except (ImportError, Exception) as e:
            pass  # LLM is optional safety net

    spans = _clean_and_merge_spans(
        text=text,
        spans=spans,
        merge_gap_chars=merge_gap_chars,
        min_span_chars=min_span_chars,
    )
    return spans


def run_pii(cfg: dict) -> Path:
    run_id = cfg.get("run", {}).get("run_id", "pii_run")
    out_dir = artifact_dir("pii", run_id)

    asr_input = Path(cfg.get("run", {}).get("input_asr_dir", f"data/runs/asr/{run_id}"))
    segments = read_jsonl(asr_input / "segments.jsonl")
    pii_cfg = cfg.get("pii", {})
    enable_rules = bool(pii_cfg.get("enable_rules", True))
    enable_ner = bool(pii_cfg.get("enable_ner", False))
    enable_llm = bool(pii_cfg.get("enable_llm", False))
    rules_cfg = pii_cfg.get("rules", {})
    ner_cfg = pii_cfg.get("ner", {})
    llm_cfg = pii_cfg.get("llm", {})
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
        spans = _detect_pii_for_text(
            text=text,
            enable_rules=enable_rules,
            rules_cfg=rules_cfg,
            ner_pipe=ner_pipe,
            ner_cfg=ner_cfg,
            enable_llm=enable_llm,
            llm_cfg=llm_cfg,
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
                "llm": enable_llm,
            },
            "ner_model_name": ner_cfg.get("model_name") if enable_ner else None,
        },
    )

    return out_dir
