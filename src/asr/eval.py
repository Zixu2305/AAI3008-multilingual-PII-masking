from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jiwer import wer

from src.utils.artifacts import artifact_dir
from src.utils.io import read_jsonl, resolve_path, write_json, write_jsonl


TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
EN_CHAR_RE = re.compile(r"[A-Za-z]")
DIGIT_RUN_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
NUM_WORD_TOKEN = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
    r"ninety|hundred|thousand|million|billion|trillion|point|and|minus|negative|oh|o)"
)
NUM_WORD_SEQ_RE = re.compile(rf"\b{NUM_WORD_TOKEN}(?:[-\s]+{NUM_WORD_TOKEN})*\b", flags=re.IGNORECASE)
NUM_WORD_SINGLE_OK = {
    "zero",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
}


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _normalize_number_forms(text: str) -> str:
    x = DIGIT_RUN_RE.sub(" num ", text)

    def _replace_num_word_seq(m: re.Match[str]) -> str:
        s = m.group(0)
        toks = re.findall(r"[A-Za-z]+", s.lower())
        if not toks:
            return s
        if len(toks) == 1 and toks[0] not in NUM_WORD_SINGLE_OK:
            return s
        if all(t == "and" for t in toks):
            return s
        return " num "

    x = NUM_WORD_SEQ_RE.sub(_replace_num_word_seq, x)
    return x


def _clean_text(text: str, normalize_numbers: bool = False) -> str:
    x = TAG_RE.sub(" ", text or "")
    x = x.replace("’", "'").replace("`", "'")
    if normalize_numbers:
        x = _normalize_number_forms(x)
    return " ".join(x.split())


def _tokenize_mixed_words(text: str, normalize_numbers: bool = False) -> list[str]:
    x = _clean_text(text, normalize_numbers=normalize_numbers).lower()
    return TOKEN_RE.findall(x)


def _tokenize_chars(text: str, normalize_numbers: bool = False) -> list[str]:
    x = _clean_text(text, normalize_numbers=normalize_numbers).lower()
    return CHAR_RE.findall(x)


def _bucket_language(text: str) -> str:
    x = _clean_text(text)
    zh_chars = len(ZH_CHAR_RE.findall(x))
    en_chars = len(EN_CHAR_RE.findall(x))
    if zh_chars == 0 and en_chars == 0:
        return "unknown"
    if zh_chars > 0 and en_chars > 0:
        return "mixed"
    if zh_chars > 0:
        return "zh"
    return "en"


def _safe_wer(ref_tokens: list[str], hyp_tokens: list[str]) -> float | None:
    if not ref_tokens:
        return None
    ref = " ".join(ref_tokens)
    hyp = " ".join(hyp_tokens)
    return float(wer(ref, hyp))


def _aggregate_hyp_by_item(segments: list[dict[str, Any]]) -> dict[str, str]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        item_id = seg.get("source_item_id")
        if not item_id:
            continue
        by_item[str(item_id)].append(seg)

    result: dict[str, str] = {}
    for item_id, rows in by_item.items():
        rows_sorted = sorted(rows, key=lambda r: (float(r.get("start", 0.0) or 0.0), str(r.get("segment_id", ""))))
        texts = [(r.get("text") or "").strip() for r in rows_sorted]
        texts = [t for t in texts if t]
        result[item_id] = " ".join(texts)
    return result


def _dedupe_segments_by_segment_id(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed = 0
    for seg in segments:
        seg_id = seg.get("segment_id")
        if seg_id is None:
            deduped.append(seg)
            continue
        seg_id_s = str(seg_id)
        if seg_id_s in seen:
            removed += 1
            continue
        seen.add(seg_id_s)
        deduped.append(seg)
    return deduped, removed


def _evaluate_split(name: str, asr_dir: Path, manifest_path: Path, normalize_numbers: bool = False) -> dict[str, Any]:
    manifest_rows = read_jsonl(manifest_path)
    segments_raw = read_jsonl(asr_dir / "segments.jsonl")
    segments, duplicate_segment_rows_removed = _dedupe_segments_by_segment_id(segments_raw)
    asr_summary_path = asr_dir / "summary.json"
    asr_summary = {}
    if asr_summary_path.exists():
        import json

        asr_summary = json.loads(asr_summary_path.read_text(encoding="utf-8"))

    hyp_by_item = _aggregate_hyp_by_item(segments)

    per_conv_rows: list[dict[str, Any]] = []
    corpus_ref_tokens: list[str] = []
    corpus_hyp_tokens: list[str] = []
    corpus_ref_chars: list[str] = []
    corpus_hyp_chars: list[str] = []
    by_lang_tokens: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"ref": [], "hyp": []})
    by_lang_chars: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"ref": [], "hyp": []})

    for rec in manifest_rows:
        item_id = str(rec.get("item_id"))
        ref_text = str(rec.get("transcript") or "")
        hyp_text = hyp_by_item.get(item_id, "")

        ref_tokens = _tokenize_mixed_words(ref_text, normalize_numbers=normalize_numbers)
        hyp_tokens = _tokenize_mixed_words(hyp_text, normalize_numbers=normalize_numbers)
        ref_chars = _tokenize_chars(ref_text, normalize_numbers=normalize_numbers)
        hyp_chars = _tokenize_chars(hyp_text, normalize_numbers=normalize_numbers)
        lang_bucket = _bucket_language(ref_text)

        w = _safe_wer(ref_tokens, hyp_tokens)
        c = _safe_wer(ref_chars, hyp_chars)

        corpus_ref_tokens.extend(ref_tokens)
        corpus_hyp_tokens.extend(hyp_tokens)
        corpus_ref_chars.extend(ref_chars)
        corpus_hyp_chars.extend(hyp_chars)
        by_lang_tokens[lang_bucket]["ref"].extend(ref_tokens)
        by_lang_tokens[lang_bucket]["hyp"].extend(hyp_tokens)
        by_lang_chars[lang_bucket]["ref"].extend(ref_chars)
        by_lang_chars[lang_bucket]["hyp"].extend(hyp_chars)

        per_conv_rows.append(
            {
                "split": name,
                "item_id": item_id,
                "utt_id": rec.get("utt_id"),
                "duration_sec": rec.get("duration_sec"),
                "lang_bucket": lang_bucket,
                "wer": w,
                "cer": c,
                "ref_word_tokens": len(ref_tokens),
                "hyp_word_tokens": len(hyp_tokens),
                "ref_chars": len(ref_chars),
                "hyp_chars": len(hyp_chars),
            }
        )

    per_lang_rows = []
    for lang in sorted(set(by_lang_tokens.keys()) | set(by_lang_chars.keys())):
        ref_t = by_lang_tokens[lang]["ref"]
        hyp_t = by_lang_tokens[lang]["hyp"]
        ref_c = by_lang_chars[lang]["ref"]
        hyp_c = by_lang_chars[lang]["hyp"]
        per_lang_rows.append(
            {
                "split": name,
                "lang_bucket": lang,
                "wer": _safe_wer(ref_t, hyp_t),
                "cer": _safe_wer(ref_c, hyp_c),
                "ref_word_tokens": len(ref_t),
                "ref_chars": len(ref_c),
            }
        )

    return {
        "split": name,
        "asr_dir": str(asr_dir),
        "manifest_path": str(manifest_path),
        "num_conversations": len(manifest_rows),
        "num_hyp_conversations": len(hyp_by_item),
        "segments_rows_raw": len(segments_raw),
        "segments_rows_used": len(segments),
        "duplicate_segment_rows_removed": duplicate_segment_rows_removed,
        "wer": _safe_wer(corpus_ref_tokens, corpus_hyp_tokens),
        "cer": _safe_wer(corpus_ref_chars, corpus_hyp_chars),
        "ref_word_tokens": len(corpus_ref_tokens),
        "ref_chars": len(corpus_ref_chars),
        "asr_runtime": {
            "elapsed_sec": asr_summary.get("elapsed_sec"),
            "total_audio_sec": asr_summary.get("total_audio_sec"),
            "rtf": asr_summary.get("rtf"),
            "audio_sec_per_wall_sec": asr_summary.get("audio_sec_per_wall_sec"),
            "segments_per_sec": asr_summary.get("segments_per_sec"),
            "segments": asr_summary.get("segments"),
            "model_name": asr_summary.get("model_name"),
        },
        "per_conversation_rows": per_conv_rows,
        "per_language_rows": per_lang_rows,
    }


def run_asr_eval(cfg: dict) -> Path:
    run_id = cfg.get("run", {}).get("run_id", "asr_eval_baseline_v1")
    out_dir = artifact_dir("asr_eval", run_id)
    eval_entries = cfg.get("inputs", {}).get("evaluations", [])
    if not eval_entries:
        raise ValueError("inputs.evaluations is required")

    normalize_numbers = bool(cfg.get("protocol", {}).get("normalize_numbers", False))
    protocol = {
        "version": cfg.get("protocol", {}).get("version", "asr_eval_v1"),
        "text_cleaning": "remove_tag_tokens_and_whitespace_fold",
        "wer_tokenization": "mixed_word_tokens: [A-Za-z0-9]+ or single CJK char",
        "cer_tokenization": "single alnum/CJK char tokens",
        "language_bucket": "reference_text_char_based: en|zh|mixed|unknown",
        "normalize_numbers": normalize_numbers,
    }

    split_summaries: list[dict[str, Any]] = []
    per_conv_rows: list[dict[str, Any]] = []
    per_lang_rows: list[dict[str, Any]] = []

    for ent in eval_entries:
        split_name = ent["name"]
        asr_dir = resolve_path(ent["asr_dir"])
        manifest_path = resolve_path(ent["manifest_path"])
        split_result = _evaluate_split(split_name, asr_dir, manifest_path, normalize_numbers=normalize_numbers)
        split_summaries.append(
            {
                k: v
                for k, v in split_result.items()
                if k not in {"per_conversation_rows", "per_language_rows"}
            }
        )
        per_conv_rows.extend(split_result["per_conversation_rows"])
        per_lang_rows.extend(split_result["per_language_rows"])

    write_jsonl(out_dir / "per_conversation.jsonl", per_conv_rows)
    write_json(out_dir / "protocol.json", protocol)
    write_json(
        out_dir / "summary.json",
        {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol["version"],
            "splits": split_summaries,
        },
    )

    _write_csv(
        out_dir / "per_split.csv",
        [
            "split",
            "num_conversations",
            "num_hyp_conversations",
            "segments_rows_raw",
            "segments_rows_used",
            "duplicate_segment_rows_removed",
            "wer",
            "cer",
            "ref_word_tokens",
            "ref_chars",
            "model_name",
            "segments",
            "elapsed_sec",
            "total_audio_sec",
            "rtf",
            "audio_sec_per_wall_sec",
            "segments_per_sec",
        ],
        [
            [
                s["split"],
                s["num_conversations"],
                s["num_hyp_conversations"],
                s.get("segments_rows_raw"),
                s.get("segments_rows_used"),
                s.get("duplicate_segment_rows_removed"),
                s["wer"],
                s["cer"],
                s["ref_word_tokens"],
                s["ref_chars"],
                s["asr_runtime"].get("model_name"),
                s["asr_runtime"].get("segments"),
                s["asr_runtime"].get("elapsed_sec"),
                s["asr_runtime"].get("total_audio_sec"),
                s["asr_runtime"].get("rtf"),
                s["asr_runtime"].get("audio_sec_per_wall_sec"),
                s["asr_runtime"].get("segments_per_sec"),
            ]
            for s in split_summaries
        ],
    )

    _write_csv(
        out_dir / "per_language.csv",
        ["split", "lang_bucket", "wer", "cer", "ref_word_tokens", "ref_chars"],
        [
            [
                r["split"],
                r["lang_bucket"],
                r["wer"],
                r["cer"],
                r["ref_word_tokens"],
                r["ref_chars"],
            ]
            for r in per_lang_rows
        ],
    )

    write_json(
        out_dir / "run_meta.json",
        {
            "stage": "asr_eval",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_splits": len(split_summaries),
            "num_conversations": len(per_conv_rows),
            "inputs": eval_entries,
        },
    )
    return out_dir
