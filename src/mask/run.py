from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.artifacts import artifact_dir
from src.utils.io import PROJECT_ROOT, read_jsonl, write_json, write_jsonl


def _resolve_audio_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_asr_input_dir(cfg: dict, pii_input: Path) -> Path | None:
    run_cfg = cfg.get("run", {})
    asr_dir = str(run_cfg.get("input_asr_dir", "")).strip()
    if asr_dir:
        return Path(asr_dir)

    pii_meta = _read_json(pii_input / "run_meta.json")
    inferred = str(pii_meta.get("input_asr_dir", "")).strip()
    if inferred:
        return Path(inferred)
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _normalize_words(raw_words: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw_words, list):
        return out

    for w in raw_words:
        if not isinstance(w, dict):
            continue
        start = _to_float(w.get("start"))
        end = _to_float(w.get("end"))
        if start is None or end is None or end <= start:
            continue
        probability = _to_float(w.get("probability"))
        out.append(
            {
                "word": str(w.get("word", "") or ""),
                "start": start,
                "end": end,
                "probability": probability,
            }
        )
    return out


def _load_asr_segment_index(asr_input_dir: Path | None) -> dict[str, dict[str, Any]]:
    if asr_input_dir is None:
        return {}

    segments = read_jsonl(asr_input_dir / "segments.jsonl")
    out: dict[str, dict[str, Any]] = {}
    for seg in segments:
        seg_id = str(seg.get("segment_id", "")).strip()
        if not seg_id:
            continue
        out[seg_id] = {
            "audio_path": seg.get("audio_path"),
            "text": str(seg.get("text", "")),
            "start": _to_float(seg.get("start")),
            "end": _to_float(seg.get("end")),
            "words": _normalize_words(seg.get("words", [])),
        }
    return out


def _build_word_char_map(text: str, words: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Best-effort mapping from ASR words to character offsets in segment text."""
    if not text or not words:
        return []

    mapping: list[dict[str, float]] = []
    cursor = 0

    for w in words:
        token = str(w.get("word", "") or "")
        if not token:
            continue

        candidates = [token]
        stripped = token.strip()
        if stripped and stripped != token:
            candidates.append(stripped)

        start_char = None
        matched = None
        for cand in candidates:
            idx = text.find(cand, cursor)
            if idx < 0 and cursor > 0:
                idx = text.find(cand)
            if idx >= 0:
                start_char = idx
                matched = cand
                break

        if start_char is None or matched is None:
            continue

        end_char = start_char + len(matched)
        start_sec = _to_float(w.get("start"))
        end_sec = _to_float(w.get("end"))
        if start_sec is None or end_sec is None or end_sec <= start_sec:
            continue

        mapping.append(
            {
                "start_char": float(start_char),
                "end_char": float(end_char),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
        )
        cursor = end_char

    return mapping


def _span_to_word_interval(
    start_char: int,
    end_char: int,
    word_char_map: list[dict[str, float]],
) -> tuple[float, float] | None:
    if not word_char_map:
        return None

    overlaps = []
    for w in word_char_map:
        ws = int(w["start_char"])
        we = int(w["end_char"])
        if end_char <= ws or start_char >= we:
            continue
        overlaps.append(w)

    if overlaps:
        return (
            min(float(w["start_sec"]) for w in overlaps),
            max(float(w["end_sec"]) for w in overlaps),
        )

    center = 0.5 * (start_char + end_char)

    def _dist(item: dict[str, float]) -> float:
        ws = float(item["start_char"])
        we = float(item["end_char"])
        if ws <= center <= we:
            return 0.0
        return min(abs(center - ws), abs(center - we))

    nearest = min(word_char_map, key=_dist)
    return float(nearest["start_sec"]), float(nearest["end_sec"])


def _span_to_interval(
    span: dict[str, Any],
    text: str,
    seg_start: float,
    seg_end: float,
    lead_sec: float,
    tail_sec: float,
    min_interval_sec: float,
    word_char_map: list[dict[str, float]],
) -> tuple[float, float, str] | None:
    if not math.isfinite(seg_start) or not math.isfinite(seg_end):
        return None
    if seg_end <= seg_start:
        return None

    start_char = span.get("start")
    end_char = span.get("end")
    if not isinstance(start_char, int) or not isinstance(end_char, int):
        return None

    text_len = len(text)
    if text_len > 0:
        start_char = max(0, min(start_char, text_len))
        end_char = max(start_char + 1, min(end_char, text_len))
    else:
        start_char = 0
        end_char = 1

    alignment_mode = "char_ratio"
    start_sec = None
    end_sec = None

    word_interval = _span_to_word_interval(start_char, end_char, word_char_map)
    if word_interval is not None:
        start_sec, end_sec = word_interval
        alignment_mode = "word"

    if start_sec is None or end_sec is None:
        duration = seg_end - seg_start
        if text_len <= 0:
            start_sec, end_sec = seg_start, seg_end
        else:
            ratio_start = start_char / text_len
            ratio_end = end_char / text_len
            start_sec = seg_start + ratio_start * duration
            end_sec = seg_start + ratio_end * duration

    start_sec = max(seg_start, float(start_sec) - lead_sec)
    end_sec = min(seg_end, float(end_sec) + tail_sec)

    if end_sec <= start_sec:
        return None

    if (end_sec - start_sec) < min_interval_sec:
        mid = 0.5 * (start_sec + end_sec)
        half = 0.5 * min_interval_sec
        start_sec = max(seg_start, mid - half)
        end_sec = min(seg_end, mid + half)
        if end_sec <= start_sec:
            return None

    return start_sec, end_sec, alignment_mode


def _merge_intervals(intervals: list[tuple[float, float]], merge_gap_sec: float) -> list[tuple[float, float]]:
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]

    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= (prev_end + merge_gap_sec):
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged if e > s]


def _build_output_path(masked_audio_dir: Path, src: Path, masked: bool) -> Path:
    try:
        rel = src.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = Path(src.name)

    if masked:
        out_rel = rel.parent / f"{rel.stem}.masked.wav"
    else:
        out_rel = rel

    dst = masked_audio_dir / out_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst


def _mask_with_pydub(
    src: Path,
    dst: Path,
    intervals: list[tuple[float, float]],
    mode: str,
    beep_freq_hz: int,
    beep_gain_db: float,
) -> str:
    from pydub import AudioSegment
    from pydub.generators import Sine

    audio = AudioSegment.from_file(src)
    total_ms = len(audio)
    out = audio

    for start_sec, end_sec in intervals:
        start_ms = max(0, int(round(start_sec * 1000.0)))
        end_ms = min(total_ms, int(round(end_sec * 1000.0)))
        if end_ms <= start_ms:
            continue

        duration_ms = end_ms - start_ms
        if mode == "beep":
            repl = Sine(beep_freq_hz).to_audio_segment(duration=duration_ms, volume=beep_gain_db)
            repl = repl.set_frame_rate(audio.frame_rate).set_channels(audio.channels).set_sample_width(audio.sample_width)
        else:
            repl = AudioSegment.silent(duration=duration_ms, frame_rate=audio.frame_rate)
            repl = repl.set_channels(audio.channels).set_sample_width(audio.sample_width)

        out = out[:start_ms] + repl + out[end_ms:]

    out.export(dst, format="wav")
    return "pydub"


def _mask_with_soundfile(
    src: Path,
    dst: Path,
    intervals: list[tuple[float, float]],
    mode: str,
    beep_freq_hz: int,
    beep_amplitude: float,
) -> str:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(src, dtype="float32")
    n_samples = data.shape[0]

    for start_sec, end_sec in intervals:
        start_idx = max(0, int(round(start_sec * sr)))
        end_idx = min(n_samples, int(round(end_sec * sr)))
        if end_idx <= start_idx:
            continue

        if mode == "beep":
            n = end_idx - start_idx
            t = np.arange(n, dtype=np.float32) / float(sr)
            tone = (beep_amplitude * np.sin(2.0 * np.pi * float(beep_freq_hz) * t)).astype(np.float32)
            if data.ndim == 1:
                data[start_idx:end_idx] = tone
            else:
                data[start_idx:end_idx, :] = tone[:, None]
        else:
            if data.ndim == 1:
                data[start_idx:end_idx] = 0.0
            else:
                data[start_idx:end_idx, :] = 0.0

    sf.write(dst, data, sr)
    return "soundfile"


def _apply_mask_audio(
    src: Path,
    dst: Path,
    intervals: list[tuple[float, float]],
    mode: str,
    beep_freq_hz: int,
    beep_gain_db: float,
    beep_amplitude: float,
) -> str:
    try:
        return _mask_with_pydub(
            src=src,
            dst=dst,
            intervals=intervals,
            mode=mode,
            beep_freq_hz=beep_freq_hz,
            beep_gain_db=beep_gain_db,
        )
    except Exception as pydub_exc:
        try:
            return _mask_with_soundfile(
                src=src,
                dst=dst,
                intervals=intervals,
                mode=mode,
                beep_freq_hz=beep_freq_hz,
                beep_amplitude=beep_amplitude,
            )
        except Exception as sf_exc:
            raise RuntimeError(
                f"audio masking failed with pydub ({pydub_exc}) and soundfile ({sf_exc})"
            )


def run_mask(cfg: dict) -> Path:
    run_id = cfg.get("run", {}).get("run_id", "masked_run")
    out_dir = artifact_dir("masked", run_id)
    masked_audio_dir = out_dir / "masked_audio"
    masked_audio_dir.mkdir(parents=True, exist_ok=True)

    pii_input = Path(cfg.get("run", {}).get("input_pii_dir", f"data/runs/pii/{run_id}"))
    spans_rows = read_jsonl(pii_input / "spans.jsonl")

    asr_input_dir = _resolve_asr_input_dir(cfg, pii_input)
    asr_index = _load_asr_segment_index(asr_input_dir)

    mask_cfg = cfg.get("mask", {})
    mode = str(mask_cfg.get("mode", "silence")).strip().lower()
    if mode not in {"silence", "beep"}:
        mode = "silence"

    # Keep backward compatibility: if only pad_sec exists, reuse it for lead/tail.
    legacy_pad = max(0.0, float(mask_cfg.get("pad_sec", 0.0) or 0.0))
    lead_sec = max(0.0, float(mask_cfg.get("lead_sec", legacy_pad if legacy_pad > 0 else 0.00) or 0.0))
    tail_sec = max(0.0, float(mask_cfg.get("tail_sec", legacy_pad if legacy_pad > 0 else 0.12) or 0.0))

    min_interval_sec = max(0.0, float(mask_cfg.get("min_interval_sec", 0.20) or 0.0))
    merge_gap_sec = max(0.0, float(mask_cfg.get("merge_gap_sec", 0.05) or 0.0))
    beep_freq_hz = int(mask_cfg.get("beep_freq_hz", 1000) or 1000)
    beep_gain_db = float(mask_cfg.get("beep_gain_db", -6.0) or -6.0)
    beep_amplitude = float(mask_cfg.get("beep_amplitude", 0.20) or 0.20)

    by_source: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "sample_segment_ids": [],
            "first_segment_id": None,
            "last_segment_id": None,
            "segments_count": 0,
            "segments_with_words": 0,
            "segments_without_words": 0,
            "masked_span_count": 0,
            "unmapped_spans": 0,
            "word_aligned_spans": 0,
            "ratio_aligned_spans": 0,
            "intervals_raw": [],
        }
    )

    for row in spans_rows:
        seg_id = str(row.get("segment_id", "")).strip()
        seg = asr_index.get(seg_id)

        source_audio = row.get("audio_path")
        if not source_audio and seg:
            source_audio = seg.get("audio_path")
        if not source_audio:
            continue

        entry = by_source[str(source_audio)]
        if seg_id:
            if entry["first_segment_id"] is None:
                entry["first_segment_id"] = seg_id
            entry["last_segment_id"] = seg_id
            if len(entry["sample_segment_ids"]) < 10:
                entry["sample_segment_ids"].append(seg_id)

        entry["segments_count"] += 1
        if seg and isinstance(seg.get("words"), list) and len(seg.get("words", [])) > 0:
            entry["segments_with_words"] += 1
        elif seg:
            entry["segments_without_words"] += 1
        spans = row.get("spans", []) or []
        entry["masked_span_count"] += len(spans)

        if not spans:
            continue
        if not seg:
            entry["unmapped_spans"] += len(spans)
            continue

        seg_start = _to_float(seg.get("start"))
        seg_end = _to_float(seg.get("end"))
        if seg_start is None or seg_end is None:
            entry["unmapped_spans"] += len(spans)
            continue

        text = str(row.get("text", "")) or str(seg.get("text", ""))
        word_char_map = _build_word_char_map(text, seg.get("words", []))

        mapped = 0
        for span in spans:
            interval = _span_to_interval(
                span=span,
                text=text,
                seg_start=seg_start,
                seg_end=seg_end,
                lead_sec=lead_sec,
                tail_sec=tail_sec,
                min_interval_sec=min_interval_sec,
                word_char_map=word_char_map,
            )
            if interval is None:
                continue
            start_sec, end_sec, align_mode = interval
            entry["intervals_raw"].append((start_sec, end_sec))
            if align_mode == "word":
                entry["word_aligned_spans"] += 1
            else:
                entry["ratio_aligned_spans"] += 1
            mapped += 1

        if mapped < len(spans):
            entry["unmapped_spans"] += (len(spans) - mapped)

    audit_rows = []
    files_missing = 0
    files_masked = 0
    files_copied_no_spans = 0
    files_failed_mask = 0

    for source_audio, stats in sorted(by_source.items()):
        src = _resolve_audio_path(source_audio)
        output_audio = None
        method = None
        mask_engine = None
        mask_error = None

        raw_intervals = [(float(s), float(e)) for s, e in stats.pop("intervals_raw", []) if e > s]
        merged_intervals = _merge_intervals(raw_intervals, merge_gap_sec=merge_gap_sec)
        masked_duration_sec = float(sum(e - s for s, e in merged_intervals))

        if src.exists() and src.is_file():
            if merged_intervals:
                dst = _build_output_path(masked_audio_dir, src, masked=True)
                try:
                    mask_engine = _apply_mask_audio(
                        src=src,
                        dst=dst,
                        intervals=merged_intervals,
                        mode=mode,
                        beep_freq_hz=beep_freq_hz,
                        beep_gain_db=beep_gain_db,
                        beep_amplitude=beep_amplitude,
                    )
                    output_audio = str(dst)
                    method = f"time_aligned_{mode}_mask"
                    files_masked += 1
                except Exception as exc:
                    mask_error = str(exc)
                    files_failed_mask += 1
                    # Keep pipeline runnable even if local audio backends are missing.
                    dst = _build_output_path(masked_audio_dir, src, masked=False)
                    shutil.copy2(src, dst)
                    output_audio = str(dst)
                    method = "fallback_copy_after_mask_error"
            else:
                dst = _build_output_path(masked_audio_dir, src, masked=False)
                shutil.copy2(src, dst)
                output_audio = str(dst)
                method = "copy_no_detected_spans"
                files_copied_no_spans += 1
        else:
            files_missing += 1
            method = "source_missing"

        audit_rows.append(
            {
                "source_audio": source_audio,
                "masked_audio": output_audio,
                "segments_count": stats["segments_count"],
                "first_segment_id": stats["first_segment_id"],
                "last_segment_id": stats["last_segment_id"],
                "sample_segment_ids": stats["sample_segment_ids"],
                "masked_span_count": stats["masked_span_count"],
                "unmapped_spans": stats["unmapped_spans"],
                "segments_with_words": stats["segments_with_words"],
                "segments_without_words": stats["segments_without_words"],
                "word_aligned_spans": stats["word_aligned_spans"],
                "ratio_aligned_spans": stats["ratio_aligned_spans"],
                "intervals_count_raw": len(raw_intervals),
                "intervals_count_merged": len(merged_intervals),
                "masked_duration_sec": round(masked_duration_sec, 3),
                "time_intervals_sec": [
                    {"start": round(s, 3), "end": round(e, 3)}
                    for s, e in merged_intervals[:200]
                ],
                "method": method,
                "mask_mode": mode,
                "mask_engine": mask_engine,
                "mask_error": mask_error,
            }
        )

    write_jsonl(out_dir / "audit.jsonl", audit_rows)
    total_word_aligned_spans = int(sum(int(r.get("word_aligned_spans", 0) or 0) for r in audit_rows))
    total_ratio_aligned_spans = int(sum(int(r.get("ratio_aligned_spans", 0) or 0) for r in audit_rows))
    sources_without_words = int(
        sum(
            1
            for r in audit_rows
            if int(r.get("segments_count", 0) or 0) > 0 and int(r.get("segments_with_words", 0) or 0) == 0
        )
    )
    write_json(
        out_dir / "run_meta.json",
        {
            "stage": "masked",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_pii_dir": str(pii_input),
            "input_asr_dir": str(asr_input_dir) if asr_input_dir else None,
            "items": len(audit_rows),
            "source_rows": len(by_source),
            "input_rows": len(spans_rows),
            "files_masked": files_masked,
            "files_copied_no_spans": files_copied_no_spans,
            "files_failed_mask": files_failed_mask,
            "files_missing": files_missing,
            "word_aligned_spans": total_word_aligned_spans,
            "ratio_aligned_spans": total_ratio_aligned_spans,
            "sources_without_word_timestamps": sources_without_words,
            "mask": {
                "mode": mode,
                "lead_sec": lead_sec,
                "tail_sec": tail_sec,
                "min_interval_sec": min_interval_sec,
                "merge_gap_sec": merge_gap_sec,
                "beep_freq_hz": beep_freq_hz,
            },
        },
    )

    return out_dir
