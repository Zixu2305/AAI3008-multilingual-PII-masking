from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from src.utils.artifacts import artifact_dir
from src.utils.io import PROJECT_ROOT, read_jsonl, write_json


@dataclass
class AudioInput:
    item_id: str
    audio_path: Path
    duration_sec: float | None = None


_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_CHAR_RE = re.compile(r"[A-Za-z]")


def _detect_segment_language(text: str) -> tuple[str, dict[str, int]]:
    zh_chars = len(_ZH_CHAR_RE.findall(text))
    en_chars = len(_EN_CHAR_RE.findall(text))

    if zh_chars == 0 and en_chars == 0:
        return "unknown", {"zh_chars": 0, "en_chars": 0}
    if zh_chars > 0 and en_chars > 0:
        return "mixed", {"zh_chars": zh_chars, "en_chars": en_chars}
    if zh_chars > 0:
        return "zh", {"zh_chars": zh_chars, "en_chars": en_chars}
    return "en", {"zh_chars": zh_chars, "en_chars": en_chars}


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_resume_state(path: Path) -> tuple[set[str], int, int]:
    if not path.exists():
        return set(), -1, 0

    processed_item_ids: set[str] = set()
    max_seg_idx = -1
    rows_count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_count += 1
            try:
                row = json.loads(line)
            except Exception:
                continue

            source_item_id = row.get("source_item_id")
            if source_item_id:
                processed_item_ids.add(str(source_item_id))

            seg_id = str(row.get("segment_id") or "")
            if seg_id.startswith("seg_"):
                try:
                    max_seg_idx = max(max_seg_idx, int(seg_id.split("_", 1)[1]))
                except Exception:
                    continue

    return processed_item_ids, max_seg_idx, rows_count


def _load_previous_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_inputs(cfg: dict) -> list[AudioInput]:
    data_cfg = cfg.get("data", {})
    source_type = data_cfg.get("source_type", "local_dir")
    max_items = int(data_cfg.get("max_items", 0) or 0)

    if source_type == "manifest":
        manifest_path = Path(data_cfg.get("manifest", {}).get("path", ""))
        manifest_rows = read_jsonl(manifest_path)
        rows: list[AudioInput] = []
        for idx, row in enumerate(manifest_rows):
            audio_path = row.get("audio_path")
            if not audio_path:
                continue
            p = Path(audio_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            duration = row.get("duration_sec")
            rows.append(
                AudioInput(
                    item_id=str(row.get("item_id") or f"item_{idx:06d}"),
                    audio_path=p,
                    duration_sec=float(duration) if duration is not None else None,
                )
            )
        return rows[:max_items] if max_items > 0 else rows

    if source_type == "local_dir":
        audio_glob = data_cfg.get("local", {}).get("audio_glob", "data/raw/**/*.wav")
        rows: list[AudioInput] = []
        for idx, path in enumerate(sorted(PROJECT_ROOT.glob(audio_glob))):
            rows.append(AudioInput(item_id=f"local_{idx:06d}", audio_path=path, duration_sec=None))
        return rows[:max_items] if max_items > 0 else rows

    raise ValueError(f"Unsupported data.source_type='{source_type}'. Use 'manifest' or 'local_dir'.")


def run_asr(cfg: dict) -> Path:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faster-whisper is not installed. Install dependencies and rerun: "
            "`pip install -r requirements.txt`."
        ) from exc

    run_id = cfg.get("run", {}).get("run_id", "asr_run")
    run_cfg = cfg.get("run", {})
    out_dir = artifact_dir("asr", run_id)
    segments_path = out_dir / "segments.jsonl"
    summary_path = out_dir / "summary.json"
    progress_path = out_dir / "progress.json"

    resume = bool(run_cfg.get("resume", True))
    checkpoint_every_files = max(1, int(run_cfg.get("checkpoint_every_files", 5) or 5))
    progress_every_files = max(1, int(run_cfg.get("progress_every_files", 1) or 1))
    write_progress = bool(run_cfg.get("write_progress", True))

    inputs_all = _load_inputs(cfg)
    manifest_item_ids = {x.item_id for x in inputs_all}

    resumed_item_ids: set[str] = set()
    prev_segments_count = 0
    prev_max_seg_idx = -1
    if resume:
        processed_item_ids, prev_max_seg_idx, prev_segments_count = _load_resume_state(segments_path)
        resumed_item_ids = processed_item_ids & manifest_item_ids
    else:
        segments_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    inputs = [x for x in inputs_all if x.item_id not in resumed_item_ids]
    skipped_existing = len(inputs_all) - len(inputs)
    if skipped_existing > 0:
        print(f"Resume enabled: skipping {skipped_existing} previously processed items.")

    asr_cfg = cfg.get("asr", {})
    language = asr_cfg.get("language", "auto")
    model_name = asr_cfg.get("model_name", "small")
    device = asr_cfg.get("device", "cpu")
    compute_type = asr_cfg.get("compute_type", "float32")
    beam_size = int(asr_cfg.get("beam_size", 5))
    vad_filter = bool(asr_cfg.get("vad_filter", True))
    word_timestamps = bool(asr_cfg.get("word_timestamps", True))
    condition_on_previous_text = bool(asr_cfg.get("condition_on_previous_text", False))

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    started = perf_counter()

    segments_buffer: list[dict[str, Any]] = []
    files_processed = 0
    files_failed = 0
    files_since_flush = 0
    seg_idx = prev_max_seg_idx + 1
    segments_written = prev_segments_count
    interrupted = False

    prev_summary = _load_previous_summary(summary_path) if resume else {}
    prev_elapsed = float(prev_summary.get("elapsed_sec", 0.0) or 0.0)
    prev_files_processed = int(prev_summary.get("files_processed", len(resumed_item_ids)) or len(resumed_item_ids))
    prev_files_failed = int(prev_summary.get("files_failed", 0) or 0)

    known_duration_by_item = {
        x.item_id: float(x.duration_sec) for x in inputs_all if x.duration_sec is not None
    }
    total_audio_sec = float(sum(known_duration_by_item.values())) if known_duration_by_item else None
    resumed_audio_sec = float(sum(known_duration_by_item.get(x, 0.0) for x in resumed_item_ids))
    processed_audio_sec_this_run = 0.0
    failed_audio_sec_this_run = 0.0

    def flush_checkpoint() -> None:
        nonlocal segments_written, files_since_flush
        if not segments_buffer:
            return
        _append_jsonl(segments_path, segments_buffer)
        segments_written += len(segments_buffer)
        segments_buffer.clear()
        files_since_flush = 0

    def write_progress_snapshot(status: str) -> None:
        if not write_progress:
            return
        elapsed_this_run = perf_counter() - started
        done_items = skipped_existing + files_processed + files_failed
        remaining_items = max(0, len(inputs_all) - done_items)
        done_in_this_run = files_processed + files_failed
        eta_sec = (elapsed_this_run / done_in_this_run * remaining_items) if done_in_this_run > 0 else None
        done_audio_sec = resumed_audio_sec + processed_audio_sec_this_run + failed_audio_sec_this_run
        payload = {
            "stage": "asr",
            "status": status,
            "run_id": run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "inputs_total": len(inputs_all),
            "resume_enabled": resume,
            "resumed_items": skipped_existing,
            "files_processed": prev_files_processed + files_processed,
            "files_failed": prev_files_failed + files_failed,
            "segments_written": segments_written + len(segments_buffer),
            "done_items": done_items,
            "remaining_items": remaining_items,
            "elapsed_sec_this_run": elapsed_this_run,
            "eta_sec": eta_sec,
            "total_audio_sec": total_audio_sec,
            "done_audio_sec": done_audio_sec if total_audio_sec is not None else None,
            "done_audio_ratio": (done_audio_sec / total_audio_sec) if (total_audio_sec and total_audio_sec > 0) else None,
            "checkpoint_every_files": checkpoint_every_files,
            "progress_every_files": progress_every_files,
        }
        write_json(progress_path, payload)

    try:
        for item in inputs:
            if not item.audio_path.exists():
                files_failed += 1
                failed_audio_sec_this_run += float(item.duration_sec or 0.0)
                continue

            try:
                seg_iter, info = model.transcribe(
                    str(item.audio_path),
                    language=None if language == "auto" else language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    word_timestamps=word_timestamps,
                    condition_on_previous_text=condition_on_previous_text,
                )
            except Exception as exc:
                files_failed += 1
                failed_audio_sec_this_run += float(item.duration_sec or 0.0)
                print(f"[WARN] ASR failed for {item.item_id}: {exc}")
                continue

            files_processed += 1
            processed_audio_sec_this_run += float(item.duration_sec or 0.0)
            seg_count_for_file = 0
            for seg in seg_iter:
                text = (seg.text or "").strip()
                segment_language, lang_counts = _detect_segment_language(text)
                segments_buffer.append(
                    {
                        "segment_id": f"seg_{seg_idx:06d}",
                        "source_item_id": item.item_id,
                        "audio_path": str(item.audio_path.relative_to(PROJECT_ROOT)),
                        "text": text,
                        "language": segment_language,
                        "language_counts": lang_counts,
                        "detected_file_language": info.language or language,
                        "detected_file_language_probability": getattr(info, "language_probability", None),
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "source": "faster_whisper",
                    }
                )
                seg_idx += 1
                seg_count_for_file += 1

            if seg_count_for_file == 0:
                segments_buffer.append(
                    {
                        "segment_id": f"seg_{seg_idx:06d}",
                        "source_item_id": item.item_id,
                        "audio_path": str(item.audio_path.relative_to(PROJECT_ROOT)),
                        "text": "",
                        "language": "unknown",
                        "language_counts": {"zh_chars": 0, "en_chars": 0},
                        "detected_file_language": info.language or language,
                        "detected_file_language_probability": getattr(info, "language_probability", None),
                        "start": 0.0,
                        "end": None,
                        "source": "faster_whisper",
                    }
                )
                seg_idx += 1

            files_since_flush += 1

            if files_since_flush >= checkpoint_every_files:
                flush_checkpoint()
                write_progress_snapshot(status="running")

            done_in_this_run = files_processed + files_failed
            if done_in_this_run % progress_every_files == 0:
                elapsed_this_run = perf_counter() - started
                done_items = skipped_existing + done_in_this_run
                remaining_items = max(0, len(inputs_all) - done_items)
                eta_sec = (elapsed_this_run / done_in_this_run * remaining_items) if done_in_this_run > 0 else 0.0
                print(
                    f"[ASR] done={done_items}/{len(inputs_all)} "
                    f"(processed={prev_files_processed + files_processed}, failed={prev_files_failed + files_failed}) "
                    f"segments={segments_written + len(segments_buffer)} "
                    f"elapsed={elapsed_this_run:.1f}s eta={eta_sec:.1f}s current={item.item_id}"
                )
    except KeyboardInterrupt:
        interrupted = True
        print("[ASR] Interrupted; flushing checkpoint for resume.")

    flush_checkpoint()

    elapsed_this_run = perf_counter() - started
    elapsed_total = prev_elapsed + elapsed_this_run
    total_files_processed = prev_files_processed + files_processed
    total_files_failed = prev_files_failed + files_failed

    write_progress_snapshot(status="interrupted" if interrupted else "completed")

    write_json(
        summary_path,
        {
            "run_id": run_id,
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
            "inputs_total": len(inputs_all),
            "files_processed": total_files_processed,
            "files_failed": total_files_failed,
            "segments": segments_written,
            "elapsed_sec": elapsed_total,
            "elapsed_sec_this_run": elapsed_this_run,
            "total_audio_sec": total_audio_sec,
            "total_audio_min": (total_audio_sec / 60.0) if total_audio_sec is not None else None,
            "rtf": (elapsed_total / total_audio_sec) if (total_audio_sec and total_audio_sec > 0) else None,
            "audio_sec_per_wall_sec": (total_audio_sec / elapsed_total)
            if (total_audio_sec and elapsed_total > 0)
            else None,
            "segments_per_sec": (segments_written / elapsed_total) if elapsed_total > 0 else 0.0,
            "status": "interrupted" if interrupted else "completed",
            "resume_enabled": resume,
            "resumed_items": skipped_existing,
            "checkpoint_every_files": checkpoint_every_files,
            "progress_every_files": progress_every_files,
        },
    )
    write_json(
        out_dir / "run_meta.json",
        {
            "stage": "asr",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "items": segments_written,
            "inputs_total": len(inputs_all),
            "files_processed": total_files_processed,
            "files_failed": total_files_failed,
            "model_name": model_name,
            "engine": "faster_whisper",
            "status": "interrupted" if interrupted else "completed",
            "resume_enabled": resume,
            "resumed_items": skipped_existing,
        },
    )

    return out_dir
