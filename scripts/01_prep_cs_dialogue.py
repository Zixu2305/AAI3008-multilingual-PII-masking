#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import re
import tarfile
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import soundfile as sf


def _detect_audio_column(features: dict[str, Any], audio_feature_type: type) -> str:
    for name, feature in features.items():
        if isinstance(feature, audio_feature_type):
            return name
    for candidate in ("audio", "speech", "wav"):
        if candidate in features:
            return candidate
    raise ValueError("Could not detect audio column")


def _detect_text_column(columns: list[str]) -> str:
    candidates = ("text", "transcript", "sentence", "utterance", "normalized_text", "content")
    for c in candidates:
        if c in columns:
            return c
    for c in columns:
        c_low = c.lower()
        if "text" in c_low or "transcript" in c_low:
            return c
    raise ValueError(f"Could not detect transcript column in {columns}")


def _normalize_audio(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr == target_sr or len(audio) == 0:
        return audio
    old_idx = np.arange(len(audio), dtype=np.float64)
    new_len = max(1, int(round(len(audio) * (target_sr / float(sr)))))
    new_idx = np.linspace(0, len(audio) - 1, num=new_len, dtype=np.float64)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def _parse_long_index(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append({"utt_id": parts[0], "textgrid_rel": parts[1], "wav_rel": parts[2]})
    return rows


def _extract_utt_id_from_item_id(item_id: str | None) -> str | None:
    if not item_id:
        return None
    parts = str(item_id).split("_", 2)
    if len(parts) < 3:
        return None
    return parts[2]


def _load_excluded_utt_ids(manifest_path: Path | None, utt_ids_file: Path | None) -> set[str]:
    excluded: set[str] = set()

    if manifest_path is not None and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                utt_id = rec.get("utt_id") or _extract_utt_id_from_item_id(rec.get("item_id"))
                if utt_id:
                    excluded.add(str(utt_id))

    if utt_ids_file is not None and utt_ids_file.exists():
        with utt_ids_file.open("r", encoding="utf-8") as f:
            for line in f:
                x = line.strip()
                if x:
                    excluded.add(x)

    return excluded


def _parse_textgrid_text(text: str) -> str:
    # Keep simple: concatenate all non-empty text tiers.
    texts = re.findall(r'text\s*=\s*"(.*?)"', text, flags=re.DOTALL)
    cleaned: list[str] = []
    for t in texts:
        x = t.strip()
        if x:
            cleaned.append(x)
    return " ".join(cleaned)


def _extract_speaker_session(utt_id: str) -> tuple[str | None, str | None]:
    m = re.search(r"_U(\d+)_S(\d+)$", utt_id)
    if not m:
        return None, None
    return f"U{m.group(1)}", f"S{m.group(2)}"


class _ConcatReader:
    def __init__(self, parts: list[Path]):
        self.parts = parts
        self.i = 0
        self.fh = None

    def _open_next(self) -> bool:
        if self.fh is not None:
            self.fh.close()
            self.fh = None
        if self.i >= len(self.parts):
            return False
        self.fh = self.parts[self.i].open("rb")
        self.i += 1
        return True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        out = bytearray()
        remaining = size
        while size < 0 or remaining > 0:
            if self.fh is None and not self._open_next():
                break
            chunk_size = 1024 * 1024 if size < 0 else remaining
            data = self.fh.read(chunk_size)
            if not data:
                if not self._open_next():
                    break
                continue
            out.extend(data)
            if size > 0:
                remaining -= len(data)
        return bytes(out)

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def _extract_members(parts: list[Path], needed_members: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    reader = _ConcatReader(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            for member in tf:
                if len(found) == len(needed_members):
                    break
                if not member.isfile():
                    continue
                name = member.name.lstrip("./")
                if name not in needed_members:
                    continue
                ef = tf.extractfile(member)
                if ef is None:
                    continue
                found[name] = ef.read()
    finally:
        reader.close()
    return found


def _duration_sec_from_wav_bytes(wav_bytes: bytes) -> tuple[float, int]:
    info = sf.info(io.BytesIO(wav_bytes))
    sr = int(info.samplerate)
    duration = float(info.frames) / float(sr) if sr > 0 else 0.0
    return duration, sr


def _write_manifest_and_stats(
    manifest_path: Path,
    stats_path: Path,
    records: list[dict[str, Any]],
    max_minutes: float,
    max_items: int,
) -> None:
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    durations = [float(r.get("duration_sec", 0.0)) for r in records]
    text_lengths = [len(str(r.get("transcript") or "")) for r in records]
    total_duration_sec = float(sum(durations))
    stats = {
        "max_minutes": float(max_minutes),
        "max_items": int(max_items),
        "selected_count": len(records),
        "total_duration_sec": total_duration_sec,
        "total_duration_min": total_duration_sec / 60.0,
        "avg_duration_sec": float(mean(durations)) if durations else 0.0,
        "min_duration_sec": float(min(durations)) if durations else 0.0,
        "max_duration_sec": float(max(durations)) if durations else 0.0,
        "transcript_length": {
            "avg_chars": float(mean(text_lengths)) if text_lengths else 0.0,
            "min_chars": int(min(text_lengths)) if text_lengths else 0,
            "max_chars": int(max(text_lengths)) if text_lengths else 0,
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_hf_duration_budget(
    ds: Any,
    split: str,
    seed: int,
    max_minutes: float,
    max_items: int,
    materialize_audio: bool,
    out_root: Path,
    audio_dir: Path,
    excluded_utt_ids: set[str],
) -> list[dict[str, Any]]:
    from datasets import Audio

    audio_col = _detect_audio_column(ds.features, Audio)
    text_col = _detect_text_column(ds.column_names)
    print(f"Detected columns -> audio: {audio_col}, transcript: {text_col}")

    rng = random.Random(seed)
    candidate_indices = list(range(len(ds)))
    rng.shuffle(candidate_indices)

    target_sec = max_minutes * 60.0
    records: list[dict[str, Any]] = []
    total_sec = 0.0

    for idx in candidate_indices:
        if len(records) >= max_items:
            break
        row = ds[idx]
        audio_obj = row.get(audio_col) or {}
        arr = np.asarray(audio_obj.get("array", []), dtype=np.float32)
        orig_sr = int(audio_obj.get("sampling_rate", 16000)) if audio_obj else 16000
        duration_sec = float(len(arr) / orig_sr) if orig_sr > 0 else 0.0

        item_id = f"{split}_{idx:08d}"
        out_rel = None
        if materialize_audio:
            norm = _normalize_audio(arr, orig_sr, 16000)
            out_wav = audio_dir / f"{item_id}.wav"
            sf.write(out_wav, norm, 16000)
            out_rel = f"{out_root.as_posix()}/audio/{item_id}.wav"

        record = {
            "item_id": item_id,
            "utt_id": row.get("utt_id") if "utt_id" in row else None,
            "split": split,
            "source": "hf",
            "audio_path": out_rel,
            "sampling_rate": orig_sr,
            "duration_sec": duration_sec,
            "transcript": str(row.get(text_col) or ""),
            "speaker_id": row.get("speaker_id") if "speaker_id" in row else None,
            "session_id": row.get("session_id") if "session_id" in row else None,
        }
        records.append(record)
        total_sec += duration_sec

        if total_sec >= target_sec:
            break

    return records


def _select_snapshot_duration_budget(
    snapshot_dir: Path,
    split: str,
    seed: int,
    max_minutes: float,
    max_items: int,
    materialize_audio: bool,
    out_root: Path,
    audio_dir: Path,
    excluded_utt_ids: set[str],
) -> list[dict[str, Any]]:
    index_path = snapshot_dir / "data" / "index" / "long_wav" / f"{split}.txt"
    if not index_path.exists():
        raise RuntimeError(f"Missing long_wav index: {index_path}")

    rows = _parse_long_index(index_path)
    if not rows:
        raise RuntimeError(f"No rows parsed from {index_path}")

    rng = random.Random(seed)
    rng.shuffle(rows)

    parts = sorted((snapshot_dir / "data" / "long_wav").glob("long_wav.tar.gz*"))
    if not parts:
        raise RuntimeError("No long_wav split archives found under snapshot data/long_wav")

    target_sec = max_minutes * 60.0
    records: list[dict[str, Any]] = []
    total_sec = 0.0
    cursor = 0
    batch_size = 64

    print(f"Using long_wav snapshot fallback at: {snapshot_dir}")
    while cursor < len(rows) and len(records) < max_items and total_sec < target_sec:
        batch_rows = rows[cursor : cursor + batch_size]
        cursor += batch_size

        needed: set[str] = set()
        for r in batch_rows:
            needed.add(r["wav_rel"].lstrip("./"))
            needed.add(r["textgrid_rel"].lstrip("./"))

        extracted = _extract_members(parts, needed)
        for r in batch_rows:
            if len(records) >= max_items or total_sec >= target_sec:
                break

            utt_id = r["utt_id"]
            if utt_id in excluded_utt_ids:
                continue
            wav_key = r["wav_rel"].lstrip("./")
            tg_key = r["textgrid_rel"].lstrip("./")
            wav_bytes = extracted.get(wav_key)
            if wav_bytes is None:
                continue

            duration_sec, orig_sr = _duration_sec_from_wav_bytes(wav_bytes)
            transcript = ""
            tg_bytes = extracted.get(tg_key)
            if tg_bytes:
                transcript = _parse_textgrid_text(tg_bytes.decode("utf-8", errors="ignore"))

            item_id = f"{split}_{len(records):08d}_{utt_id}"
            out_rel = None
            if materialize_audio:
                wav_data, sr_read = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
                norm = _normalize_audio(np.asarray(wav_data), int(sr_read), 16000)
                out_wav = audio_dir / f"{item_id}.wav"
                sf.write(out_wav, norm, 16000)
                out_rel = f"{out_root.as_posix()}/audio/{item_id}.wav"

            speaker_id, session_id = _extract_speaker_session(utt_id)
            record = {
                "item_id": item_id,
                "utt_id": utt_id,
                "split": split,
                "source": "snapshot_fallback",
                "audio_path": out_rel,
                "sampling_rate": orig_sr,
                "duration_sec": duration_sec,
                "transcript": transcript,
                "speaker_id": speaker_id,
                "session_id": session_id,
            }
            records.append(record)
            total_sec += duration_sec

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CS-Dialogue subset by duration budget")
    parser.add_argument("--out_dir", default="data/raw/cs_dialogue")
    parser.add_argument("--subset", default="long_wav")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_minutes", type=float, default=60.0)
    parser.add_argument("--max_items", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="BAAI/CS-Dialogue")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--snapshot_dir", default="data/raw/cs_dialogue/hf_snapshot")
    parser.add_argument("--exclude_manifest", default=None)
    parser.add_argument("--exclude_utt_ids_file", default=None)
    parser.add_argument("--materialize_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_output", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_root = Path(args.out_dir) / "long_wav"
    audio_dir = out_root / "audio"
    manifest_path = out_root / "manifest.jsonl"
    stats_path = out_root / "stats.json"
    out_root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for p in audio_dir.glob("*.wav"):
            p.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)
        print(f"Cleaned previous outputs at {out_root}")

    print(
        "Prep config -> "
        f"subset={args.subset}, split={args.split}, max_minutes={args.max_minutes}, "
        f"max_items={args.max_items}, seed={args.seed}, materialize_audio={args.materialize_audio}"
    )

    records: list[dict[str, Any]] = []
    excluded_utt_ids = _load_excluded_utt_ids(
        manifest_path=Path(args.exclude_manifest) if args.exclude_manifest else None,
        utt_ids_file=Path(args.exclude_utt_ids_file) if args.exclude_utt_ids_file else None,
    )
    if excluded_utt_ids:
        print(f"Loaded {len(excluded_utt_ids)} excluded utt_id values.")

    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

        configs = []
        try:
            configs = get_dataset_config_names(args.dataset_name)
            print(f"Available configs: {configs}")
        except Exception:
            pass
        try:
            splits = get_dataset_split_names(args.dataset_name, args.dataset_config)
            print(f"Available splits: {splits}")
        except Exception:
            pass

        chosen_config = args.dataset_config
        if args.subset in configs:
            chosen_config = args.subset
            print("Using requested subset/config long_wav from HF dataset")
        elif chosen_config is None and configs:
            chosen_config = configs[0]
            print(f"Requested subset '{args.subset}' unavailable in HF configs; using '{chosen_config}'")

        print(f"Trying HF load: name={args.dataset_name}, config={chosen_config}, split={args.split}")
        ds = load_dataset(args.dataset_name, chosen_config, split=args.split)
        print(f"HF load succeeded: rows={len(ds)}, columns={ds.column_names}")

        records = _select_hf_duration_budget(
            ds=ds,
            split=args.split,
            seed=args.seed,
            max_minutes=args.max_minutes,
            max_items=args.max_items,
            materialize_audio=args.materialize_audio,
            out_root=out_root,
            audio_dir=audio_dir,
            excluded_utt_ids=excluded_utt_ids,
        )
    except Exception as hf_err:
        print(f"HF load path failed: {hf_err}")
        records = _select_snapshot_duration_budget(
            snapshot_dir=Path(args.snapshot_dir),
            split=args.split,
            seed=args.seed,
            max_minutes=args.max_minutes,
            max_items=args.max_items,
            materialize_audio=args.materialize_audio,
            out_root=out_root,
            audio_dir=audio_dir,
            excluded_utt_ids=excluded_utt_ids,
        )

    if not records:
        raise RuntimeError("No records selected under current budget and inputs")

    _write_manifest_and_stats(
        manifest_path=manifest_path,
        stats_path=stats_path,
        records=records,
        max_minutes=args.max_minutes,
        max_items=args.max_items,
    )

    print(f"Wrote {len(records)} records to {manifest_path}")
    print(f"Wrote stats to {stats_path}")
    print(f"First line preview: {manifest_path.open('r', encoding='utf-8').readline().strip()}")


if __name__ == "__main__":
    main()
