#!/usr/bin/env python3
"""Prepare gold-set JSON files into JSONL dev/eval splits for evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _reconstruct_offsets(transcript: str, entities: list[dict]) -> list[dict]:
    """Add start_char/end_char to entities that lack them, via str.find."""
    out = []
    used_positions: set[int] = set()
    for ent in entities:
        e = dict(ent)
        text = e.get("text", "")
        if not text:
            out.append(e)
            continue

        # Already has valid offsets
        if (
            isinstance(e.get("start_char"), int)
            and isinstance(e.get("end_char"), int)
            and e["start_char"] >= 0
            and e["end_char"] > e["start_char"]
        ):
            out.append(e)
            used_positions.add(e["start_char"])
            continue

        # Find the entity text in transcript, avoiding already-used positions
        search_start = 0
        found = False
        while True:
            idx = transcript.find(text, search_start)
            if idx == -1:
                break
            if idx not in used_positions:
                e["start_char"] = idx
                e["end_char"] = idx + len(text)
                used_positions.add(idx)
                found = True
                break
            search_start = idx + 1

        if not found:
            # Fallback: use first occurrence even if position is reused
            idx = transcript.find(text)
            if idx >= 0:
                e["start_char"] = idx
                e["end_char"] = idx + len(text)
            else:
                e["start_char"] = None
                e["end_char"] = None

        out.append(e)
    return out


def _load_gold_file(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gold set into dev/eval JSONL")
    parser.add_argument("--gold_dir", default="gold_set")
    parser.add_argument("--out_dir", default="data/prepared/gold_set")
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for lang_file in ["en.json", "zh.json", "mix.json"]:
        fpath = gold_dir / lang_file
        if not fpath.exists():
            print(f"Warning: {fpath} not found, skipping")
            continue

        records = _load_gold_file(fpath)
        for rec in records:
            rec["entities"] = _reconstruct_offsets(rec["transcript"], rec["entities"])
            all_records.append(rec)

    dev_records = [r for r in all_records if r.get("split") == "dev"]
    eval_records = [r for r in all_records if r.get("split") == "eval"]

    dev_path = out_dir / "dev.jsonl"
    eval_path = out_dir / "eval.jsonl"

    for path, records in [(dev_path, dev_records), (eval_path, eval_records)]:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {path}")

    # Stats
    for split_name, records in [("dev", dev_records), ("eval", eval_records)]:
        langs = {}
        types = {}
        offset_ok = 0
        offset_missing = 0
        for r in records:
            lang = r.get("language", "?")
            langs[lang] = langs.get(lang, 0) + 1
            for e in r.get("entities", []):
                t = e.get("pii_type", "?")
                types[t] = types.get(t, 0) + 1
                if isinstance(e.get("start_char"), int):
                    offset_ok += 1
                else:
                    offset_missing += 1
        print(f"\n{split_name}: {len(records)} records")
        print(f"  Languages: {langs}")
        print(f"  PII types: {types}")
        print(f"  Offsets OK: {offset_ok}, missing: {offset_missing}")


if __name__ == "__main__":
    main()
