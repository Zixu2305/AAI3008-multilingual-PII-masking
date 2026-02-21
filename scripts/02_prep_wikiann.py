#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from datasets import load_dataset


def _export_lang(lang: str, out_dir: Path, clean_output: bool) -> None:
    lang_dir = out_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["train", "validation", "test"]
    split_files = [lang_dir / f"{s}.jsonl" for s in split_names]
    legacy_samples_path = lang_dir / "samples.jsonl"
    label_map_path = lang_dir / "label_map.json"
    stats_path = lang_dir / "stats.json"

    if clean_output:
        for p in split_files + [legacy_samples_path, label_map_path, stats_path]:
            p.unlink(missing_ok=True)

    per_split_stats: dict[str, dict] = {}
    aggregate_counter: Counter[str] = Counter()
    total_sentences = 0

    names: list[str] | None = None

    for split in split_names:
        print(f"Loading wikiann/{lang} split={split}...")
        ds = load_dataset("wikiann", lang, split=split)

        if names is None:
            ner_feature = ds.features["ner_tags"].feature
            names = list(getattr(ner_feature, "names", []))

        out_path = lang_dir / f"{split}.jsonl"
        token_lengths: list[int] = []
        split_counter: Counter[str] = Counter()

        with out_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(ds):
                tokens = [str(t) for t in row.get("tokens", [])]
                gold_tags = [int(t) for t in row.get("ner_tags", [])]
                gold_tag_names = [names[t] if 0 <= t < len(names) else "UNKNOWN" for t in gold_tags]

                token_lengths.append(len(tokens))
                split_counter.update(gold_tag_names)
                aggregate_counter.update(gold_tag_names)

                rec = {
                    "item_id": f"{lang}_{split}_{idx:08d}",
                    "lang": lang,
                    "split": split,
                    "tokens": tokens,
                    "gold_tags": gold_tags,
                    "gold_tag_names": gold_tag_names,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        line_count = sum(1 for _ in out_path.open("r", encoding="utf-8"))
        if line_count == 0:
            raise RuntimeError(f"No samples written to {out_path}")

        first_line = out_path.open("r", encoding="utf-8").readline().strip()
        print(f"Wrote {line_count} samples to {out_path}")
        print(f"First line preview: {first_line}")

        per_split_stats[split] = {
            "num_sentences": line_count,
            "avg_tokens_per_sentence": float(mean(token_lengths)) if token_lengths else 0.0,
            "entity_distribution": dict(sorted(split_counter.items())),
        }
        total_sentences += line_count

    assert names is not None
    label_map = {
        "dataset": "wikiann",
        "lang": lang,
        "idx_to_name": {str(i): n for i, n in enumerate(names)},
        "name_to_idx": {n: i for i, n in enumerate(names)},
    }
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "dataset": "wikiann",
        "lang": lang,
        "total_sentences": total_sentences,
        "splits": per_split_stats,
        "entity_distribution_all": dict(sorted(aggregate_counter.items())),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {label_map_path}")
    print(f"Wrote: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full WikiAnn EN+ZH splits")
    parser.add_argument("--out_dir", default="data/raw/wikiann")
    parser.add_argument("--clean_output", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in ("en", "zh"):
        _export_lang(lang=lang, out_dir=out_dir, clean_output=args.clean_output)


if __name__ == "__main__":
    main()
