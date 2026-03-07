#!/usr/bin/env python3
"""Convert gold-set char-offset entities to subword BIO tags for NER fine-tuning."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


# 11 BIO labels: O + B/I for NAME, PHONE, EMAIL, ADDRESS, ID
LABEL_LIST = [
    "O",
    "B-NAME", "I-NAME",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-ADDRESS", "I-ADDRESS",
    "B-ID", "I-ID",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def _tokenize_and_align(
    text: str,
    entities: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any] | None:
    """Tokenize text and create aligned BIO tags from char-level entity offsets."""
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )

    offset_mapping = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    # Sort entities by start position
    sorted_entities = sorted(
        [e for e in entities if isinstance(e.get("start_char"), int) and isinstance(e.get("end_char"), int)],
        key=lambda e: (e["start_char"], e["end_char"]),
    )

    # Create labels for each token
    labels = [-100] * len(input_ids)  # -100 = ignore (for special tokens)

    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == 0 and tok_end == 0:
            # Special token ([CLS], [SEP], <s>, </s>)
            labels[i] = -100
            continue

        labels[i] = LABEL_TO_ID["O"]  # Default: O

        for ent in sorted_entities:
            ent_start = ent["start_char"]
            ent_end = ent["end_char"]
            pii_type = ent["pii_type"]

            if tok_start >= ent_end or tok_end <= ent_start:
                continue

            # Token overlaps with entity
            if tok_start <= ent_start:
                # First token of entity -> B-
                b_label = f"B-{pii_type}"
            else:
                # Continuation -> I-
                b_label = f"I-{pii_type}"

            if b_label in LABEL_TO_ID:
                labels[i] = LABEL_TO_ID[b_label]
            break  # First matching entity wins

    return {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "offset_mapping": offset_mapping,
    }


def _create_cv_splits(records: list[dict], n_folds: int = 5, seed: int = 42) -> list[list[dict]]:
    """Create n_folds cross-validation splits."""
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, idx in enumerate(indices):
        folds[i % n_folds].append(idx)

    splits = []
    for fold_idx in range(n_folds):
        val_indices = set(folds[fold_idx])
        train_recs = [records[i] for i in range(len(records)) if i not in val_indices]
        val_recs = [records[i] for i in folds[fold_idx]]
        splits.append({
            "fold": fold_idx,
            "train": train_recs,
            "val": val_recs,
        })

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert gold set to BIO format for NER fine-tuning")
    parser.add_argument("--gold_path", default="data/prepared/gold_set/dev.jsonl")
    parser.add_argument("--out_dir", default="data/prepared/gold_bio")
    parser.add_argument("--tokenizer", default="xlm-roberta-base")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers required. pip install transformers")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load gold records
    gold_path = Path(args.gold_path)
    records = []
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} gold records from {gold_path}")

    # Convert each record to BIO format
    bio_records = []
    skipped = 0
    for rec in records:
        text = rec.get("transcript", "")
        entities = rec.get("entities", [])

        result = _tokenize_and_align(text, entities, tokenizer)
        if result is None:
            skipped += 1
            continue

        bio_rec = {
            "record_id": rec["record_id"],
            "language": rec.get("language"),
            "text": text,
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "labels": result["labels"],
        }
        bio_records.append(bio_rec)

    print(f"Converted {len(bio_records)} records ({skipped} skipped)")

    # Write all records
    all_path = out_dir / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as f:
        for rec in bio_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {all_path}")

    # Write CV splits
    splits = _create_cv_splits(bio_records, n_folds=args.n_folds)
    for split_info in splits:
        fold = split_info["fold"]
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "val"):
            split_recs = split_info[split_name]
            split_path = fold_dir / f"{split_name}.jsonl"
            with split_path.open("w", encoding="utf-8") as f:
                for rec in split_recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Fold {fold} {split_name}: {len(split_recs)} records -> {split_path}")

    # Write label map
    label_map = {
        "label_list": LABEL_LIST,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
    }
    label_map_path = out_dir / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Wrote {label_map_path}")


if __name__ == "__main__":
    main()
