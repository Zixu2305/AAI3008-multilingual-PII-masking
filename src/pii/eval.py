from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pii.ner import load_ner_pipeline, normalize_entity_label, predict_ner_spans
from src.utils.artifacts import artifact_dir
from src.utils.io import read_jsonl, resolve_path, write_json, write_jsonl


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _split_bio_tag(tag: str) -> tuple[str, str]:
    raw = (tag or "O").strip().upper()
    if raw == "O" or "-" not in raw:
        return "O", ""
    prefix, entity_type = raw.split("-", 1)
    if prefix not in {"B", "I"}:
        return "O", ""
    if not entity_type:
        return "O", ""
    return prefix, entity_type


def _normalize_tags(tags: list[str], allowed_types: set[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        prefix, entity_type = _split_bio_tag(tag)
        if prefix == "O" or entity_type not in allowed_types:
            out.append("O")
        else:
            out.append(f"{prefix}-{entity_type}")
    return out


def _extract_entities(tags: list[str]) -> list[tuple[str, int, int]]:
    entities: list[tuple[str, int, int]] = []
    i = 0
    while i < len(tags):
        prefix, entity_type = _split_bio_tag(tags[i])
        if prefix == "O":
            i += 1
            continue

        start = i
        i += 1
        while i < len(tags):
            nxt_prefix, nxt_type = _split_bio_tag(tags[i])
            if nxt_prefix == "I" and nxt_type == entity_type:
                i += 1
                continue
            break
        entities.append((entity_type, start, i))
    return entities


def _prf_scores(tp: int, fp: int, fn: int, beta: float) -> dict[str, float]:
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    b2 = beta * beta
    fbeta = ((1 + b2) * precision * recall / (b2 * precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fbeta": float(fbeta),
    }


def _tokens_to_text_and_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            pos += 1
        start = pos
        parts.append(tok)
        pos += len(tok)
        offsets.append((start, pos))
    return "".join(parts), offsets


def _spans_to_bio_tags(
    spans: list[dict[str, Any]],
    token_offsets: list[tuple[int, int]],
    allowed_types: set[str],
) -> list[str]:
    pred_tags = ["O"] * len(token_offsets)
    ranked = sorted(
        spans,
        key=lambda s: (
            float(-(s.get("score") or 0.0)),
            int(s.get("start", 0)),
            int(-(int(s.get("end", 0)) - int(s.get("start", 0)))),
        ),
    )
    for span in ranked:
        entity_type = normalize_entity_label(str(span.get("type", "")))
        if entity_type not in allowed_types:
            continue

        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        if start < 0 or end <= start:
            continue

        overlap = [
            idx
            for idx, (tok_start, tok_end) in enumerate(token_offsets)
            if tok_end > start and tok_start < end
        ]
        if not overlap:
            continue
        if any(pred_tags[idx] != "O" for idx in overlap):
            continue

        pred_tags[overlap[0]] = f"B-{entity_type}"
        for idx in overlap[1:]:
            pred_tags[idx] = f"I-{entity_type}"
    return pred_tags


def _evaluate_split(
    split_name: str,
    data_path: Path,
    ner_pipe: Any,
    allowed_types: set[str],
    label_aliases: dict[str, str],
    max_length: int | None,
    score_threshold: float | None,
    beta: float,
    max_items: int,
) -> dict[str, Any]:
    rows = read_jsonl(data_path)
    if max_items > 0:
        rows = rows[:max_items]

    per_item_rows: list[dict[str, Any]] = []
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: Counter[str] = Counter()
    token_total = 0
    token_correct = 0
    split_lang = None

    for rec in rows:
        tokens = [str(t) for t in rec.get("tokens", [])]
        gold_tags = [str(t) for t in rec.get("gold_tag_names", [])]
        if not tokens:
            continue

        if split_lang is None:
            split_lang = rec.get("lang")

        if len(gold_tags) != len(tokens):
            gold_tags = (gold_tags + ["O"] * len(tokens))[: len(tokens)]

        gold_tags = _normalize_tags(gold_tags, allowed_types=allowed_types)
        text, offsets = _tokens_to_text_and_offsets(tokens)
        pred_spans = predict_ner_spans(
            text=text,
            ner_pipeline=ner_pipe,
            label_map=label_aliases,
            allowed_labels=allowed_types,
            max_length=max_length,
            score_threshold=score_threshold,
        )
        pred_tags = _spans_to_bio_tags(
            spans=pred_spans,
            token_offsets=offsets,
            allowed_types=allowed_types,
        )

        token_total += len(tokens)
        token_correct += sum(int(g == p) for g, p in zip(gold_tags, pred_tags))

        gold_entities = set(_extract_entities(gold_tags))
        pred_entities = set(_extract_entities(pred_tags))
        tp_set = gold_entities & pred_entities
        fp_set = pred_entities - gold_entities
        fn_set = gold_entities - pred_entities

        split_counts["tp"] += len(tp_set)
        split_counts["fp"] += len(fp_set)
        split_counts["fn"] += len(fn_set)

        for entity_type, _, _ in tp_set:
            type_counts[entity_type]["tp"] += 1
        for entity_type, _, _ in fp_set:
            type_counts[entity_type]["fp"] += 1
        for entity_type, _, _ in fn_set:
            type_counts[entity_type]["fn"] += 1

        per_item_rows.append(
            {
                "split": split_name,
                "item_id": rec.get("item_id"),
                "lang": rec.get("lang"),
                "token_count": len(tokens),
                "gold_entities": len(gold_entities),
                "pred_entities": len(pred_entities),
                "tp": len(tp_set),
                "fp": len(fp_set),
                "fn": len(fn_set),
            }
        )

    split_metrics = _prf_scores(
        tp=int(split_counts["tp"]),
        fp=int(split_counts["fp"]),
        fn=int(split_counts["fn"]),
        beta=beta,
    )
    split_summary = {
        "split": split_name,
        "lang": split_lang,
        "data_path": str(data_path),
        "num_sentences": len(per_item_rows),
        "token_total": int(token_total),
        "token_correct": int(token_correct),
        "token_accuracy": float(token_correct / token_total) if token_total else 0.0,
        "tp": int(split_counts["tp"]),
        "fp": int(split_counts["fp"]),
        "fn": int(split_counts["fn"]),
        **split_metrics,
    }

    per_type_rows = []
    for entity_type in sorted(type_counts):
        counts = type_counts[entity_type]
        metrics = _prf_scores(
            tp=int(counts["tp"]),
            fp=int(counts["fp"]),
            fn=int(counts["fn"]),
            beta=beta,
        )
        per_type_rows.append(
            {
                "split": split_name,
                "entity_type": entity_type,
                "tp": int(counts["tp"]),
                "fp": int(counts["fp"]),
                "fn": int(counts["fn"]),
                "support": int(counts["tp"] + counts["fn"]),
                **metrics,
            }
        )

    return {
        "summary": split_summary,
        "per_type_rows": per_type_rows,
        "per_item_rows": per_item_rows,
    }


def run_pii_eval(cfg: dict) -> Path:
    run_id = cfg.get("run", {}).get("run_id", "pii_eval_wikiann_v1")
    out_dir = artifact_dir("pii_eval", run_id)
    eval_entries = cfg.get("inputs", {}).get("evaluations", [])
    if not eval_entries:
        raise ValueError("inputs.evaluations is required")

    pii_cfg = cfg.get("pii", {})
    ner_cfg = pii_cfg.get("ner", {})
    model_name = str(ner_cfg.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("pii.ner.model_name is required")

    beta = float(cfg.get("protocol", {}).get("beta", 1.0))
    max_items = int(cfg.get("protocol", {}).get("max_items_per_split", 0) or 0)
    max_length = ner_cfg.get("max_length")
    score_threshold = ner_cfg.get("score_threshold")
    allowed_types = {
        normalize_entity_label(x)
        for x in cfg.get("protocol", {}).get("allowed_entity_types", ["PER", "ORG", "LOC"])
    }
    allowed_types = {x for x in allowed_types if x}
    if not allowed_types:
        raise ValueError("protocol.allowed_entity_types must contain at least one entity type")

    label_aliases_cfg = cfg.get("protocol", {}).get("label_aliases", {})
    label_aliases = {
        normalize_entity_label(k): normalize_entity_label(v)
        for k, v in label_aliases_cfg.items()
        if normalize_entity_label(k) and normalize_entity_label(v)
    }

    ner_pipe = load_ner_pipeline(model_name=model_name, device_cfg=ner_cfg.get("device", "cpu"))

    split_summaries: list[dict[str, Any]] = []
    per_type_rows: list[dict[str, Any]] = []
    per_item_rows: list[dict[str, Any]] = []

    for ent in eval_entries:
        split_name = str(ent["name"])
        data_path = resolve_path(ent["data_path"])
        split_result = _evaluate_split(
            split_name=split_name,
            data_path=data_path,
            ner_pipe=ner_pipe,
            allowed_types=allowed_types,
            label_aliases=label_aliases,
            max_length=max_length,
            score_threshold=score_threshold,
            beta=beta,
            max_items=max_items,
        )
        split_summaries.append(split_result["summary"])
        per_type_rows.extend(split_result["per_type_rows"])
        per_item_rows.extend(split_result["per_item_rows"])

    total_tp = int(sum(s["tp"] for s in split_summaries))
    total_fp = int(sum(s["fp"] for s in split_summaries))
    total_fn = int(sum(s["fn"] for s in split_summaries))
    weighted_token_total = int(sum(s["token_total"] for s in split_summaries))
    weighted_token_correct = int(sum(s["token_correct"] for s in split_summaries))

    overall_metrics = _prf_scores(tp=total_tp, fp=total_fp, fn=total_fn, beta=beta)
    overall = {
        "num_splits": len(split_summaries),
        "num_sentences": len(per_item_rows),
        "token_accuracy": float(weighted_token_correct / weighted_token_total) if weighted_token_total else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        **overall_metrics,
    }

    protocol = {
        "version": cfg.get("protocol", {}).get("version", "pii_ner_eval_v1"),
        "scheme": "entity_level_strict_bio",
        "beta": beta,
        "allowed_entity_types": sorted(allowed_types),
        "label_aliases": label_aliases,
        "max_items_per_split": max_items,
    }

    write_jsonl(out_dir / "per_item.jsonl", per_item_rows)
    write_json(out_dir / "protocol.json", protocol)
    write_json(
        out_dir / "summary.json",
        {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol["version"],
            "splits": split_summaries,
            "overall": overall,
        },
    )

    _write_csv(
        out_dir / "per_split.csv",
        [
            "split",
            "lang",
            "num_sentences",
            "token_accuracy",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "fbeta",
            "beta",
        ],
        [
            [
                s["split"],
                s.get("lang"),
                s["num_sentences"],
                s["token_accuracy"],
                s["tp"],
                s["fp"],
                s["fn"],
                s["precision"],
                s["recall"],
                s["f1"],
                s["fbeta"],
                beta,
            ]
            for s in split_summaries
        ],
    )

    _write_csv(
        out_dir / "per_type.csv",
        [
            "split",
            "entity_type",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "fbeta",
            "beta",
        ],
        [
            [
                r["split"],
                r["entity_type"],
                r["support"],
                r["tp"],
                r["fp"],
                r["fn"],
                r["precision"],
                r["recall"],
                r["f1"],
                r["fbeta"],
                beta,
            ]
            for r in per_type_rows
        ],
    )

    write_json(
        out_dir / "run_meta.json",
        {
            "stage": "pii_eval",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_splits": len(split_summaries),
            "num_sentences": len(per_item_rows),
            "inputs": eval_entries,
            "model_name": model_name,
        },
    )
    return out_dir
