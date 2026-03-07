"""Span-level evaluation of PII detection against gold-set annotations."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.artifacts import artifact_dir
from src.utils.io import read_jsonl, write_json, write_jsonl


def _prf_scores(tp: int, fp: int, fn: int, beta: float = 2.0) -> dict[str, float]:
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    b2 = beta * beta
    fbeta = ((1 + b2) * precision * recall / (b2 * precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        f"f{beta:.0f}": round(fbeta, 4),
    }


def _span_overlap_ratio(pred_start: int, pred_end: int, gold_start: int, gold_end: int) -> float:
    """Compute character-level overlap ratio (intersection / union)."""
    overlap_start = max(pred_start, gold_start)
    overlap_end = min(pred_end, gold_end)
    if overlap_start >= overlap_end:
        return 0.0
    intersection = overlap_end - overlap_start
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    return intersection / union if union > 0 else 0.0


def _text_containment(pred_text: str, gold_text: str) -> bool:
    """Check if one text contains the other (after stripping whitespace)."""
    p = pred_text.strip()
    g = gold_text.strip()
    if not p or not g:
        return False
    return p in g or g in p


def _match_spans(
    pred_spans: list[dict],
    gold_entities: list[dict],
    overlap_threshold: float = 0.5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Match predicted spans to gold entities.

    Returns (tp_pairs, fp_spans, fn_entities).
    Match criteria: same pii_type AND (overlap >= threshold OR text containment).
    """
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    tp_pairs: list[dict] = []

    for gi, gold in enumerate(gold_entities):
        g_start = gold.get("start_char")
        g_end = gold.get("end_char")
        g_type = gold.get("pii_type", "").upper()
        g_text = gold.get("text", "")

        best_pi = -1
        best_overlap = 0.0

        for pi, pred in enumerate(pred_spans):
            if pi in matched_pred:
                continue
            p_type = pred.get("type", "").upper()
            if p_type != g_type:
                continue

            p_start = pred.get("start")
            p_end = pred.get("end")
            p_text = pred.get("text", "")

            # Check overlap or text containment
            matched = False
            if (
                isinstance(g_start, int)
                and isinstance(g_end, int)
                and isinstance(p_start, int)
                and isinstance(p_end, int)
            ):
                overlap = _span_overlap_ratio(p_start, p_end, g_start, g_end)
                if overlap >= overlap_threshold:
                    matched = True
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_pi = pi

            if not matched and _text_containment(p_text, g_text):
                if best_pi == -1:
                    best_pi = pi
                    best_overlap = 0.5  # nominal

        if best_pi >= 0:
            matched_gold.add(gi)
            matched_pred.add(best_pi)
            tp_pairs.append({
                "gold": gold,
                "pred": pred_spans[best_pi],
                "overlap": best_overlap,
            })

    fp_spans = [pred_spans[i] for i in range(len(pred_spans)) if i not in matched_pred]
    fn_entities = [gold_entities[i] for i in range(len(gold_entities)) if i not in matched_gold]

    return tp_pairs, fp_spans, fn_entities


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def evaluate_gold(
    pred_records: list[dict],
    gold_records: list[dict],
    beta: float = 2.0,
    overlap_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate predicted PII spans against gold-set annotations.

    Args:
        pred_records: list of dicts with keys: record_id, spans (list of detected PII)
        gold_records: list of dicts from gold set JSONL
        beta: F-beta weight (default 2.0 for recall-heavy)
        overlap_threshold: minimum overlap ratio for span match

    Returns:
        dict with overall, per_type, per_lang, per_record results
    """
    # Index gold records by record_id
    gold_by_id = {r["record_id"]: r for r in gold_records}

    per_record_results: list[dict] = []
    error_records: list[dict] = []
    type_counts: dict[str, Counter] = defaultdict(Counter)
    lang_counts: dict[str, Counter] = defaultdict(Counter)
    total_counts: Counter = Counter()

    for pred_rec in pred_records:
        rid = pred_rec["record_id"]
        gold_rec = gold_by_id.get(rid)
        if gold_rec is None:
            continue

        pred_spans = pred_rec.get("spans", [])
        gold_entities = gold_rec.get("entities", [])
        lang = gold_rec.get("language", "unknown")

        tp_pairs, fp_spans, fn_entities = _match_spans(
            pred_spans, gold_entities, overlap_threshold
        )

        tp = len(tp_pairs)
        fp = len(fp_spans)
        fn = len(fn_entities)

        total_counts["tp"] += tp
        total_counts["fp"] += fp
        total_counts["fn"] += fn

        lang_counts[lang]["tp"] += tp
        lang_counts[lang]["fp"] += fp
        lang_counts[lang]["fn"] += fn

        # Per-type counts
        for pair in tp_pairs:
            pii_type = pair["gold"].get("pii_type", "UNKNOWN")
            type_counts[pii_type]["tp"] += 1
        for span in fp_spans:
            pii_type = span.get("type", "UNKNOWN")
            type_counts[pii_type]["fp"] += 1
        for ent in fn_entities:
            pii_type = ent.get("pii_type", "UNKNOWN")
            type_counts[pii_type]["fn"] += 1

        rec_metrics = _prf_scores(tp, fp, fn, beta)
        rec_result = {
            "record_id": rid,
            "language": lang,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_count": len(gold_entities),
            "pred_count": len(pred_spans),
            **rec_metrics,
        }
        per_record_results.append(rec_result)

        # Track errors
        if fp > 0 or fn > 0:
            error_records.append({
                "record_id": rid,
                "language": lang,
                "transcript_preview": gold_rec.get("transcript", "")[:200],
                "false_positives": [
                    {"type": s.get("type"), "text": s.get("text"), "source": s.get("source")}
                    for s in fp_spans
                ],
                "false_negatives": [
                    {"type": e.get("pii_type"), "text": e.get("text")}
                    for e in fn_entities
                ],
            })

    # Aggregate metrics
    overall = _prf_scores(total_counts["tp"], total_counts["fp"], total_counts["fn"], beta)
    overall["tp"] = total_counts["tp"]
    overall["fp"] = total_counts["fp"]
    overall["fn"] = total_counts["fn"]
    overall["num_records"] = len(per_record_results)

    per_type = {}
    for pii_type in sorted(type_counts):
        c = type_counts[pii_type]
        metrics = _prf_scores(c["tp"], c["fp"], c["fn"], beta)
        per_type[pii_type] = {
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "support": c["tp"] + c["fn"],
            **metrics,
        }

    per_lang = {}
    for lang in sorted(lang_counts):
        c = lang_counts[lang]
        metrics = _prf_scores(c["tp"], c["fp"], c["fn"], beta)
        per_lang[lang] = {
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            **metrics,
        }

    return {
        "overall": overall,
        "per_type": per_type,
        "per_lang": per_lang,
        "per_record": per_record_results,
        "errors": error_records,
    }


def run_pii_eval_gold(cfg: dict) -> Path:
    """Run gold-set evaluation as a pipeline stage."""
    run_id = cfg.get("run", {}).get("run_id", "pii_eval_gold_v1")
    out_dir = artifact_dir("pii_eval", run_id)

    beta = float(cfg.get("protocol", {}).get("beta", 2.0))
    overlap_threshold = float(cfg.get("protocol", {}).get("overlap_threshold", 0.5))

    # Load gold records
    gold_path = cfg.get("inputs", {}).get("gold_path", "data/prepared/gold_set/dev.jsonl")
    from src.utils.io import resolve_path
    gold_records = read_jsonl(resolve_path(gold_path))

    # Load predicted spans
    pred_path = cfg.get("inputs", {}).get("pred_path")
    if pred_path:
        pred_records = read_jsonl(resolve_path(pred_path))
    else:
        # Run the PII pipeline on gold transcripts
        pred_records = _run_pii_on_gold(gold_records, cfg)

    results = evaluate_gold(
        pred_records=pred_records,
        gold_records=gold_records,
        beta=beta,
        overlap_threshold=overlap_threshold,
    )

    # Write outputs
    write_json(out_dir / "summary.json", {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "version": "pii_gold_eval_v1",
            "scheme": "span_overlap",
            "beta": beta,
            "overlap_threshold": overlap_threshold,
        },
        "overall": results["overall"],
        "per_type": results["per_type"],
        "per_lang": results["per_lang"],
    })

    write_jsonl(out_dir / "per_record.jsonl", results["per_record"])
    write_jsonl(out_dir / "errors.jsonl", results["errors"])

    # Per-type CSV
    type_headers = ["pii_type", "support", "tp", "fp", "fn", "precision", "recall", "f1", f"f{beta:.0f}"]
    type_rows = []
    for pii_type, m in results["per_type"].items():
        type_rows.append([pii_type, m["support"], m["tp"], m["fp"], m["fn"],
                          m["precision"], m["recall"], m["f1"], m.get(f"f{beta:.0f}", 0)])
    _write_csv(out_dir / "per_type.csv", type_headers, type_rows)

    # Per-lang CSV
    lang_headers = ["language", "tp", "fp", "fn", "precision", "recall", "f1", f"f{beta:.0f}"]
    lang_rows = []
    for lang, m in results["per_lang"].items():
        lang_rows.append([lang, m["tp"], m["fp"], m["fn"],
                          m["precision"], m["recall"], m["f1"], m.get(f"f{beta:.0f}", 0)])
    _write_csv(out_dir / "per_lang.csv", lang_headers, lang_rows)

    write_json(out_dir / "run_meta.json", {
        "stage": "pii_eval",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(gold_path),
        "num_records": len(gold_records),
    })

    return out_dir


def _run_pii_on_gold(gold_records: list[dict], cfg: dict) -> list[dict]:
    """Run the current PII pipeline on gold-set transcripts."""
    from src.pii.run import _rule_spans_for_text, _clean_and_merge_spans
    from src.pii.ner import load_ner_pipeline, predict_ner_spans

    pii_cfg = cfg.get("pii", {})
    enable_rules = bool(pii_cfg.get("enable_rules", True))
    enable_ner = bool(pii_cfg.get("enable_ner", False))
    rules_cfg = pii_cfg.get("rules", {})
    ner_cfg = pii_cfg.get("ner", {})
    post_cfg = cfg.get("postprocess", {})
    merge_gap = int(post_cfg.get("merge_gap_chars", 0) or 0)
    min_span = int(post_cfg.get("min_span_chars", 1) or 1)

    enable_llm = bool(pii_cfg.get("enable_llm", False))

    ner_pipe = None
    if enable_ner:
        model_name = str(ner_cfg.get("model_name", "")).strip()
        if model_name:
            ner_pipe = load_ner_pipeline(model_name=model_name, device_cfg=ner_cfg.get("device", "cpu"))

    pred_records = []
    for rec in gold_records:
        text = rec.get("transcript", "")
        spans: list[dict] = []

        if enable_rules:
            spans.extend(_rule_spans_for_text(text, rules_cfg=rules_cfg))

        if ner_pipe is not None:
            spans.extend(predict_ner_spans(
                text=text,
                ner_pipeline=ner_pipe,
                label_map=ner_cfg.get("label_map"),
                max_length=ner_cfg.get("max_length"),
                score_threshold=ner_cfg.get("score_threshold"),
            ))

        if enable_llm:
            try:
                from src.pii.llm_detect import detect_pii_llm
                llm_cfg = pii_cfg.get("llm", {})
                llm_spans = detect_pii_llm(text, llm_cfg)
                spans.extend(llm_spans)
            except ImportError:
                pass

        spans = _clean_and_merge_spans(
            text=text, spans=spans,
            merge_gap_chars=merge_gap, min_span_chars=min_span,
        )

        pred_records.append({
            "record_id": rec["record_id"],
            "transcript": text,
            "spans": spans,
        })

    return pred_records
