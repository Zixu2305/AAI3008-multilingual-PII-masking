from __future__ import annotations

import os
import re
from typing import Any


BIO_PREFIX_RE = re.compile(r"^[BIES]-", flags=re.IGNORECASE)


def normalize_entity_label(raw_label: str | None) -> str:
    if not raw_label:
        return ""
    return BIO_PREFIX_RE.sub("", str(raw_label)).strip().upper()


def _normalize_label_map(label_map: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (label_map or {}).items():
        nk = normalize_entity_label(k)
        nv = normalize_entity_label(v)
        if nk and nv:
            out[nk] = nv
    return out


def resolve_hf_device(device_cfg: Any) -> int:
    if isinstance(device_cfg, int):
        return device_cfg

    device_s = str(device_cfg or "cpu").strip().lower()
    if device_s in {"cpu", "-1"}:
        return -1
    if device_s.isdigit():
        return int(device_s)

    if device_s.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return -1
            if ":" in device_s:
                _, idx_s = device_s.split(":", 1)
                if idx_s.isdigit():
                    return int(idx_s)
            return 0
        except Exception:
            return -1

    return -1


def _resolve_hf_token(token_env: str | None = None) -> str | None:
    env_names = [token_env, "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    for env_name in env_names:
        if not env_name:
            continue
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            return token
    return None


def _model_candidates(primary: str, fallback: str | None) -> list[str]:
    candidates: list[str] = []
    for value in (primary, fallback):
        name = str(value or "").strip()
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _format_tokenizer_error(exc: Exception) -> str:
    err_s = str(exc).lower()
    if "tiktoken" in err_s or "sentencepiece" in err_s:
        return (
            "Failed to load NER tokenizer. Install optional dependencies and retry: "
            "`pip install tiktoken sentencepiece`."
        )
    return str(exc)


def load_ner_pipeline(
    model_name: str,
    device_cfg: Any,
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    hf_token_env: str | None = None,
    fallback_model_name: str | None = None,
) -> Any:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for NER runs. Install requirements or run inside the Docker app container."
        ) from exc

    device = resolve_hf_device(device_cfg)
    token = _resolve_hf_token(hf_token_env)
    load_kwargs: dict[str, Any] = {
        "local_files_only": bool(local_files_only),
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if revision:
        load_kwargs["revision"] = revision
    if token:
        load_kwargs["token"] = token

    errors: list[str] = []
    for candidate in _model_candidates(model_name, fallback_model_name):
        try:
            model = AutoModelForTokenClassification.from_pretrained(candidate, **load_kwargs)

            # Some tokenizer files require optional runtime deps (e.g. tiktoken/sentencepiece).
            # Keep fast tokenizer path, but provide a clear dependency hint when that is the root cause.
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True, **load_kwargs)
            except Exception as exc:
                raise RuntimeError(_format_tokenizer_error(exc)) from exc

            ner_pipe = pipeline(
                task="token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )
            setattr(ner_pipe, "_requested_model_name", model_name)
            setattr(ner_pipe, "_resolved_model_name", candidate)
            return ner_pipe
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to load NER model. Tried:\n{joined}")


def predict_ner_spans(
    text: str,
    ner_pipeline: Any,
    label_map: dict[str, str] | None = None,
    allowed_labels: set[str] | None = None,
    max_length: int | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    if not text.strip():
        return []

    normalized_map = _normalize_label_map(label_map)
    allowed_norm = {normalize_entity_label(x) for x in (allowed_labels or set()) if normalize_entity_label(x)}

    # Keep call signature compatible with older/newer transformers pipeline versions.
    # Some versions reject truncation/max_length kwargs on token-classification.
    _ = max_length
    raw_entities = ner_pipeline(text)
    spans: list[dict[str, Any]] = []
    for ent in raw_entities:
        raw_label = ent.get("entity_group") or ent.get("entity")
        label = normalize_entity_label(raw_label)
        label = normalize_entity_label(normalized_map.get(label, label))
        if not label:
            continue
        if allowed_norm and label not in allowed_norm:
            continue

        start = ent.get("start")
        end = ent.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end <= start:
            continue

        score = ent.get("score")
        if score_threshold is not None and isinstance(score, (int, float)) and float(score) < float(score_threshold):
            continue

        spans.append(
            {
                "start": start,
                "end": end,
                "type": label,
                "text": text[start:end],
                "score": float(score) if isinstance(score, (int, float)) else None,
                "source": "ner",
            }
        )

    spans.sort(key=lambda s: (int(s["start"]), int(s["end"]), str(s["type"])))
    return spans
