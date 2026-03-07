"""LLM-based PII detection as a safety net for rule+NER pipeline."""
from __future__ import annotations

import json
import re
from typing import Any

_SYSTEM_PROMPT = """\
You are a PII detection system. Extract all personally identifiable information from the given text.

Return a JSON array of objects, each with:
- "pii_type": one of "NAME", "PHONE", "EMAIL", "ADDRESS", "ID"
- "text": the exact text span from the input

Rules:
- NAME: full names of people (e.g., "Daniel Tan", "李梦星")
- PHONE: phone numbers in any format, including Chinese numerals (e.g., "9123 4567", "九三四五，六六七七")
- EMAIL: email addresses
- ADDRESS: physical addresses, street names, block numbers (e.g., "Block 789, Green Valley Road", "大牌三零一")
- ID: identification numbers, reference numbers, employee IDs

Return ONLY valid JSON array. If no PII found, return [].
"""


def detect_pii_llm(text: str, llm_cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Detect PII using an LLM API call.

    Args:
        text: input text to scan
        llm_cfg: config dict with keys: provider, model, api_key, max_tokens

    Returns:
        list of span dicts with keys: start, end, type, text, source
    """
    if not text.strip():
        return []

    cfg = llm_cfg or {}
    provider = cfg.get("provider", "anthropic")
    model = cfg.get("model", "claude-sonnet-4-20250514")
    max_tokens = int(cfg.get("max_tokens", 2048))

    try:
        if provider == "anthropic":
            return _call_anthropic(text, model, max_tokens)
        elif provider == "openai":
            return _call_openai(text, model, max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    except Exception as e:
        print(f"LLM PII detection failed: {e}")
        return []


def _call_anthropic(text: str, model: str, max_tokens: int) -> list[dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )

    content = response.content[0].text
    return _parse_llm_response(text, content)


def _call_openai(text: str, model: str, max_tokens: int) -> list[dict[str, Any]]:
    import openai

    client = openai.OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )

    content = response.choices[0].message.content
    return _parse_llm_response(text, content)


def _parse_llm_response(original_text: str, response_text: str) -> list[dict[str, Any]]:
    """Parse LLM response and locate spans in original text."""
    # Extract JSON array from response
    try:
        # Try direct parse
        entities = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON array in response
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if match:
            try:
                entities = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(entities, list):
        return []

    spans = []
    used_positions: set[int] = set()

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        pii_type = str(ent.get("pii_type", "")).upper()
        ent_text = str(ent.get("text", "")).strip()

        if not pii_type or not ent_text:
            continue
        if pii_type not in {"NAME", "PHONE", "EMAIL", "ADDRESS", "ID"}:
            continue

        # Find the entity text in the original text
        search_start = 0
        found = False
        while True:
            idx = original_text.find(ent_text, search_start)
            if idx == -1:
                break
            if idx not in used_positions:
                spans.append({
                    "start": idx,
                    "end": idx + len(ent_text),
                    "type": pii_type,
                    "text": ent_text,
                    "source": "llm",
                })
                used_positions.add(idx)
                found = True
                break
            search_start = idx + 1

        if not found:
            # Still add with first occurrence
            idx = original_text.find(ent_text)
            if idx >= 0:
                spans.append({
                    "start": idx,
                    "end": idx + len(ent_text),
                    "type": pii_type,
                    "text": ent_text,
                    "source": "llm",
                })

    return spans
