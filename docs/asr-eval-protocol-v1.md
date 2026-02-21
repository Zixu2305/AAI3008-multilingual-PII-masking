# ASR Evaluation Protocol v1 (Reusable Baseline Criteria)

This protocol defines the fixed criteria used to evaluate ASR runs so future model comparisons use the same methodology.

## Scope

- Applies to CS-Dialogue ASR baseline and future ASR model runs.
- Evaluation unit:
  - Conversation-level references from prepared manifest rows (`item_id`).
  - Hypothesis aggregated from ASR segment outputs by `source_item_id`.

## Inputs

- ASR run directory:
  - `data/runs/asr/<run_id>/segments.jsonl`
  - `data/runs/asr/<run_id>/summary.json`
- Prepared manifest:
  - `data/raw/cs_dialogue/<split>/long_wav/manifest.jsonl`

## Normalization and tokenization (fixed)

1. Remove tag-like tokens: `<...>`
2. Normalize whitespace (collapse multiple spaces)
3. Lowercase English text
4. WER tokenization:
   - Tokens are either:
     - English alnum spans: `[A-Za-z0-9]+`
     - Single CJK chars: `[\u4e00-\u9fff]`
5. CER tokenization:
   - Single char tokens from `[A-Za-z0-9\u4e00-\u9fff]`

Notes:
- This keeps Chinese character sensitivity while still handling English words reasonably in code-switch text.
- WER/CER values from other tokenization schemes are not directly comparable.

## Metrics

- Accuracy:
  - WER (corpus-level)
  - CER (corpus-level)
- Runtime (from ASR summary):
  - `elapsed_sec`
  - `total_audio_sec`
  - `rtf`
  - `audio_sec_per_wall_sec`
  - `segments_per_sec`
- Breakdown:
  - Per conversation WER/CER
  - Per reference language bucket (`en`, `zh`, `mixed`, `unknown`)

## Language bucket rule

- Bucket is derived from reference transcript characters:
  - `mixed`: both Chinese and English letters present
  - `zh`: only Chinese chars
  - `en`: only English letters
  - `unknown`: neither detected

## Outputs

`data/runs/asr_eval/<run_id>/`

- `summary.json`
- `per_split.csv`
- `per_language.csv`
- `per_conversation.jsonl`
- `protocol.json`
- `run_meta.json`

## Reproducibility requirements

- Keep split manifests fixed for comparison.
- Keep ASR decode config fixed unless intentionally testing a change.
- Keep protocol version fixed (`asr_eval_v1`) when comparing to baseline.
