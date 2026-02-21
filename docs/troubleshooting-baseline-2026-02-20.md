# Baseline Troubleshooting Log (2026-02-20)

This note records the issues encountered while preparing baseline evaluation artifacts and what was changed.

## 1) ASR stage produced empty transcripts

- Symptom:
  - `data/runs/asr/asr_baseline_local/segments.jsonl` had rows with empty `"text"`.
  - Downstream PII and eval outputs were all zero-signal.
- Root cause:
  - `src/asr/run.py` was still a placeholder implementation and did not run any ASR model.
- Fix:
  - Replaced placeholder with real `faster-whisper` inference.
  - Added support for `data.source_type: manifest|local_dir`.
  - Added `summary.json` with run-level timing and counts.
  - Added `data.max_items` support for quick dev runs.

## 2) Stage execution race created empty downstream files

- Symptom:
  - A run that launched pipeline stages in parallel created empty PII/mask outputs.
- Root cause:
  - `pii` started before `asr` finished writing artifacts.
- Fix:
  - Run stages sequentially for dependencies:
    - ASR -> PII -> Mask -> Eval.

## 3) Local dependency install blockers

- Symptom:
  - `ModuleNotFoundError: faster_whisper` on local script execution.
  - `pip install -r requirements.txt` blocked by PEP 668 (`externally-managed-environment`).
  - Creating a `.venv` still failed to install due network resolution limits in this environment.
- Fix/workaround:
  - Ran ASR inside Docker app container where dependencies are available.

## 4) Masking bottleneck after real ASR

- Symptom:
  - After ASR began outputting segment-level rows (2k+ rows), masking became extremely slow.
- Root cause:
  - `src/mask/run.py` copied full source audio once per segment row, repeatedly duplicating the same large files.
- Fix:
  - Updated masking stage to deduplicate by `source_audio`:
    - Aggregate segment stats per source audio.
    - Copy each source file once.
    - Write file-level audit rows in `audit.jsonl`.
    - Keep audit compact (segment counts + sample IDs), not full segment-id lists.
  - `run_meta.json` now includes:
    - `input_rows`, `source_rows`, `files_copied`, `files_missing`.

## Current status after fixes

- ASR now generates non-empty transcript segments.
- PII and eval consume real ASR outputs.
- Mask stage no longer scales linearly with duplicate full-file copies across segments.

## 5) Baseline completion update (2026-02-20)

Completed for ASR baseline prep/eval inputs:
- Created duration-budgeted dataset artifacts:
  - `data/raw/cs_dialogue/cs_dev/long_wav/{manifest.jsonl,stats.json}`
  - `data/raw/cs_dialogue/cs_final/long_wav/{manifest.jsonl,stats.json}`
- Added dedicated ASR configs:
  - `configs/asr_cs_dev.yaml`
  - `configs/asr_cs_final.yaml`
- Ran ASR for both splits:
  - `data/runs/asr/asr_baseline_cs_dev/{segments.jsonl,summary.json,run_meta.json}`
  - `data/runs/asr/asr_baseline_cs_final/{segments.jsonl,summary.json,run_meta.json}`
- Extended ASR summaries with runtime metrics:
  - `total_audio_sec`, `total_audio_min`, `rtf`, `audio_sec_per_wall_sec`
- Created notebook-friendly consolidated baseline metrics:
  - `data/runs/asr/baseline_asr_overview.csv`
  - `data/runs/asr/baseline_asr_overview.json`

Still pending / caveats:
- `cs_dev` target was 60 min, but current artifact is ~114.1 min.
  - Cause: available snapshot items are long utterances; duration budget is coarse at file level.
- WER/CER are still not computed.
  - Need a validated text normalization/alignment protocol for CS-Dialogue references.
- HF direct dataset load path remains unavailable in this environment; snapshot fallback is used.

## 6) cs_dev / cs_final overlap fix (2026-02-20)

- Symptom:
  - `cs_dev` and `cs_final` shared the same source conversations (dev became a prefix subset of final).
- Root cause:
  - Prep runs sampled independently from the same source with the same seed and had no exclusion rule.
- Fix:
  - Added exclusion support to `scripts/01_prep_cs_dialogue.py`:
    - `--exclude_manifest`
    - `--exclude_utt_ids_file`
  - Regenerated `cs_final` with:
    - `--exclude_manifest data/raw/cs_dialogue/cs_dev/long_wav/manifest.jsonl`
  - Re-ran `asr_baseline_cs_final` on the updated disjoint manifest.

Result:
- `cs_dev` utt_ids:
  - `ZH-CN_U0058_S0`, `ZH-CN_U0096_S0`
- `cs_final` utt_ids:
  - `ZH-CN_U0038_S0`, `ZH-CN_U0080_S0`, `ZH-CN_U0087_S0`, `ZH-CN_U1034_S0`, `ZH-CN_U2013_S0`
- Overlap:
  - none

## 7) ASR WER/CER protocol and evaluation run (2026-02-20)

- Implemented reusable ASR evaluation runner:
  - `src/asr/eval.py`
  - `scripts/11_eval_asr.py`
  - `configs/asr_eval.yaml`
- Added protocol documentation:
  - `docs/asr-eval-protocol-v1.md`
- Generated protocol-based evaluation artifacts:
  - `data/runs/asr_eval/asr_eval_baseline_v1/summary.json`
  - `data/runs/asr_eval/asr_eval_baseline_v1/per_split.csv`
  - `data/runs/asr_eval/asr_eval_baseline_v1/per_language.csv`
  - `data/runs/asr_eval/asr_eval_baseline_v1/per_conversation.jsonl`

Protocol highlights:
- Fixed text normalization/tokenization for mixed EN+ZH.
- Conversation-level aggregation from ASR segments by `source_item_id`.
- Corpus-level WER/CER plus per-conversation and per-language-bucket breakdown.

## 8) Housekeeping done (2026-02-20)

- Updated run instructions and references in `README.md`.
- Refreshed consolidated ASR overview artifacts after split updates:
  - `data/runs/asr/baseline_asr_overview.csv`
  - `data/runs/asr/baseline_asr_overview.json`

## 9) Notebook expansion + local baseline retirement (2026-02-20)

Work completed:
- Replaced placeholder notebook with runnable ASR analysis notebook:
  - `notebooks/01_asr_analysis.ipynb`
- Notebook sections now include:
  - artifact loading from `data/runs/asr_eval/asr_eval_baseline_v1`
  - split-level WER/CER/runtime summaries
  - weighted aggregate metrics
  - per-conversation ranking and outlier checks
  - worst-conversation text-level inspection (reference vs hypothesis)
  - segment-language distribution analysis
- Baseline metrics referenced in notebook:
  - `cs_dev`: WER `0.1561`, CER `0.1260`, RTF `0.0142`
  - `cs_final`: WER `0.3650`, CER `0.5015`, RTF `0.0134`
  - weighted overall: WER `0.3056`, CER `0.3879`
  - notable outlier: `train_00000004_ZH-CN_U1034_S0` (WER `0.9028`, CER `1.7497`)

Cleanup:
- Removed local ASR run artifacts:
  - deleted `data/runs/asr/asr_baseline_local/`
- Updated default config references to avoid stale local run paths:
  - `configs/asr.yaml`: default run id -> `asr_baseline_cs_dev`
  - `configs/pii.yaml`: `run.input_asr_dir` -> `data/runs/asr/asr_baseline_cs_dev`
  - `configs/eval.yaml`: `inputs.asr_dir` -> `data/runs/asr/asr_baseline_cs_dev`

## 10) Local artifact cleanup + full-set ASR eval rerun (2026-02-20)

Cleanup completed:
- Removed stale local-run directories:
  - `data/runs/pii/pii_baseline_local/`
  - `data/runs/eval/eval_baseline_local/`
  - `data/runs/masked/pii_baseline_local/`

Full-set ASR eval run:
- Added config:
  - `configs/asr_eval_full_set.yaml`
- Executed:
  - `python3 scripts/11_eval_asr.py --config configs/asr_eval_full_set.yaml` (in Docker app container)
- New output:
  - `data/runs/asr_eval/asr_eval_full_set_v1/`
    - `summary.json`
    - `per_split.csv`
    - `per_language.csv`
    - `per_conversation.jsonl`
    - `run_meta.json`

Run scope:
- Evaluates the full prepared benchmark set (`cs_dev` + `cs_final`), total 7 conversations.

## 11) Long-run ASR resiliency patch for full CS-Dialogue (2026-02-20)

Issue:
- Full `cs_full` ASR run (~130 hours audio) had poor observability and no safe resume points.

Changes made:
- Patched `src/asr/run.py` to support:
  - append-mode checkpoints to `segments.jsonl`
  - `run.resume` skip logic based on existing `source_item_id` values
  - periodic heartbeat in `progress.json` with ETA and done/remaining counts
  - periodic console progress logs (`[ASR] done=...`)
  - graceful interrupt handling with checkpoint flush and resumable state
  - per-file failure isolation (warn and continue, instead of hard abort)
- Added full-run tracking config defaults:
  - `configs/asr_cs_full.yaml`
    - `resume: true`
    - `checkpoint_every_files: 2`
    - `progress_every_files: 1`
    - `write_progress: true`
- Added smoke config:
  - `configs/asr_cs_full_smoke.yaml` (`max_items: 1`) for fast verification.

Validation:
- Smoke run completed with tracking artifacts:
  - `data/runs/asr/asr_baseline_cs_full_smoke/segments.jsonl`
  - `data/runs/asr/asr_baseline_cs_full_smoke/progress.json`
  - `data/runs/asr/asr_baseline_cs_full_smoke/summary.json`
