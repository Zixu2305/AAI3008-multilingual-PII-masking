# Consolidated Progress Since Gold Set Upload

Date generated: 2026-03-17 (Asia/Singapore)

## Scope and boundary

- Baseline commit for this report: `f9389e8` (2026-03-02, `KayCheng-01`, "Upload Gold Set")
- Time window covered: commits after `f9389e8` up to current `HEAD`

## Gold set upload (starting point)

Commit: `f9389e8` (2026-03-02, `KayCheng-01`)

What was added:
- Gold-set annotation JSON files:
  - `gold_set/en.json`
  - `gold_set/zh.json`
  - `gold_set/mix.json`
- Gold-set audio structure under:
  - `gold_set/dev/{En,Zh,Mix}/...`
  - `gold_set/eval/{En,Zh,Mix}/...`

## Post-upload commits by other contributors

### 1) `ed7d1ba` (2026-03-07, Jiarui23) — "NER Update. Huge chunk."

Main work delivered:
- Gold-set prep and NER training/eval flow:
  - Added `scripts/03_prep_gold_set.py` (JSON -> prepared JSONL)
  - Added `scripts/04_prep_gold_bio.py` (char-offset entities -> BIO token labels)
  - Added `notebooks/04_finetune_ner.ipynb`
  - Added `notebooks/05_pii_gold_eval.ipynb`
- Gold-set evaluation stage:
  - Added `src/pii/eval_gold.py`
  - Added `scripts/22_eval_pii_gold.py` (later consolidated into `python -m src.cli pii-eval-gold`)
  - Added CLI command support in `src/cli.py` (`pii-eval-gold`)
  - Added `configs/pii_eval_gold.yaml`
- PII pipeline enhancements:
  - Expanded rule set in `src/pii/run.py` and `configs/pii.yaml`
  - Added optional LLM safety-net module `src/pii/llm_detect.py`
- Prepared gold artifacts committed:
  - `data/prepared/gold_set/dev.jsonl`
  - `data/prepared/gold_set/eval.jsonl`
- Added project docs:
  - `NERedits.md`
  - `rules.md`
- Added a transcript-level Streamlit PII demo:
  - `app.py` (later replaced by `app/streamlit_app.py`)
- Added local NER model folders (tokenizer/config side only):
  - `data/models/ner_finetuned_gold_v1/final/best/...`
  - `data/models/ner_finetuned_gold_v2/final/best/...`

### 2) `404ec08` (2026-03-07, Jiarui23) — "Updated some more regex."

Main work delivered:
- Regex refinement in `src/pii/run.py`:
  - Added `SG_POSTAL_AFTER_ADDR_RE` to detect bare 6-digit postal codes following addresses
  - Tightened one English address variant to reduce over-broad matches
- Documentation updates in:
  - `rules.md`
  - `NERedits.md`

### 3) `0b077f4` (2026-03-14, Neo Chuan Zong) — "ASR Fine tuning and Eval"

Main work delivered:
- Added ASR fine-tuning/evaluation notebook suite:
  - `notebooks/06_finetune_multilingual_whisper.ipynb`
  - `notebooks/07_finetune_multilingual_whisper_full_audio_chunked.ipynb`
  - `notebooks/08_infer_from_saved_checkpoint_colab_temp.ipynb`
  - `notebooks/09_asr_eval_three_model_comparison.ipynb`
- Added ASR findings summaries:
  - `ASR_EVAL_FINDINGS.md`
  - `docs/asr_findings_summary.md`

### 4) `7bbe4fc` (2026-03-16, Neo Chuan Zong) — "Create ASR_MODEL_INFERENCE_GUIDE.md"

Main work delivered:
- Added ASR checkpoint usage guide:
  - `ASR_MODEL_INFERENCE_GUIDE.md`

## Contributor summary (post-upload only)

- `Jiarui23`: 2 commits, 28 unique files touched
- `Neo Chuan Zong`: 2 commits, 7 unique files touched

## Current integration state (important)

### Integrated into repo pipeline code

- Gold-set prep/eval codepath for PII is implemented and callable:
  - `python -m src.cli pii-eval-gold --config configs/pii_eval_gold.yaml`
- PII rules + optional NER + optional LLM logic exists in `src/pii/run.py`

### Present but not fully integrated by default

- NER fine-tuned checkpoint paths are documented, but default configs still point to baseline model:
  - `configs/pii.yaml` -> `Davlan/xlm-roberta-base-ner-hrl`
  - `configs/pii_eval_gold.yaml` -> `Davlan/xlm-roberta-base-ner-hrl`

### Not stored in this repo

- NER weight binaries are not in repo (only tokenizer/config files are present in local model folders).
- ASR fine-tuned checkpoints are referenced in notebooks/docs (e.g., Drive paths), but not committed to this repository.

## Practical conclusion

Since the gold set upload, the team has completed:
- A substantial NER/gold-eval workflow integration into scripts/CLI and core `src/pii/*` modules.
- Rule-based PII pattern expansion for local address/ID/postal formats.
- A separate ASR fine-tuning and evaluation notebook track with written findings and inference guidance.

The remaining gap is artifact portability: bringing final model weight files (NER and ASR) into an accessible runtime location and wiring default configs to those checkpoints for reproducible end-to-end runs.
