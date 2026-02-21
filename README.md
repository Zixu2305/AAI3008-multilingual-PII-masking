# AAI3008 Multilingual PII Masking

Multilingual pipeline for ASR -> PII detection -> audio masking -> evaluation.

## Project Map

Top-level folders/files:

- `app/`: demo-facing app code (UI entry points such as Streamlit).
- `configs/`: YAML configs for dataset prep, model runs, and evaluation runs.
- `data/`: all datasets and generated run artifacts.
- `data/raw/`: prepared source datasets used by the pipeline.
- `data/runs/`: outputs from each pipeline stage, organized by `run_id`.
- `docker/`: container definitions (`Dockerfile`, `docker-compose.yml`).
- `docs/`: protocol docs, troubleshooting logs, and process notes.
- `notebooks/`: analysis/report notebooks (read artifacts; do not own pipeline execution).
- `scripts/`: executable wrappers for each stage and prep task.
- `src/`: core pipeline implementation modules.
- `.env` / `.env.example`: environment configuration for Docker/runtime.
- `requirements.txt`: Python dependencies.
- `README.md`: project usage and structure reference.
- `master-context.md`: roadmap/spec context document.

`src/` module intent:

- `src/asr/`: ASR inference and ASR evaluation logic.
- `src/pii/`: PII span detection logic.
- `src/mask/`: audio masking stage logic.
- `src/eval/`: general pipeline evaluation aggregation.
- `src/utils/`: shared helpers (IO, artifact path handling).
- `src/cli.py`: unified CLI entrypoint.

`data/runs/` stage definitions:

- `asr`: model inference outputs (segment text/timestamps + runtime summary).
- `asr_eval`: ASR scoring outputs (WER/CER + per-split/per-language/per-conversation metrics).
- `pii`: detected PII spans over ASR text.
- `masked`: masked-audio outputs + masking audit records.
- `eval`: end-to-end pipeline summary tables/JSON.

## Setup

```bash
cp .env.example .env

docker compose --env-file .env -f docker/docker-compose.yml build

docker compose --env-file .env -f docker/docker-compose.yml run --rm app bash
```

## Dataset prep (baseline)

HF caches are mounted from host (typically `~/.cache/huggingface`) and are not the same as extracted dataset files. Prep scripts materialize concrete artifacts under `data/raw/...`.

### CS-Dialogue (duration budgeted subset)

```bash
docker compose --env-file .env -f docker/docker-compose.yml run --rm app \
  python3 scripts/01_prep_cs_dialogue.py --subset long_wav --split train --max_minutes 60
```

Verification:

```bash
wc -l data/raw/cs_dialogue/long_wav/manifest.jsonl
ls -lah data/raw/cs_dialogue/long_wav/audio | head
cat data/raw/cs_dialogue/long_wav/stats.json
```

### WikiAnn (full EN + ZH splits)

```bash
docker compose --env-file .env -f docker/docker-compose.yml run --rm app \
  python3 scripts/02_prep_wikiann.py
```

Verification:

```bash
wc -l data/raw/wikiann/en/train.jsonl
wc -l data/raw/wikiann/zh/train.jsonl
head -n 1 data/raw/wikiann/en/train.jsonl
```

Outputs:

```text
data/raw/
  cs_dialogue/
    long_wav/
      audio/*.wav
      manifest.jsonl
      stats.json
  wikiann/
    en/
      train.jsonl
      validation.jsonl
      test.jsonl
      label_map.json
      stats.json
    zh/
      train.jsonl
      validation.jsonl
      test.jsonl
      label_map.json
      stats.json
```

## Run Pipeline Stages

### Existing scripts (thin wrappers)

```bash
python scripts/10_run_asr.py --config configs/asr.yaml
python scripts/11_eval_asr.py --config configs/asr_eval.yaml
python scripts/20_run_pii.py --config configs/pii.yaml
python scripts/21_eval_pii.py --config configs/pii_eval_wikiann.yaml
python scripts/30_mask_audio.py --config configs/pii.yaml
python scripts/40_eval.py --config configs/eval.yaml
```

### Script quick guide (when to use)

- `scripts/00_smoke_asr.py`
  - Purpose: minimal quick ASR sanity check using `configs/asr.yaml`.
  - Status: optional/legacy helper.
  - Note: mostly redundant now that `scripts/10_run_asr.py` is stable.
- `scripts/01_prep_cs_dialogue.py`
  - Purpose: prepare duration-budgeted CS-Dialogue subsets and manifests.
  - Status: core.
- `scripts/02_prep_wikiann.py`
  - Purpose: export full WikiAnn EN/ZH train/validation/test JSONL artifacts.
  - Status: core.
- `scripts/10_run_asr.py`
  - Purpose: run ASR inference and produce segment artifacts.
  - Status: core.
- `scripts/11_eval_asr.py`
  - Purpose: run fixed-protocol ASR evaluation (WER/CER + runtime breakdowns).
  - Status: core for ASR benchmarking.
- `scripts/20_run_pii.py`
  - Purpose: run PII span detection on ASR outputs.
  - Status: core.
- `scripts/21_eval_pii.py`
  - Purpose: run WikiAnn EN/ZH NER baseline evaluation (entity-level P/R/F1/Fbeta).
  - Status: core for NER benchmarking.
- `scripts/30_mask_audio.py`
  - Purpose: run masking stage from PII outputs.
  - Status: core pipeline stage (current masking logic is placeholder behavior).
- `scripts/40_eval.py`
  - Purpose: general pipeline eval summary aggregation.
  - Status: core for end-to-end pipeline runs.

### Unified CLI (additive)

```bash
python -m src.cli asr --config configs/asr.yaml
python -m src.cli pii --config configs/pii.yaml
python -m src.cli pii-eval --config configs/pii_eval_wikiann.yaml
python -m src.cli mask --config configs/pii.yaml
python -m src.cli eval --config configs/eval.yaml
```

## Run Artifact Contract

All stage outputs are written under `data/runs/<stage>/<run_id>/`:

```text
data/runs/
  asr/<run_id>/
    segments.jsonl
    summary.json
    run_meta.json
  asr_eval/<run_id>/
    summary.json
    per_split.csv
    per_language.csv
    per_conversation.jsonl
    protocol.json
    run_meta.json
  pii/<run_id>/
    spans.jsonl
    run_meta.json
  pii_eval/<run_id>/
    summary.json
    per_split.csv
    per_type.csv
    per_item.jsonl
    protocol.json
    run_meta.json
  masked/<run_id>/
    masked_audio/
    audit.jsonl
    run_meta.json
  eval/<run_id>/
    summary.json
    per_type.csv
    per_lang.csv
    run_meta.json
```

`run_id` is config-driven from YAML (`run.run_id`).

## Troubleshooting Notes

Baseline run issues and fixes are documented in:

- `docs/troubleshooting-baseline-2026-02-20.md`
- `docs/asr-eval-protocol-v1.md`
