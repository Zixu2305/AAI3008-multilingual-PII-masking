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
- `scripts/`: dataset preparation scripts.
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

Canonical execution path is the unified CLI in `src/cli.py`:

```bash
python -m src.cli asr --config configs/asr.yaml
python -m src.cli asr-eval --config configs/asr_eval.yaml
python -m src.cli pii --config configs/pii.yaml
python -m src.cli pii-eval --config configs/pii_eval_wikiann.yaml
python -m src.cli pii-eval-gold --config configs/pii_eval_gold.yaml
python -m src.cli mask --config configs/pii.yaml
python -m src.cli eval --config configs/eval.yaml
```

Current default ASR config points to the local CTranslate2 export at `data/models/whisper-large-v3/ct2`. If that folder is unavailable on another machine, either switch `asr.model_name` back to a standard `faster-whisper` model name such as `large-v3`, or point it at the Hugging Face CT2 repo `ImShooShoo/whisper-large-v3`.

Current default PII config points to the local v1 fine-tuned NER checkpoint at `data/models/ner_finetuned_gold_v1/final/best`. If that folder is unavailable on another machine, either switch `pii.ner.model_name` back to `Davlan/xlm-roberta-base-ner-hrl`, or point it at the Hugging Face repo `ImShooShoo/ner_finetuned_gold_roberta`.

The `ID` category is handled as a hybrid privacy label, not NRIC-only. Runtime rules cover exact NRIC/FIN patterns plus cue-word-gated identifiers such as `reference ID 9`, `application ID 12`, and `员工编号 STAFF-ID-5521`.

## Dataset Prep Scripts

Dataset prep remains script-based:

- `scripts/01_prep_cs_dialogue.py`: prepare duration-budgeted CS-Dialogue subsets + manifests.
- `scripts/02_prep_wikiann.py`: export WikiAnn EN/ZH splits into JSONL.
- `scripts/03_prep_gold_set.py`: convert gold JSON to prepared JSONL (`dev` / `eval`).
- `scripts/04_prep_gold_bio.py`: convert gold spans to BIO token labels for NER fine-tuning.

## Streamlit App

### Local quickstart

Tested with Python 3.10.

1. Create and activate a virtual environment.
2. Install Python dependencies.
3. Ensure FFmpeg is available on your machine.
4. Launch Streamlit and open the app on `localhost`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

FFmpeg is recommended for uploaded `mp3` / `m4a` inputs and for the masking stage audio backends.

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`

Open the app at:

```text
http://localhost:8501
```

The repo now includes [`.streamlit/config.toml`](./.streamlit/config.toml), which prefers `localhost` for the browser URL because the microphone recorder generally needs `localhost` or `https`.

### What the reader should prepare

- Python environment with `requirements.txt` installed
- Optional local model folders if you do not want first-run downloads:
  - `data/models/whisper-large-v3/ct2`
  - `data/models/ner_finetuned_gold_v1/final/best`
- Internet access for first run if local models are missing
- Optional `HF_TOKEN` only if you later make the Hugging Face model repos private
- Browser microphone permission if using the recorder

### Run command

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app now resolves models in this order:

- ASR: local `data/models/whisper-large-v3/ct2` -> Hugging Face `ImShooShoo/whisper-large-v3` -> baseline `large-v3`
- NER: local `data/models/ner_finetuned_gold_v1/final/best` -> Hugging Face `ImShooShoo/ner_finetuned_gold_roberta` -> baseline `Davlan/xlm-roberta-base-ner-hrl`

Downloaded model files are cached under `data/cache/hf/`. Set `HF_TOKEN` only if you later switch to private Hugging Face repos.

### Notes before a demo

- If you know the clip is only English or only Chinese, set the ASR language in the sidebar instead of leaving it on `auto`. This avoids mixed-language re-decode and is faster.
- On CPU, the fine-tuned `whisper-large-v3` path can be slow on longer bilingual clips. Multi-minute audio can take 10+ minutes.
- For live demos, a GPU machine is strongly preferred.
- If you run the app on another machine, use an SSH tunnel and still open the app on `http://localhost:8501` from your browser:

```bash
ssh -L 8501:localhost:8501 user@your-machine
```

### Microphone recorder

- Use Chrome or Edge first.
- Open the app on `localhost` or `https`. Plain LAN URLs such as `http://192.168.x.x:8501` can prevent the recorder from starting.
- If the browser does not show a permission prompt, check site-level microphone permissions and reload the page.

### Common first-run issues

- `ValueError: tiktoken is required...`
  - Fix: `pip install -r requirements.txt`
- Missing model folders
  - The app will try local models first, then download:
    - ASR: `ImShooShoo/whisper-large-v3`
    - NER: `ImShooShoo/ner_finetuned_gold_roberta`
- Masking stage copies original audio instead of masking
  - Usually an audio backend / FFmpeg issue; check FFmpeg installation

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
