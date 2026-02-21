# Master Context Template — AAI3008 Multilingual PII Masking (ASR → PII/NER → Audio Masking)

Use this as the **single source of truth** for the project roadmap, resources plan, datasets, and deliverables.  
You can paste this into Codex or other chats as shared context.

---

## 0) Project goal

Build an end-to-end pipeline that **de-identifies spoken audio** by detecting **PII** and producing:

- **De-identified (masked) audio**
- **Redacted transcript**
- **Audit JSON** (what was masked, time ranges, labels, confidence, reason)

Primary objective: **minimize PII leakage** while preserving non-PII context.

---

## 1) Datasets

### 1.1 Baseline public datasets (for model selection + early evaluation)

#### A) Speech dataset (ASR robustness): `BAAI/CS-Dialogue`
Purpose:
- Evaluate **ASR behavior on code-switching speech** (EN+ZH), segmentation stability, runtime.

Important:
- Must be **duration-controlled** (sample by total minutes, not number of items).
- Use deterministic sampling (seed) so baseline comparisons are fair and reproducible.

Recommended subset:
- `cs_test`: held-out evaluation subset used as the canonical ASR baseline run

#### B) Text dataset (NER baseline): `wikiann` (EN + ZH only)
Purpose:
- Evaluate **NER quality** (PER/LOC/ORG) as proxy for name/location PII detection baseline.

Policy:
- Use **all data** for EN and ZH.
- Use the dataset’s **train/validation/test** splits as given.

---

### 1.2 Curated gold dataset (for end-to-end privacy evaluation)
Purpose:
- This is the **true KPI** dataset: PII detection + masking effectiveness in audio.

Characteristics:
- Small but high-quality and balanced:
  - English, Chinese, mixed (code-switch)
  - PII types: name, phone, email, ID-like, address-like (as defined in project)
- Includes gold labels:
  - transcript + PII spans (and optionally time spans if available)

---

## 2) Repo working principles

### 2.1 Artifact-driven workflow (reproducible)
- **Scripts/CLI write artifacts** to `data/runs/<stage>/<run_id>/...`
- **Notebooks only read artifacts** and generate plots/tables for reporting

### 2.2 Notebooks vs scripts vs app
- **Notebooks**: evaluation, analysis, plots/tables; also fine-tuning notebooks on cloud
- **Scripts/CLI**: authoritative pipeline execution (batch runs, deterministic outputs)
- **UI app**: thin wrapper over pipeline modules for demo (Streamlit/FastAPI)

---

## 3) Resources plan (what runs where)

### 3.1 Local Linux PC (RTX 2080 Super)
Use for:
- Baseline inference + evaluation (CS-Dialogue ASR slice, WikiAnn NER)
- Gold-set end-to-end evaluation runs
- Integration/testing for pipeline
- Demo preparation and final artifacts

### 3.2 Cloud/Colab (only when needed)
Use for:
- **Fine-tuning** (typically NER/PII model)
- Scaling ASR runs (larger model or longer durations) if local is too slow

Cloud artifacts to export back:
- Fine-tuned weights (HF model folder or checkpoint)
- Training configs + logs
- Update local config to load from saved weights

---

## 4) Evaluation plan (what to measure)

### 4.1 ASR evaluation (CS-Dialogue)
Metrics:
- Runtime: **RTF / throughput**
- Quality: **WER/CER** *only if reference transcript exists and is usable*
- Qualitative: code-switch errors, number recognition, segmentation stability

Reporting:
- Report baseline results on held-out `cs_test` (current canonical).

### 4.2 NER evaluation (WikiAnn EN+ZH)
Metrics:
- Precision / Recall / F1 (and optionally **F2** to align with privacy emphasis)
- Breakdown by:
  - language (EN vs ZH)
  - entity type (PER/LOC/ORG)

Protocol:
- Use train/val for iteration decisions
- Report final baseline metrics on test split

### 4.3 End-to-end evaluation (Gold set)
Metrics:
- PII detection: Recall/F2 by PII type + language bucket
- Masking effectiveness:
  - **Leak rate proxy**: re-run ASR on masked audio and check if PII still appears
  - Over-masking rate: non-PII removed unnecessarily
- Provide before/after examples + audit JSON excerpts

---

## 5) Roadmap (phases + deliverables)

### Phase 1 — Baseline datasets + baseline evaluation (public data)
Objective:
- Establish baselines and pick initial models/configs.

Tasks:
1. Confirm dataset prep outputs exist:
   - CS-Dialogue: duration-controlled manifest (+ optional materialized wav subset)
   - WikiAnn: full EN/ZH train/val/test exported JSONL
2. Run ASR baseline on held-out `cs_test` (canonical baseline split).
3. Run NER baseline evaluation on WikiAnn EN+ZH test split (`en_test`, `zh_test`), using train/val only for iteration when needed.
4. Produce evaluation notebooks:
   - ASR: speed + sample outputs (+ WER/CER if possible)
   - NER: metrics tables + error examples

Deliverables:
- `data/runs/asr/<run_id>/...`
- `data/runs/asr_eval/<run_id>/...`
- `data/runs/pii_eval/<run_id>/...`
- Notebook figures/tables suitable for report

### Phase 1 status (as of 2026-02-21)
- Dataset artifacts confirmed:
  - `data/raw/cs_dialogue/cs_test/long_wav/{manifest.jsonl,stats.json}`
  - `data/raw/wikiann/{en,zh}/{train,validation,test}.jsonl`
- Canonical ASR baseline/eval:
  - `data/runs/asr/asr_baseline_cs_test/`
  - `data/runs/asr_eval/asr_eval_cs_test_v1/`
- Canonical NER baseline/eval:
  - `data/runs/pii_eval/pii_eval_wikiann_baseline_v1/`
- Analysis notebooks:
  - `notebooks/01_asr_analysis.ipynb`
  - `notebooks/02_pii_eval.ipynb`

---

### Phase 2 — Thin end-to-end pipeline (early integration)
Objective:
- De-risk integration early: transcript spans → timestamps → masking.

Tasks:
1. Implement minimal end-to-end:
   - ASR → transcript + timestamps
   - PII detection (rules + NER) → spans
   - span-to-time mapping → mask audio
   - audit JSON export
2. Run on a few local audio files (not gold yet)

Deliverables:
- Masked audio examples + audit JSON
- Pipeline runnable via scripts/CLI

---

### Phase 3 — Gold set evaluation (end-to-end KPI)
Objective:
- Measure privacy effectiveness on curated gold set.

Tasks:
1. Create gold set v1 (balanced, labeled)
2. Run end-to-end pipeline on gold set
3. Evaluate: PII recall/F2, leak proxy, over-masking
4. Identify bottleneck: ASR vs NER vs alignment

Deliverables:
- Gold set + labels
- `data/runs/eval/<run_id>/summary.json` + tables
- Examples for report/demo

---

### Phase 4 — Targeted fine-tuning (only if justified) + re-evaluation
Objective:
- Improve the bottleneck and prove improvement.

Decision rules:
- If errors are mostly NER/PII misses → fine-tune PII/NER model (cloud/Colab)
- If errors are mostly alignment/timestamps → improve alignment/mapping logic
- If errors are mostly ASR recognition → adjust ASR model/settings (avoid ASR fine-tuning unless necessary)

Tasks:
1. Fine-tune (cloud/Colab), export weights + logs
2. Update local configs to point to new weights
3. Re-run gold evaluation
4. Produce before/after comparison tables (ablation summary)

Deliverables:
- Fine-tuned weights + training logs
- Improved gold metrics + baseline vs improved comparison

---

### Phase 5 — Demo-ready pipeline + UI
Objective:
- Stable, presentable deliverable for grading/demo.

Tasks:
1. Harden scripts/CLI (robust IO, consistent artifacts, clear errors)
2. Add minimal UI app (Streamlit/FastAPI):
   - upload/select audio
   - run pipeline
   - show redacted transcript + detected PII list + timestamps
   - output masked audio + audit JSON
3. Final evaluation visuals + demo examples

Deliverables:
- Runnable demo pipeline + UI app
- Final evaluation figures/tables

---

## 6) “What uses what” summary (quick reference)

- **Local PC**: baseline inference/eval + gold eval + end-to-end runs + demo
- **Cloud/Colab**: fine-tuning and/or heavy ASR scaling; export weights back
- **Notebooks**: analysis/plots reading from artifacts
- **Scripts/CLI**: pipeline execution and artifact generation
- **App**: thin wrapper for presentation

---

## 7) Notes for Codex/agents
- Prefer deterministic configs (seeded sampling, run_id driven outputs).
- Do not download full CS-Dialogue; always duration-control.
- WikiAnn EN+ZH can be fully exported; dataset has splits.
- Keep notebooks as “read-only” from artifacts; scripts own the pipeline.

---
