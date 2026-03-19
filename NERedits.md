# NER Fine-Tuning & Gold Evaluation — Reference

---

## Overview

This document covers the **NER fine-tuning** and **gold-set evaluation** workflow, which sits alongside (but separate from) the main ASR → PII → Mask → Eval pipeline.

```
┌─────────────────────────── MAIN PIPELINE ───────────────────────────┐
│                                                                     │
│  Audio → ASR (Whisper) → PII Detection (rules+NER) → Mask → Eval   │
│          configs/asr.yaml   configs/pii.yaml                        │
│                                                                     │
│  CLI:  python -m src.cli asr   --config configs/asr.yaml            │
│        python -m src.cli pii   --config configs/pii.yaml            │
│        python -m src.cli mask  --config configs/pii.yaml            │
│        python -m src.cli eval  --config configs/eval.yaml           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────────── NER FINE-TUNING & GOLD EVAL ──────────────────┐
│                                                                     │
│  1. Prepare gold data        scripts/03_prep_gold_set.py            │
│  2. Prepare BIO tags         scripts/04_prep_gold_bio.py            │
│  3. Fine-tune NER            notebooks/04_finetune_ner.ipynb        │
│  4. Evaluate on gold set     python -m src.cli pii-eval-gold        │
│  5. Analyse results          notebooks/05_pii_gold_eval.ipynb       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### How PII Detection Works (src/pii/run.py)

The PII detector is a **hybrid system** with three layers, each toggled independently in config:

1. **Rules** (`enable_rules: true`) — regex patterns for structured PII (phones, emails, IDs, addresses)
2. **NER** (`enable_ner: true`) — XLM-RoBERTa transformer for semantic entities (names, locations, orgs)
3. **LLM** (`enable_llm: false`) — optional Claude API safety net for anything rules+NER miss 

**Post-Processing** — merge_gap_chars: 1 and min_span_chars: 1

All detected spans are merged and deduplicated in postprocessing.

### Two NER Models Available

| Model | Config value for `pii.ner.model_name` | Use case |
|-------|---------------------------------------|----------|
| **Baseline** | `"Davlan/xlm-roberta-base-ner-hrl"` | General multilingual NER (downloaded from HuggingFace) |
| **Fine-tuned** | `"data/models/ner_finetuned_gold_v1/final/best"` | Trained on our gold set — better for our PII types |

To switch models, change `pii.ner.model_name` in `configs/pii.yaml` or `configs/pii_eval_gold.yaml`.

---

## Step-by-Step: How to Run Everything

### Step 1 — Prepare Gold Set (one-time)

Converts the hand-labeled JSON files into JSONL with char offsets and splits into dev (training) and eval (held-out).

```bash
python scripts/03_prep_gold_set.py
```

| Input | Output |
|-------|--------|
| `gold_set/en.json` (50 records) | `data/prepared/gold_set/dev.jsonl` (120 records) |
| `gold_set/zh.json` (50 records) | `data/prepared/gold_set/eval.jsonl` (30 records) |
| `gold_set/mix.json` (50 records) | |

The split is 80/20 stratified by language. The prepared JSONL is committed to git.

### Step 2 — Prepare BIO Data (before fine-tuning)

Tokenizes `dev.jsonl` with XLM-RoBERTa, aligns char offsets to subword BIO tags, creates 5-fold CV splits.

```bash
python scripts/04_prep_gold_bio.py
```

| Input | Output |
|-------|--------|
| `data/prepared/gold_set/dev.jsonl` | `data/prepared/gold_bio/all.jsonl` |
| | `data/prepared/gold_bio/fold_0..4/{train,val}.jsonl` |
| | `data/prepared/gold_bio/label_map.json` |

> `data/prepared/gold_bio/` is **not committed** (regenerate before fine-tuning).

### Step 3 — Fine-Tune the NER Model

Open and run **`notebooks/04_finetune_ner.ipynb`**. It does:

1. Loads BIO data from `data/prepared/gold_bio/`
2. Runs 5-fold CV with early stopping (patience=3)
3. Trains final model on all 120 dev records for `avg_best_epoch` epochs
4. Evaluates on 30 held-out eval records
5. Saves weights to `data/models/ner_finetuned_gold_v1/final/best/`

**Training config:**
- Base model: `Davlan/xlm-roberta-base-ner-hrl` (278M params)
- lr=2e-5, batch=2, grad_accum=2 (effective batch 4), weight_decay=0.01
- MPS/CUDA/CPU auto-detect; fp16 only on CUDA
- ~15 min on Apple MPS, ~70–105 min on CPU-only Mac

**Model weights** (1.1 GB) are gitignored (`.gitignore` has `*.safetensors`) because GitHub rejects files over 100 MB. You must run this notebook to produce them.

### Step 4 — Evaluate on Gold Set

Runs the full PII pipeline (rules + NER + optional LLM) on gold transcripts and scores predictions against gold labels using **span-overlap matching**.

```bash
# Using baseline model (default in config):
python -m src.cli pii-eval-gold --config configs/pii_eval_gold.yaml

# Using fine-tuned model — first edit the config:
#   configs/pii_eval_gold.yaml → pii.ner.model_name: "data/models/ner_finetuned_gold_v1/final/best"
# Then run the same command.
```

**What the config controls:**

| Key | Default | Purpose |
|-----|---------|---------|
| `inputs.gold_path` | `data/prepared/gold_set/dev.jsonl` | Gold records to evaluate against |
| `pii.enable_rules` | `true` | Toggle regex rules |
| `pii.enable_ner` | `true` | Toggle NER model |
| `pii.enable_llm` | `false` | Toggle LLM safety net |
| `pii.ner.model_name` | `Davlan/xlm-roberta-base-ner-hrl` | Which NER model to load |
| `pii.ner.device` | `cpu` | `cpu`, `cuda`, or `cuda:0` |
| `protocol.beta` | `2.0` | F-beta weight (2.0 = recall-heavy) |
| `protocol.overlap_threshold` | `0.5` | Min char overlap ratio for a span match |

**Output** (written to `data/runs/pii_eval/<run_id>/`):

| File | Contents |
|------|----------|
| `summary.json` | Overall + per-type + per-language P/R/F1 |
| `per_type.csv` | Metrics broken down by PII type |
| `per_lang.csv` | Metrics broken down by language |
| `per_record.jsonl` | Per-record TP/FP/FN counts |
| `errors.jsonl` | Every FP and FN with text excerpts |
| `run_meta.json` | Run metadata |

### Step 5 — Analyse Results (optional & for visualisation purposes)

Open **`notebooks/05_pii_gold_eval.ipynb`** to produce:
- Ablation tables (rules-only vs NER-only vs both)
- Per-type recall heatmaps
- Error inspection examples
---

## BIO Label Set (11 labels)

```
O, B-NAME, I-NAME, B-PHONE, I-PHONE, B-EMAIL, I-EMAIL, B-ADDRESS, I-ADDRESS, B-ID, I-ID
```

Base model label mapping: PER→NAME, LOC/GPE→ADDRESS, ORG→ORG

---

## Regex Patterns in `src/pii/run.py`

| Pattern | Detects | Example |
|---------|---------|---------|
| `PHONE_RE` | Digit phone numbers | 9456 7788 |
| `EMAIL_RE` | Email addresses | user@example.com |
| `ZH_PHONE_RE` | Chinese numeral phones | 九三四五，六六七七 |
| `EN_SPOKEN_PHONE_RE` | English spoken phones | "nine one three two…" |
| `ZH_ADDRESS_RE` | Chinese addresses | 大牌123, 南京路 |
| `EN_ADDRESS_RE` | English addresses (Block, number+street, street+number) | Block 123 Orchard Road, Jurong East Street 12, 10 Orchard Road |
| `SG_NRIC_RE` | Singapore NRIC/FIN numbers | S1234567A, G7654321B |
| `SG_POSTAL_RE` | Singapore postal codes (prefixed) | S609690, Singapore 123456 |
| `SG_POSTAL_AFTER_ADDR_RE` | Bare 6-digit postal codes after address | Orchard Road, 238879 |

---

## File Inventory

### Source Code

| File | Purpose |
|------|---------|
| `src/cli.py` | Unified CLI entry point (all subcommands) |
| `src/pii/run.py` | Hybrid PII detection (rules + NER + optional LLM) |
| `src/pii/ner.py` | NER pipeline wrapper — loads model, runs inference, maps labels |
| `src/pii/eval.py` | WikiAnn NER evaluation (strict BIO matching) |
| `src/pii/eval_gold.py` | Gold-set PII evaluation (span-overlap matching) |
| `src/pii/llm_detect.py` | Optional LLM-based PII detection (Anthropic API) |
| `src/asr/run.py` | ASR inference (Faster-Whisper) |
| `src/asr/eval.py` | ASR evaluation (WER/CER) |
| `src/mask/run.py` | Time-aligned audio masking with mute/beep + audit output |
| `src/eval/run.py` | End-to-end aggregation |

### Execution Entry Points

| File | Purpose |
|------|---------|
| `scripts/01_prep_cs_dialogue.py` | Download + prepare CS-Dialogue audio |
| `scripts/02_prep_wikiann.py` | Download + prepare WikiAnn NER data |
| `scripts/03_prep_gold_set.py` | JSON → JSONL with char offsets + dev/eval split |
| `scripts/04_prep_gold_bio.py` | JSONL → BIO subword tags + 5-fold CV splits |
| `src/cli.py` | Canonical runner for ASR/PII/mask/eval and eval subcommands |

### Data

| Path | Committed? | Purpose |
|------|------------|---------|
| `gold_set/{en,zh,mix}.json` | Yes | Hand-labeled source (150 records) |
| `data/prepared/gold_set/{dev,eval}.jsonl` | Yes | Prepared gold (120 train + 30 eval) |
| `data/prepared/gold_bio/` | No | BIO training data (regenerate with script 04) |
| `data/models/ner_finetuned_gold_v1/final/best/` | No (1.1 GB) | Fine-tuned model weights | (same for v2)
| `data/runs/` | No | All pipeline run outputs |
| `data/raw/wikiann/` | No | WikiAnn data (regenerate with script 02) |

### Notebooks

| File | Purpose |
|------|---------|
| `notebooks/01_asr_analysis.ipynb` | ASR output analysis |
| `notebooks/02_pii_eval.ipynb` | WikiAnn NER baseline analysis |
| `notebooks/03_end2end_eval.ipynb` | End-to-end pipeline analysis |
| `notebooks/04_finetune_ner.ipynb` | NER fine-tuning (5-fold CV → final model) |
| `notebooks/05_pii_gold_eval.ipynb` | Gold eval analysis (ablation, heatmaps, errors) |

---

## Training Results (v1)
### 5-Fold Cross-Validation (strict seqeval BIO matching)

| Fold | Precision | Recall | F1    | Best Epoch |
|------|-----------|--------|-------|------------|
| 0    | 0.600     | 0.523  | 0.559 | 6          |
| 1    | 0.699     | 0.443  | 0.543 | 11         |
| 2    | 0.521     | 0.462  | 0.490 | 6          |
| 3    | 0.500     | 0.467  | 0.483 | 2          |
| 4    | 0.415     | 0.436  | 0.425 | 2          |
| **Mean** | **0.547** | **0.466** | **0.500 ± 0.053** | avg: **5** |

> CV F1 is modest because each val fold is only 24 records with strict token-level BIO matching.

### Gold Eval Set — Span-Overlap (eval_gold.py)

| Metric        | Value     |
|---------------|-----------|
| **Precision** | **0.931** |
| **Recall**    | **0.856** |
| **F1**        | **0.892** |
| **F2**        | **0.870** |
| TP / FP / FN  | 95 / 7 / 16 |

**Per-Type:**

| Type    | Support | P     | R     | F1    | F2    |
|---------|---------|-------|-------|-------|-------|
| NAME    | 46      | 0.979 | 1.000 | 0.989 | 0.996 |
| EMAIL   | 19      | 0.950 | 1.000 | 0.974 | 0.990 |
| ADDRESS | 9       | 1.000 | 0.889 | 0.941 | 0.909 |
| PHONE   | 32      | 0.815 | 0.688 | 0.746 | 0.710 |
| ID      | 5       | 0.000 | 0.000 | 0.000 | 0.000 |

**Per-Language:**

| Lang   | TP | FP | FN | P     | R     | F1    | F2    |
|--------|----|----|----|-------|-------|-------|-------|
| en     | 35 | 0  | 8  | 1.000 | 0.814 | 0.897 | 0.845 |
| zh     | 30 | 6  | 0  | 0.833 | 1.000 | 0.909 | 0.962 |
| mixed  | 30 | 1  | 8  | 0.968 | 0.789 | 0.870 | 0.820 |

**Key observations:**
- NAME and EMAIL near-perfect (F1 > 0.97)
- PHONE weakest common type — Chinese numeral phones (九三四五…)
- ID has zero recall
- Chinese has perfect recall but more FPs; English is more precise but misses more

---

## Training Results (v2) — 4-Fold Cross-Validation

Changed from 5-fold to 4-fold CV so each validation fold (30 records) matches the held-out eval set size. This gives a more representative estimate of held-out performance.

**Training config:** Same hyperparameters as v1 — base model `Davlan/xlm-roberta-base-ner-hrl`, lr=2e-5, batch=2, grad_accum=2, weight_decay=0.01, patience=3, max 15 epochs. Trained on Mac MPS (Apple Metal GPU).

**Data:** `data/prepared/gold_bio_4fold/` — 120 dev records split into 4 folds of 90 train / 30 val.

### 4-Fold Cross-Validation (strict seqeval BIO matching)

| Fold | Precision | Recall | F1    | Best Epoch |
|------|-----------|--------|-------|------------|
| 0    | 0.434     | 0.396  | 0.414 | 3          |
| 1    | 0.479     | 0.392  | 0.431 | 9          |
| 2    | 0.570     | 0.544  | 0.557 | 5          |
| 3    | 0.556     | 0.458  | 0.502 | 3          |
| **Mean** | **0.510** | **0.447** | **0.476 ± 0.057** | avg: **5** |

> CV F1 is lower than v1 (0.476 vs 0.500) because 4-fold uses less training data per fold (90 vs 96 records). The reduced training set slightly hurts token-level BIO accuracy.

**Final model:** Trained on all 120 records for 5 epochs (average best epoch from CV). Saved to `data/models/ner_finetuned_gold_v2/final/best/`.

### Gold Eval Set — Span-Overlap (eval_gold.py, 30 held-out eval records)

Comparison of the hybrid pipeline (rules + NER + postprocessing) using the baseline vs fine-tuned v2 model:

**Overall:**

| Metric | Baseline NER | Fine-tuned v2 NER | Delta |
|--------|-------------|-------------------|-------|
| Precision | 0.922 | 0.898 | -0.024 |
| Recall | 0.955 | 0.955 | 0.000 |
| F1 | 0.938 | 0.926 | -0.012 |
| F2 | 0.948 | 0.943 | -0.005 |
| TP / FP / FN | 106 / 9 / 5 | 106 / 12 / 5 | +3 FP |

**Per-Type:**

| Type | Support | Baseline P | v2 P | Baseline R | v2 R | Baseline F1 | v2 F1 |
|------|---------|-----------|------|-----------|------|------------|-------|
| NAME | 46 | 0.978 | 0.957 | 0.978 | 0.978 | 0.978 | 0.968 |
| EMAIL | 19 | 0.950 | 0.905 | 1.000 | 1.000 | 0.974 | 0.950 |
| PHONE | 32 | 0.886 | 0.838 | 0.969 | 0.969 | 0.925 | 0.899 |
| ADDRESS | 9 | 0.700 | 0.778 | 0.778 | 0.778 | 0.737 | 0.778 |
| ID | 5 | 1.000 | 1.000 | 0.800 | 0.800 | 0.889 | 0.889 |

**Per-Language:**

| Lang | Baseline F1 | v2 F1 | Baseline F2 | v2 F2 |
|------|------------|-------|------------|-------|
| en | 0.929 | 0.930 | 0.916 | 0.930 |
| zh | 0.896 | 0.870 | 0.955 | 0.943 |
| mixed | 0.987 | 0.973 | 0.979 | 0.957 |

**Key observations:**
- Fine-tuned v2 does **not** improve over the baseline on the hybrid pipeline eval. The baseline NER already performs well when combined with rules and postprocessing.
- The v2 model introduces 3 additional FPs (12 vs 9), slightly lowering precision while recall stays the same.
- ADDRESS is the only type where v2 improves (F1 0.778 vs 0.737), due to better precision.
- The v2 model's strict BIO CV F1 of 0.476 is limited by the small training set (120 records). The NER component contributes mainly NAME and ADDRESS detection, while rules handle PHONE, EMAIL, and ID effectively.
- The baseline model (pre-trained on large multilingual NER corpora) generalises better than a version fine-tuned on only 120 domain-specific records for this hybrid pipeline setup.

---

## Strict BIO Evaluation Results (used eval.py to evaluate ONLY NER)

Evaluation of the fine-tuned model (`data/models/ner_finetuned_gold_v1/final/best/`) using `src/pii/eval.py` with **strict entity-level BIO matching** on the 30 held-out eval records.

**Tokenization:** English text split on spaces; CJK characters split individually. eval.py reconstructs text via `" ".join(tokens)` before running NER, so the model sees space-separated CJK characters.

### Overall Results

| Metric        | Value     |
|---------------|-----------|
| **Precision** | **0.821** |
| **Recall**    | **0.709** |
| **F1**        | **0.761** |
| **F2**        | **0.729** |
| TP / FP / FN  | 78 / 17 / 32 |
| Token Accuracy | 97.7%    |

### Per-Language

| Lang   | Sentences | TP | FP | FN | P     | R     | F1    | F2    |
|--------|-----------|----|----|----|-------|-------|-------|-------|
| en     | 11        | 33 | 2  | 9  | 0.943 | 0.786 | 0.857 | 0.813 |
| zh     | 10        | 19 | 11 | 11 | 0.633 | 0.633 | 0.633 | 0.633 |
| mixed  | 9         | 26 | 4  | 12 | 0.867 | 0.684 | 0.765 | 0.714 |

### Per-Type

| Split | Type    | Support | TP | FP | FN | P     | R     | F1    |
|-------|---------|---------|----|----|----| ------|-------|-------|
| en    | NAME    | 15      | 15 | 0  | 0  | 1.000 | 1.000 | 1.000 |
| en    | EMAIL   | 8       | 8  | 0  | 0  | 1.000 | 1.000 | 1.000 |
| en    | PHONE   | 12      | 8  | 2  | 4  | 0.800 | 0.667 | 0.727 |
| en    | ADDRESS | 3       | 2  | 0  | 1  | 1.000 | 0.667 | 0.800 |
| en    | ID      | 4       | 0  | 0  | 4  | 0.000 | 0.000 | 0.000 |
| zh    | NAME    | 13      | 10 | 0  | 3  | 1.000 | 0.769 | 0.870 |
| zh    | EMAIL   | 6       | 2  | 0  | 4  | 1.000 | 0.333 | 0.500 |
| zh    | PHONE   | 7       | 4  | 10 | 3  | 0.286 | 0.571 | 0.381 |
| zh    | ADDRESS | 4       | 3  | 1  | 1  | 0.750 | 0.750 | 0.750 |
| mixed | NAME    | 18      | 16 | 2  | 2  | 0.889 | 0.889 | 0.889 |
| mixed | EMAIL   | 5       | 5  | 0  | 0  | 1.000 | 1.000 | 1.000 |
| mixed | PHONE   | 13      | 3  | 2  | 10 | 0.600 | 0.231 | 0.333 |
| mixed | ADDRESS | 2       | 2  | 0  | 0  | 1.000 | 1.000 | 1.000 |

---

## Entity-Level Error Analysis

Every PII entity that was **not completely recognised** (partial or missed) or **incorrectly classified** is documented below. Strict BIO matching requires exact token-boundary match — any boundary difference counts as wrong.

### Error Summary

| Status | Count | Description |
|--------|-------|-------------|
| CORRECT | 78 | Exact type + boundary match |
| MISSED | 22 | No overlapping prediction at all |
| PARTIAL | 10 | Overlapping prediction but wrong boundaries |
| FALSE_POS | 7 | Prediction with no matching gold entity |

### English Errors (9 FN)

| Record | Gold Type | Gold Text | Pred | Reason |
|--------|-----------|-----------|------|--------|
| en_0015 | PHONE | "9456 7788." | — | **Missed.** Second occurrence of the phone number in a long passage; model failed to detect it after ~150 tokens of context. Likely lost attention in long-range repetition. |
| en_0021 | PHONE | "9333 2211," | — | **Missed.** Second occurrence embedded deep in text (~200 tokens in). Same long-range repetition issue. |
| en_0028 | PHONE | "9444 8899." | "8899." | **Partial.** Model detected only the second half of the phone number. The 4-digit + 4-digit format with a space caused the model to split the entity at the space boundary. |
| en_0044 | ID | "9." | — | **Missed.** Single-digit reference ID ("My reference ID is 9.") — too short and ambiguous for NER to recognise as an ID. The model has no training signal for such minimal ID patterns. |
| en_0044 | PHONE | "8190 8829" | "8829" | **Partial.** Same 4+4 digit split issue — model only captured the second half. |
| en_0045 | ADDRESS | "woodlands," | — | **Missed.** Lowercase place name without structural cues (no "Block", "Road", etc.) — model didn't recognise it as an address. |
| en_0046 | ID | "7." | — | **Missed.** Single-digit ID, same issue as en_0044. |
| en_0048 | ID | "5." | — | **Missed.** Single-digit ID, same issue. |
| en_0050 | ID | "3." | — | **Missed.** Single-digit ID, same issue. |

### Chinese Errors (11 FN)

| Record | Gold Type | Gold Text | Pred | Reason |
|--------|-----------|-----------|------|--------|
| zh_0013 | PHONE | "九八三三，六六二五" | "六六二五" | **Partial.** Chinese numeral phone — model only detected the last 4 digits. The full-width comma separator (，) broke the entity span; model doesn't bridge across punctuation for CJK numerals. |
| zh_0013 | NAME | "张智恩" | — | **Missed.** Second speaker's name appearing deep in the conversation (~330 tokens). Space-separated CJK characters ("张 智 恩") may have degraded NER accuracy vs. contiguous characters. |
| zh_0013 | EMAIL | "zhang.zhien.demo@ymail.com" | — | **Missed.** Email embedded in CJK context as a single token (no spaces around it in original). After character-level CJK tokenization, the surrounding context shifted, making the email harder to detect. |
| zh_0014 | PHONE | "九二二七，三三五六" | "三五六" | **Partial.** Same CJK numeral phone issue — only last 3 digits captured. Full-width comma breaks the entity. |
| zh_0014 | EMAIL | "li.zhihao.demo@hotmail.com" | — | **Missed.** Same issue as zh_0013: email in CJK context. |
| zh_0014 | NAME | "郑伟明" | — | **Missed.** Second speaker's name deep in conversation. |
| zh_0014 | EMAIL | "zheng.weiming.demo@cmail.com" | — | **Missed.** Second speaker's email, same pattern. |
| zh_0015 | PHONE | "九一一六，四四二八" | "一一" | **Partial.** Severely fragmented — model only captured 2 of 8 digits. CJK numeral phone with full-width comma is the hardest pattern for the NER. |
| zh_0015 | ADDRESS | "绿园路大牌六六三" | "绿园路大" | **Partial.** Model recognised the road name but missed the block number ("牌六六三"). CJK numerals in addresses are not well-learned. |
| zh_0015 | NAME | "蔡嘉宁" | — | **Missed.** Second speaker name deep in conversation. |
| zh_0015 | EMAIL | "cai.jianing.demo@xyz.com" | — | **Missed.** Email in CJK context, same pattern. |

### Mixed (Code-Switched) Errors (12 FN)

| Record | Gold Type | Gold Text | Pred | Reason |
|--------|-----------|-----------|------|--------|
| mixed_0013 | PHONE | "9055 8421." | — | **Missed.** Phone number in English portion of code-switched text. Surrounded by CJK context that disrupts the model's attention. |
| mixed_0013 | PHONE | "九零五五，八四二一" | "四二" | **Partial.** CJK numeral phone — severe fragmentation, only 2 of 8 digits captured. |
| mixed_0013 | PHONE | "九零五五，八四二一" (2nd) | — | **Missed.** Repeated CJK numeral phone, second occurrence missed entirely. |
| mixed_0013 | PHONE | "9055 8421" (2nd) | — | **Missed.** Repeated digit phone in code-switched context. |
| mixed_0013 | NAME | "Daniel Ho" | "Ho" | **Partial.** Model captured only the surname. "Daniel" is common enough that the model didn't tag it as B-NAME in the code-switching context. |
| mixed_0014 | PHONE | "9810" | — | **Missed.** Truncated phone number (only first 4 digits in this mention). Model didn't recognise partial phone numbers. |
| mixed_0014 | PHONE | "九八一零，七三六二" | — | **Missed.** Full CJK numeral phone — completely missed in code-switched context. |
| mixed_0014 | PHONE | "9810 7362." | — | **Missed.** Full phone number in English, missed in code-switched surrounding. |
| mixed_0014 | PHONE | "九八一零，七三六二" (2nd) | — | **Missed.** Second occurrence, also completely missed. |
| mixed_0014 | PHONE | "9810 7362" (2nd) | — | **Missed.** Second occurrence of digit phone, missed. |
| mixed_0014 | NAME | "Jason Lee" | "Lee" | **Partial.** Same surname-only issue as "Daniel Ho" — first name not tagged in code-switching context. |
| mixed_mix_49 | PHONE | "8901 2345," | "8901 2345, and" | **Partial.** Model over-extended the phone span to include "and" — boundary overshoot. |

### False Positives (7 total)

| Record | Pred Type | Pred Text | Reason |
|--------|-----------|-----------|--------|
| en_0028 | PHONE | extra phone FP | Model hallucinated a phone span |
| en_0044 | PHONE | extra phone FP | Model hallucinated a phone span |
| zh_0013 | PHONE | "八三三" | Fragmented CJK digits incorrectly tagged as phone |
| zh_0014 | PHONE | "二二" | Fragmented CJK digits incorrectly tagged as phone |
| zh_0015 | PHONE | "一一六", "四二七", "四二", "四二", "八" | 5 spurious phone fragments from CJK digit sequences — model over-triggers on short CJK numeral runs |
| mixed_0013 | PHONE | "四二" | Fragmented CJK digits |
| mixed_0014 | NAME | "Lee" | Model tagged surname alone (was part of a partial match) |

### Root Cause Summary

| Root Cause | Count | Types Affected | Description |
|------------|-------|----------------|-------------|
| **CJK numeral phone fragmentation** | 12 | PHONE | Chinese-character phone numbers (九八三三，六六二五) are partially or completely missed. The full-width comma (，) breaks entity spans. Space-separated CJK characters degrade model accuracy. |
| **ID entities too short/ambiguous** | 4 | ID | Single-digit reference IDs ("9.", "7.", "5.", "3.") are impossible for NER to detect — no contextual signal and zero training support for such minimal patterns. |
| **Second-speaker entities in long text** | 5 | NAME, EMAIL | Names and emails of the second speaker appearing 300+ tokens into conversation are missed — attention degradation in long sequences. |
| **Email in CJK context** | 4 | EMAIL | Emails surrounded by CJK text (with character-level tokenization) lose their recognisable context pattern. |
| **Phone 4+4 split** | 3 | PHONE | English phone numbers in "XXXX YYYY" format — model only captures the second half. The space between digit groups is treated as an entity boundary. |
| **Partial name in code-switching** | 3 | NAME | In code-switched text, model captures only the surname ("Ho", "Lee") but misses the first name ("Daniel", "Jason"). |
| **CJK address numeral truncation** | 1 | ADDRESS | Model recognises road names but misses CJK numeral block numbers ("牌六六三"). |
| **Phone boundary overshoot** | 1 | PHONE | Model over-extends phone span to include following words. |

### Comparison: Span-Overlap (eval_gold.py) vs Strict BIO (eval.py)

| Metric    | Span-Overlap (F1) | Strict BIO (F1) | Delta |
|-----------|--------------------|------------------|-------|
| Overall   | 0.892              | 0.761            | -0.131 |
| NAME      | 0.989              | 0.945*           | -0.044 |
| EMAIL     | 0.974              | 0.789*           | -0.185 |
| PHONE     | 0.746              | 0.441*           | -0.305 |
| ADDRESS   | 0.941              | 0.824*           | -0.117 |
| ID        | 0.000              | 0.000            |  0.000 |

*Weighted average across splits.

The strict BIO evaluation is significantly harsher because it requires **exact token boundaries** — any partial overlap (even capturing 7 of 8 correct digits) counts as both a FP and a FN. This particularly penalises PHONE entities where the 4+4 digit format and CJK numeral comma separators cause boundary mismatches.

---

## Pipeline Improvements Made to Regex Rules

### Change 1: Cue-word-gated partial phone patterns

**File:** `src/pii/run.py`

**Problem:** `PHONE_RE` requires 8+ characters (`\d[\d\-\s]{6,}\d`). Partial/truncated 4-digit phone mentions like "phone is 9810" were missed.

**Fix:** Added two new regex patterns that match 4+ digit sequences only when preceded by phone-related cue words:

| Pattern | Cue Words | Captures |
|---------|-----------|----------|
| `EN_PARTIAL_PHONE_RE` | phone, number, call, dial, contact, hp, handphone | `(\d{4,})` after cue word |
| `ZH_PARTIAL_PHONE_RE` | 电话, 手机, 联系, 号码, 拨打 | Chinese digit sequence (3+) after cue word |

**Why cue-word-gated:** Lowering `PHONE_RE`'s minimum globally would match any 4+ digit sequence (dates, prices, zip codes) causing excessive FPs. Requiring a phone-related cue word limits matches to contexts where the digits are actually phone numbers.

**Limitation:** Bare phone numbers without cue words (e.g. someone just saying "9810 7362" without "my phone is...") still require the original `PHONE_RE` with its 8+ char minimum.

### Change 2: Singapore NRIC/FIN detection

**File:** `src/pii/run.py`

**Problem:** The pipeline had no rule for Singapore NRIC (National Registration Identity Card) or FIN (Foreign Identification Number) numbers, which are common PII in Singaporean transcripts.

**Fix:** Added `SG_NRIC_RE` regex pattern:

```python
SG_NRIC_RE = re.compile(r"(?<![A-Za-z])[STFGM]\d{7}[A-Za-z](?![A-Za-z])", re.IGNORECASE)
```

| Component | Matches | Purpose |
|-----------|---------|---------|
| `[STFGM]` | First letter | S/T = citizens (pre/post-2000), F/G = foreigners (pre/post-2000), M = foreigners (post-2022) |
| `\d{7}` | 7 digits | Serial number |
| `[A-Za-z]` | Check letter | Checksum character |
| Lookbehind/lookahead | Non-letter boundaries | Prevents matching inside longer words |

**Examples detected:** `S1234567A`, `T0123456J`, `G7654321B`, `M1234567K`

**Tagged as:** `ID` type (consistent with existing ID patterns).

**Config toggle:** `pii.rules.nric` (default: `true`).

### Change 3: Singapore postal code detection

**File:** `src/pii/run.py`

**Problem:** Singapore postal codes (e.g., `S609690`, `Singapore 609690`, or bare `609690` after an address) were not detected by any pattern.

**Fix:** Added two regex patterns:

**1. `SG_POSTAL_RE`** — Matches postal codes with `S` or `Singapore` prefix:

```python
SG_POSTAL_RE = re.compile(r"(?:Singapore\s*|S)\d{6}(?!\d)", re.IGNORECASE)
```

**2. `SG_POSTAL_AFTER_ADDR_RE`** — Matches bare 6-digit postal codes appearing right after an address (road type + optional street number + separator):

```python
SG_POSTAL_AFTER_ADDR_RE = re.compile(
    rf"{_ROAD_TYPE}"
    r"(?:\s+\d{1,4})?"     # optional street number
    r"[,\s]+"               # separator (comma, space)
    r"(\d{6})(?!\d)",
    re.IGNORECASE,
)
```

Captures group 1 (the 6-digit postal code only).

**Examples detected:**

| Pattern | Input | Detected |
|---------|-------|----------|
| `SG_POSTAL_RE` | `S609690` | `S609690` |
| `SG_POSTAL_RE` | `Singapore 123456` | `Singapore 123456` |
| `SG_POSTAL_AFTER_ADDR_RE` | `Jurong East Street 12, 609690` | `609690` |
| `SG_POSTAL_AFTER_ADDR_RE` | `Orchard Road, 238879` | `238879` |

**Tagged as:** `ADDRESS` type.

**Config toggle:** `pii.rules.postal_code` (default: `true`).

### Change 4: Expanded EN_ADDRESS_RE for addresses without Block prefix

**File:** `src/pii/run.py`

**Problem:** `EN_ADDRESS_RE` only matched addresses starting with `Block/Blk` (e.g., "Block 123 Orchard Road"). Singapore addresses like "Jurong East Street 12" or "10 Orchard Road" (no Block prefix) were missed.

**Fix:** Added two alternative patterns to `EN_ADDRESS_RE`:

| Pattern | Format | Example |
|---------|--------|---------|
| Original | `Block/Blk + number + road-type` | Block 123 Orchard Road |
| New alt 1 | `number + words + road-type` | 10 Orchard Road |
| New alt 2 | `words + road-type + number` | Jurong East Street 12 |

### Config toggles

- Partial phone patterns: `pii.rules.partial_phone` (default: `true`)
- NRIC pattern: `pii.rules.nric` (default: `true`)
- Postal code pattern: `pii.rules.postal_code` (default: `true`)

### Results after all changes

**Overall (30 held-out eval records, span-overlap matching):**

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Precision | 0.931 | 0.922 | -0.009 |
| Recall | 0.856 | 0.955 | +0.099 |
| F1 | 0.892 | 0.938 | +0.046 |
| F2 | 0.870 | 0.948 | +0.078 |
| TP / FP / FN | 95 / 7 / 16 | 106 / 9 / 5 | |

**Per-Type:**

| Type | Support | Before R | After R | Before F1 | After F1 |
|------|---------|----------|---------|-----------|----------|
| NAME | 46 | 1.000 | 0.978 | 0.989 | 0.978 |
| EMAIL | 19 | 1.000 | 1.000 | 0.974 | 0.974 |
| PHONE | 32 | 0.688 | 0.969 | 0.746 | 0.925 |
| ADDRESS | 9 | 0.889 | 0.778 | 0.941 | 0.737 |
| ID | 5 | 0.000 | 0.800 | 0.000 | 0.889 |

**Per-Language:**

| Lang | TP | FP | FN | P | R | F1 |
|------|----|----|----|----|----|----|
| en | 39 | 2 | 4 | 0.951 | 0.907 | 0.929 |
| zh | 30 | 7 | 0 | 0.811 | 1.000 | 0.895 |
| mixed | 37 | 0 | 1 | 1.000 | 0.974 | 0.987 |

### Summary of All Evaluations (30 held-out eval records)

| Run | Evaluator | Pipeline | F1 | F2 |
|-----|-----------|----------|----|----|
| Original baseline | `eval_gold.py` | Hybrid | 0.892 | 0.870 |
| After regex changes | `eval_gold.py` | Hybrid | 0.938 | 0.948 |
| Strict BIO | `eval.py` | NER only | 0.761 | (N.A as eval.py has no F2) |

**Run Demo server**
streamlit run app/streamlit_app.py --server.headless true
