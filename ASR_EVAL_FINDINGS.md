# ASR Evaluation Findings — Three-Model Comparison

**Notebook:** `09_asr_eval_three_model_comparison.ipynb`  
**Models:** `whisper-small`, `whisper-medium`, `whisper-large-v3` (all fine-tuned)  
**Test set:** 1,118 segments — 682 English (SingaVoice, SESSION0/1) · 436 Chinese (IMDA NSC, G0001/G0002)  
**Hardware:** T4 GPU, float16

---

## Overall Results

| Model | EN WER ↓ | EN CER ↓ | ZH WER ↓ | ZH CER ↓ | MER ↓ | RTF ↓ |
|---|---|---|---|---|---|---|
| **whisper-large-v3** | **15.52%** | **3.90%** | **105.50%** | **271.05%** | **41.89%** | 0.381 |
| whisper-medium | 23.22% | 11.30% | 436.01% | 1272.02% | 51.25% | 0.301 |
| whisper-small | 65.85% | 65.18% | 254.13% | 579.23% | 63.56% | **0.146** |

> WER above 100% occurs when decoder insertions substantially exceed the reference length — a known failure mode on very short segments under incorrect language conditions.

---

## Per-Session Breakdown

| Session | Language | small | medium | large-v3 | Metric |
|---|---|---|---|---|---|
| SESSION0 | English | 73.30% | 23.85% | 15.58% | WER |
| SESSION1 | English | 57.86% | 22.54% | 15.46% | WER |
| G0001 | Chinese | 433.33% | 623.39% | 113.45% | WER |
| G0002 | Chinese | 138.49% | 315.09% | 100.38% | WER |

---

## Error Type Totals (all 1,118 segments)

| Model | Hits | Substitutions | Deletions | Insertions |
|---|---|---|---|---|
| whisper-large-v3 | 7,366 | 1,498 | 90 | 121 |
| whisper-medium | 6,627 | 2,113 | 214 | 1,529 |
| whisper-small | 4,667 | 3,696 | 591 | 2,531 |

**large-v3** has the fewest errors across all categories. **whisper-small** produces the most insertions (2,531) and deletions (591), consistent with the hallucination loops observed in qualitative review.

---

## Key Findings

### English
- `whisper-large-v3` achieves **15.5% WER** on Singaporean accented read speech — a reasonable baseline indicating the English fine-tuning was moderately successful.
- `whisper-small` hallucinates in **Bahasa Melayu** on ~5–10% of English utterances, generating repetitive loops (e.g. `"sepatutnya sepatutnya sepatutnya..."`). This affects utterances containing uncommon English words or Singaporean place names.
- All three models agree on easy utterances. Only 8/1,118 segments have WER ≤ 5% across all models; 386/1,118 have WER ≥ 30% for all models.

### Chinese
- All three models struggle significantly with the conversational IMDA NSC data. Even the best model (`large-v3`) achieves a CER of **271%**.
- The dominant failure mode is **language switching**: models output English instead of Chinese for short backchannel segments (e.g. REF: `嗯` → PRED: `"Thank you"` or `"Mm-hmm"`).
- `whisper-medium` and `whisper-small` enter **infinite repetition loops** on some segments, generating hundreds of repetitions of a single phrase — a known Whisper failure when the language token or temperature is misconfigured.
- `whisper-small` occasionally hallucinates in Japanese, Korean, or Cyrillic for Chinese input, confusing short Mandarin backchannel sounds for phonetically similar content in other languages.
- `whisper-small` sometimes outputs **Traditional Chinese** characters for Simplified Chinese references, inflating CER even when the word was correctly recognised.
- Many Chinese segments are 1–3 characters (backchannels: `嗯`, `对`, `呃`). A single incorrect character on a 1-character reference produces CER of several hundred to several thousand percent for repetition loops, disproportionately inflating aggregate CER figures.
- Despite high aggregate CER, `large-v3` correctly transcribes most longer utterances (10+ characters), with CER often below 20% on content-rich turns.

### Counterintuitive: `whisper-medium` is worse than `whisper-small` on Chinese
`whisper-medium` (CER 1,272%) performs far worse on Chinese than `whisper-small` (CER 579%), despite being a larger model. This is likely a consequence of the dual-language fine-tuning configuration: the medium checkpoint used `force_lang=False` (auto language detection), which may have caused it to default to English decoding more aggressively on ambiguous Chinese inputs compared to the smaller model.

---

## Implications for Fine-Tuning

- **Model scale is the dominant factor.** `large-v3` outperforms both smaller checkpoints on every metric. For a Singapore English/Chinese ASR deployment, the larger model is justified if latency allows (RTF 0.38 on T4).
- **English fine-tuning generalised reasonably well.** The 15.5% WER on accented Singaporean read speech is a usable baseline.
- **Chinese fine-tuning has not solved the core problem.** The conversational NSC data introduces short backchannel segments that trigger language confusion in all models, and the dual mixed training strategy has not fully prevented cross-language hallucination.
- **The `force_lang=False` configuration in `whisper-medium` appears harmful for Chinese.** The lack of an explicit language token at decode time makes the model more susceptible to defaulting to English on ambiguous short segments.

---

## Recommended Next Steps

1. **Force language token at decode time** (`<|zh|>`) for Chinese segments to prevent English hallucination loops.
2. **Filter or re-weight short backchannel segments** during training — their disproportionate CER contribution masks genuine improvements on content-rich utterances.
3. **Evaluate with a minimum segment length** (e.g. ≥ 0.5s, ≥ 3 characters) to get a cleaner picture of model capability on substantive speech.
4. **Investigate `force_lang=True` vs. `force_lang=False`** on Chinese more carefully — the medium model result suggests auto-detection is unreliable for short Mandarin segments.
5. **Expand Chinese test coverage** beyond two speakers of the same conversation; broader NSC sessions would give a more reliable generalisation estimate.

---

*Outputs saved to `MyDrive/LLM/eval_three_models/` — CSV, JSON summary, and 16 visualisation plots.*
