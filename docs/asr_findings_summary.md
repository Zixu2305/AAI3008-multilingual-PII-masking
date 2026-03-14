# ASR Findings Summary (Mar 2026)

This summarizes the ASR work across:
- `notebooks/06_finetune_multilingual_whisper.ipynb`
- `notebooks/07_finetune_multilingual_whisper_full_audio_chunked.ipynb`
- `notebooks/08_infer_from_saved_checkpoint_colab_temp.ipynb`
- `c:\University\Y2T2\LLM\Project\inference_comparison\ft_full_transcription.txt`
- `c:\University\Y2T2\LLM\Project\inference_comparison\base_full_transcription.txt`

## What is working
- The fine-tuning pipeline runs end-to-end on the curated gold set and can export `best_checkpoint`.
- The inference notebook can load a saved checkpoint from Drive and transcribe long audio using chunked decoding.
- Per-language evaluation and visualization are available and produce consistent metrics.

## Main observed problems
1. Repeated phrases and loops
- Example patterns in the transcripts: repeated short phrases such as "Where is it?" and repeated fillers like "It is your typical...".
- Long runs of repeated tokens in Malay-like output (e.g., repeated "Bagaimana..." and repeated "Ia sangat jengis").
- Long repeated Chinese segments in the baseline transcript (e.g., repetitive Chinese lines).

2. Wrong language drift (Malay/Indonesian)
- Both the baseline and fine-tuned outputs sometimes drift to Malay/Indonesian.
- Mixed-language segments can be translated or normalized into a single language instead of preserving both scripts.

3. Under-transcription on long audio
- The 11-minute audio produced a shorter-than-expected transcript.
- This suggests chunking/decoding limits or generation truncation.

## Why repetition happens (root causes)
- **Chunked decoding without proper overlap merging**: repeated content can appear at chunk boundaries.
- **Decode loop / low confidence**: Whisper can fall into repetition when the audio is low SNR, non-speech, or unclear; beam search can reinforce loops.
- **No-speech control**: without explicit no-speech handling, the model may hallucinate or repeat when silence/background occurs.
- **Training data mismatch**: gold-set transcripts are short call-style utterances; the 11-minute YouTube-style audio has different conversational structure.
- **Chunk transcript alignment** (full-audio notebook): proportional text slicing creates weak alignment and can reinforce repetition.

## Why the model outputs other languages
- Whisper is multilingual and tends to switch languages when confidence is low.
- Language control is prompt-based, not a hard constraint. Even with forced decoder IDs, drift can happen mid-utterance.
- Mixed-language audio plus code-switching increases uncertainty. The model may pick a high-probability language (Malay/Indonesian) with similar phonetics.

## Why output is too short
- `MAX_DECODE_LEN` or chunk-level decode truncation can cap text.
- Short chunk lengths + too-strict decode length can under-generate.
- Long stretches of silence or music can reduce effective text content.

## Findings from the transcripts
- Fine-tuned transcript still contains Malay-like segments and repetitions, but has more English continuity at the start.
- Baseline transcript shows stronger repetition and longer nonsensical loops.
- Both struggle on long-form content due to domain mismatch and chunking/decoding behavior.

## What to try next (prioritized)
1. Improve segmentation and silence handling
- Use VAD (voice activity detection) to split on speech boundaries instead of fixed time windows.
- Add overlap + merge logic to avoid duplicated boundary text.
- Drop or skip segments dominated by silence/music.

2. Tighten decoding controls
- Force language per chunk when known (e.g., `en` or `zh`), and for mixed content decode both `en` and `zh` then select with a script-preserving heuristic.
- Add repetition penalties (e.g., `repetition_penalty`, `no_repeat_ngram_size`) when using `generate()`.
- Increase `max_length` to avoid truncation on longer chunks.

3. Use stronger or domain-appropriate ASR models
- Try larger Whisper checkpoints: `openai/whisper-medium`, `openai/whisper-large-v3`.
- Try distilled Whisper variants for speed but keep quality high.
- Consider WhisperX (Whisper + alignment) to improve segment boundaries and reduce repeats.

4. Data improvements for fine-tuning
- Expand training data with long-form conversational audio similar to the target domain.
- Add chunk-level aligned transcripts (timestamps or segments), not just full-clip text.
- Remove duplicates and correct malformed mixed IDs and encoding issues.

5. Evaluate in a consistent way
- Always compare fine-tuned output vs baseline on the same audio using the same chunking and decode settings.
- If a reference transcript exists, compute WER/CER to quantify improvements.

## Recommended next steps for PII accuracy
- Use higher-quality ASR output (reduce repetition and language drift) before PII extraction.
- If PII is mostly in English/Chinese, force language decoding per segment to keep scripts stable.
- Consider post-ASR cleanup: remove repeated lines and collapse duplicate fragments before PII detection.

## Notebooks status
- `notebooks/06_finetune_multilingual_whisper.ipynb` now saves only `best_checkpoint` to Drive artifacts.
- `notebooks/07_finetune_multilingual_whisper_full_audio_chunked.ipynb` uses full-audio chunking for training/eval.
- `notebooks/08_infer_from_saved_checkpoint_colab_temp.ipynb` loads checkpoints from Drive and has baseline-vs-finetuned comparison.

If you want, I can add a dedicated VAD + overlap-merging inference notebook or implement repetition-penalized decoding in `08`.
