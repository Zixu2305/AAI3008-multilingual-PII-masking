from __future__ import annotations

import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure `src` package imports resolve even if launched from `app/`.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.asr.run import run_asr
from src.eval.run import run_eval
from src.mask.run import run_mask
from src.pii.run import run_pii
from src.utils.io import PROJECT_ROOT, read_jsonl


BASELINE_NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
LOCAL_FINETUNED_NER_MODEL = "data/models/ner_finetuned_gold_v1/final/best"
REMOTE_FINETUNED_NER_MODEL = "ImShooShoo/ner_finetuned_gold_roberta"
BASELINE_ASR_MODEL = "large-v3"
LOCAL_FINETUNED_ASR_MODEL = "data/models/whisper-large-v3/ct2"
REMOTE_FINETUNED_ASR_MODEL = "ImShooShoo/whisper-large-v3"
HF_CACHE_ROOT = Path("data") / "cache" / "hf"
HF_ASR_CACHE_DIR = str(HF_CACHE_ROOT / "asr")
HF_NER_CACHE_DIR = str(HF_CACHE_ROOT / "ner")
HF_TOKEN_ENV = "HF_TOKEN"
BASELINE_NER_DEVICE = "cpu"
DEFAULT_RULES = {
    "phone": True,
    "email": True,
    "zh_phone": True,
    "en_spoken_phone": True,
    "partial_phone": True,
    "id_number": True,
    "generic_id": True,
    "address": True,
    "nric": True,
    "postal_code": True,
}
DEFAULT_LABEL_MAP = {
    "PER": "NAME",
    "PERSON": "NAME",
    "LOC": "ADDRESS",
    "GPE": "ADDRESS",
    "ORG": "ORG",
}


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _safe_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "uploaded_audio"


def _guess_audio_suffix(input_audio: Any) -> str:
    name = str(getattr(input_audio, "name", "") or "").strip()
    suffix = Path(name).suffix
    if suffix:
        return suffix.lower()

    content_type = str(getattr(input_audio, "type", "") or "").lower()
    if "wav" in content_type:
        return ".wav"
    if "mp3" in content_type or "mpeg" in content_type:
        return ".mp3"
    if "ogg" in content_type:
        return ".ogg"
    if "webm" in content_type:
        return ".webm"
    if "m4a" in content_type or "mp4" in content_type:
        return ".m4a"
    return ".wav"


def _snapshot_audio_input(input_audio: Any, fallback_name: str) -> dict[str, Any]:
    raw_name = str(getattr(input_audio, "name", "") or "").strip() or fallback_name
    suffix = _guess_audio_suffix(input_audio)
    content_type = str(getattr(input_audio, "type", "") or "").strip() or None

    if isinstance(input_audio, dict) and "payload" in input_audio:
        payload = bytes(input_audio.get("payload") or b"")
        raw_name = str(input_audio.get("name") or raw_name).strip() or raw_name
        suffix = str(input_audio.get("suffix") or suffix).strip() or suffix
        content_type = str(input_audio.get("content_type") or content_type or "").strip() or None
    elif hasattr(input_audio, "getbuffer"):
        payload = bytes(input_audio.getbuffer())
    elif hasattr(input_audio, "getvalue"):
        payload = bytes(input_audio.getvalue())
    elif hasattr(input_audio, "read"):
        if hasattr(input_audio, "seek"):
            try:
                input_audio.seek(0)
            except Exception:
                pass
        payload = input_audio.read()
    else:
        raise TypeError("Unsupported audio input object; expected UploadedFile-like input.")

    if not payload:
        raise ValueError("Audio input is empty.")

    return {
        "payload": payload,
        "name": raw_name,
        "suffix": suffix,
        "content_type": content_type,
    }


def _seed_hf_token_from_secrets() -> None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "huggingface_token"):
        try:
            raw = st.secrets.get(key)
        except Exception:
            raw = None
        token = str(raw or "").strip()
        if token:
            os.environ.setdefault(HF_TOKEN_ENV, token)
            return


def _get_request_host_and_proto() -> tuple[str | None, str | None]:
    try:
        headers = dict(st.context.headers)
    except Exception:
        return None, None

    host = str(headers.get("X-Forwarded-Host") or headers.get("Host") or "").strip().lower() or None
    proto = str(headers.get("X-Forwarded-Proto") or headers.get("X-Scheme") or "").strip().lower() or None
    return host, proto


def _mic_context_message() -> tuple[str, str]:
    host, proto = _get_request_host_and_proto()
    if proto == "https":
        return "success", f"Microphone context looks valid: `{host or 'https'}`"
    if host:
        hostname = host.split(":", 1)[0]
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            return "success", f"Microphone context looks valid: `{host}`"
        return (
            "warning",
            "Microphone recording usually only works on `https` or `localhost`. "
            f"Current host looks like `{host}`. If recording does not start, open the app via "
            "`http://localhost:8501` or use an SSH tunnel.",
        )
    return (
        "info",
        "Microphone recording requires a browser-secure context: `https` or `localhost`.",
    )


def _local_asr_exists() -> bool:
    return (PROJECT_ROOT / LOCAL_FINETUNED_ASR_MODEL / "model.bin").exists()


def _local_ner_exists() -> bool:
    return (PROJECT_ROOT / LOCAL_FINETUNED_NER_MODEL / "model.safetensors").exists()


def _resolve_asr_model_name() -> tuple[str, str]:
    if _local_asr_exists():
        return LOCAL_FINETUNED_ASR_MODEL, "local fine-tuned"
    if REMOTE_FINETUNED_ASR_MODEL:
        return REMOTE_FINETUNED_ASR_MODEL, "huggingface fine-tuned"
    return BASELINE_ASR_MODEL, "baseline"


def _resolve_ner_model_name() -> tuple[str, str]:
    if _local_ner_exists():
        return LOCAL_FINETUNED_NER_MODEL, "local fine-tuned"
    if REMOTE_FINETUNED_NER_MODEL:
        return REMOTE_FINETUNED_NER_MODEL, "huggingface fine-tuned"
    return BASELINE_NER_MODEL, "baseline"


def _ner_model_caption(model_name: str) -> str:
    if model_name == LOCAL_FINETUNED_NER_MODEL:
        return "finetuned_gold_v1 (local)"
    if model_name == REMOTE_FINETUNED_NER_MODEL:
        return "finetuned_gold_v1 (huggingface)"
    return model_name


def _asr_model_caption(model_name: str) -> str:
    if model_name == LOCAL_FINETUNED_ASR_MODEL:
        return "whisper-large-v3 finetuned (local ct2)"
    if model_name == REMOTE_FINETUNED_ASR_MODEL:
        return "whisper-large-v3 finetuned (huggingface ct2)"
    return model_name


def _write_input_audio(input_audio: Any, run_tag: str, fallback_name: str) -> tuple[Path, str]:
    snapshot = _snapshot_audio_input(input_audio, fallback_name)
    raw_name = str(snapshot.get("name") or fallback_name).strip() or fallback_name
    suffix = str(snapshot.get("suffix") or _guess_audio_suffix(input_audio)).strip() or ".wav"
    safe_name = _safe_slug(Path(raw_name).stem)
    rel_dir = Path("data") / "uploads" / "streamlit" / run_tag
    abs_dir = PROJECT_ROOT / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    audio_rel = rel_dir / f"{safe_name}{suffix}"
    audio_abs = PROJECT_ROOT / audio_rel
    payload = bytes(snapshot.get("payload") or b"")
    audio_abs.write_bytes(payload)
    return audio_abs, str(audio_rel)


def _to_project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _bundle_artifacts_zip(
    *,
    run_tag: str,
    source_audio_abs: Path,
    asr_out: Path,
    pii_out: Path,
    masked_out: Path,
    eval_out: Path,
    asr_cfg: dict[str, Any],
    pii_cfg: dict[str, Any],
    mask_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
) -> Path:
    package_dir = PROJECT_ROOT / "data" / "runs" / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    zip_path = package_dir / f"pipeline_artifacts_{run_tag}.zip"

    stage_dirs = [Path(asr_out), Path(pii_out), Path(masked_out), Path(eval_out)]
    manifest = {
        "tag": run_tag,
        "created_at": datetime.now().isoformat(),
        "source_audio": _to_project_relative(source_audio_abs),
        "artifact_dirs": [_to_project_relative(p) for p in stage_dirs],
        "configs": {
            "asr": asr_cfg,
            "pii": pii_cfg,
            "mask": mask_cfg,
            "eval": eval_cfg,
        },
    }

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if source_audio_abs.exists():
            zf.write(source_audio_abs, arcname=_to_project_relative(source_audio_abs))

        for stage_dir in stage_dirs:
            if not stage_dir.exists():
                continue
            for file_path in sorted(stage_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=_to_project_relative(file_path))

        zf.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return zip_path


def _write_manifest(audio_rel_path: str, run_tag: str) -> str:
    rel_manifest = Path("data") / "uploads" / "streamlit" / run_tag / "manifest.jsonl"
    abs_manifest = PROJECT_ROOT / rel_manifest
    row = {
        "item_id": f"{run_tag}_item_0001",
        "audio_path": audio_rel_path,
    }
    abs_manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(rel_manifest)


def _build_asr_cfg(
    run_id: str,
    manifest_path: str,
    device: str,
    model_name: str,
    language: str,
    mixed_language_enabled: bool,
) -> dict[str, Any]:
    compute_type = "float16" if device.startswith("cuda") else "float32"
    fallback_model = BASELINE_ASR_MODEL if model_name != BASELINE_ASR_MODEL else None
    return {
        "run": {
            "run_id": run_id,
            "resume": False,
            "checkpoint_every_files": 1,
            "progress_every_files": 1,
            "write_progress": True,
        },
        "data": {
            "source_type": "manifest",
            "max_items": 0,
            "manifest": {"path": manifest_path},
        },
        "asr": {
            "engine": "faster_whisper",
            "model_name": model_name,
            "fallback_model_name": fallback_model,
            "device": device,
            "compute_type": compute_type,
            "download_root": HF_ASR_CACHE_DIR,
            "local_files_only": False,
            "hf_token_env": HF_TOKEN_ENV,
            "language": language,
            "max_segment_sec": 12.0,
            "segment_overlap_sec": 0.8,
            "word_timestamps": True,
            "vad_filter": True,
            "beam_size": 5,
            "condition_on_previous_text": False,
            "mixed_language": {
                "enabled": bool(mixed_language_enabled),
                "languages": ["en", "zh"],
                "fallback_language": "en",
            },
        },
    }


def _build_pii_cfg(run_id: str, asr_run_id: str, device: str) -> dict[str, Any]:
    ner_model_name, _ = _resolve_ner_model_name()
    fallback_model = BASELINE_NER_MODEL if ner_model_name != BASELINE_NER_MODEL else None
    return {
        "run": {
            "run_id": run_id,
            "input_asr_dir": f"data/runs/asr/{asr_run_id}",
        },
        "pii": {
            "enable_rules": True,
            "enable_ner": True,
            "enable_llm": False,
            "rules": dict(DEFAULT_RULES),
            "ner": {
                "model_name": ner_model_name,
                "fallback_model_name": fallback_model,
                "device": device,
                "cache_dir": HF_NER_CACHE_DIR,
                "local_files_only": False,
                "hf_token_env": HF_TOKEN_ENV,
                "max_length": 256,
                "score_threshold": 0.0,
                "label_map": dict(DEFAULT_LABEL_MAP),
            },
        },
        "postprocess": {
            "merge_gap_chars": 1,
            "min_span_chars": 1,
        },
    }


def _build_mask_cfg(run_id: str, pii_run_id: str) -> dict[str, Any]:
    return {
        "run": {
            "run_id": run_id,
            "input_pii_dir": f"data/runs/pii/{pii_run_id}",
        },
        "mask": {
            "mode": "silence",
            "lead_sec": 0.00,
            "tail_sec": 0.15,
            "min_interval_sec": 0.20,
            "merge_gap_sec": 0.05,
            "beep_freq_hz": 1000,
            "beep_gain_db": -6.0,
            "beep_amplitude": 0.20,
        },
    }


def _build_eval_cfg(run_id: str, asr_run_id: str, pii_run_id: str, masked_run_id: str) -> dict[str, Any]:
    return {
        "run": {
            "run_id": run_id,
        },
        "inputs": {
            "asr_dir": f"data/runs/asr/{asr_run_id}",
            "pii_dir": f"data/runs/pii/{pii_run_id}",
            "masked_dir": f"data/runs/masked/{masked_run_id}",
            "gold_dir": "data/gold",
        },
    }


def _redact_text(text: str, spans: list[dict[str, Any]]) -> str:
    if not text:
        return ""
    if not spans:
        return text

    out: list[str] = []
    cursor = 0
    ordered = sorted(spans, key=lambda s: (int(s.get("start", -1)), int(s.get("end", -1))))
    for span in ordered:
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        pii_type = str(span.get("type", "PII")).upper()
        if start < 0 or end <= start:
            continue
        if start < cursor:
            continue
        if start > len(text):
            continue
        end = min(end, len(text))
        out.append(text[cursor:start])
        out.append(f"[{pii_type}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _segments_table(asr_segments: list[dict[str, Any]], pii_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans_by_segment = {str(row.get("segment_id")): row.get("spans", []) for row in pii_rows}
    rows: list[dict[str, Any]] = []

    for seg in asr_segments:
        seg_id = str(seg.get("segment_id"))
        text = str(seg.get("text", ""))
        spans = spans_by_segment.get(seg_id, [])
        rows.append(
            {
                "segment_id": seg_id,
                "language": seg.get("language", "unknown"),
                "start": round(float(seg.get("start", 0.0) or 0.0), 2),
                "end": None if seg.get("end") is None else round(float(seg.get("end", 0.0)), 2),
                "detected_spans": len(spans),
                "transcript": text,
                "redacted_transcript": _redact_text(text, spans),
            }
        )

    return rows


def _render_run_ids(asr_id: str, pii_id: str, masked_id: str, eval_id: str) -> None:
    st.markdown("### Run IDs")
    st.code(
        "\n".join(
            [
                f"ASR:    {asr_id}",
                f"PII:    {pii_id}",
                f"MASKED: {masked_id}",
                f"EVAL:   {eval_id}",
            ]
        ),
        language="text",
    )


def main() -> None:
    st.set_page_config(page_title="PII Audio Pipeline", layout="wide")
    _seed_hf_token_from_secrets()

    st.markdown(
        """
        <style>
          :root {
            --bg-soft: #f2f8f6;
            --ink: #16332f;
            --accent: #0f766e;
            --accent-soft: #d8f2ed;
            --warn-soft: #fff4db;
          }
          .block-container {padding-top: 1.5rem;}
          .hero {background: linear-gradient(120deg, var(--bg-soft), #fff); border: 1px solid #d2e6df; padding: 1rem 1.2rem; border-radius: 12px; margin-bottom: 1rem;}
          .hero h1 {color: var(--ink); margin: 0 0 .2rem 0; font-size: 1.7rem;}
          .hero p {margin: 0; color: #2f524d;}
          .pill {display: inline-block; background: var(--accent-soft); color: var(--accent); border-radius: 999px; padding: .2rem .6rem; margin-right: .35rem; font-size: .78rem;}
        </style>
        <div class="hero">
          <h1>End-to-End Audio PII Pipeline</h1>
          <p>Upload or record audio and run <b>ASR -> PII -> Mask -> Eval</b> with rules plus NER. The app prefers local fine-tuned checkpoints, otherwise it downloads the Hugging Face repos and falls back to baseline only if needed.</p>
          <div style="margin-top:.55rem;">
            <span class="pill">ASR: local -> Hugging Face -> baseline</span>
            <span class="pill">NER: local -> Hugging Face -> baseline</span>
            <span class="pill">Rules + NER (LLM off)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Pipeline Defaults")

        # Keep UI simple: fixed baseline models with limited ASR language controls.
        asr_device = "cuda" if _has_cuda() else "cpu"
        ner_device = BASELINE_NER_DEVICE
        asr_model, asr_source = _resolve_asr_model_name()
        ner_model, ner_source = _resolve_ner_model_name()
        asr_language = st.selectbox("ASR language", options=["auto", "en", "zh"], index=0)
        mixed_language_enabled = st.checkbox(
            "Mixed-language re-decode (en/zh)",
            value=(asr_language == "auto"),
            disabled=(asr_language != "auto"),
        )
        if asr_language != "auto":
            mixed_language_enabled = False

        st.write(f"ASR device: `{asr_device}`")
        st.write(f"NER device: `{ner_device}`")
        st.write(f"ASR model: `{_asr_model_caption(asr_model)}`")
        st.write(f"NER model: `{_ner_model_caption(ner_model)}`")
        st.write(f"ASR source preference: `{asr_source}`")
        st.write(f"NER source preference: `{ner_source}`")
        st.write(f"ASR language: `{asr_language}`")
        st.write(f"Mixed-language re-decode: `{mixed_language_enabled}`")
        st.write(f"HF cache root: `{HF_CACHE_ROOT}`")
        package_artifacts = st.checkbox("Package artifacts as zip", value=True)

        st.markdown("### Note")
        st.info(
            "Masking stage uses time-aligned span masking. "
            "If audio backend dependencies are unavailable, it falls back to copying source audio and logs the reason."
        )

    input_mode = st.radio(
        "Audio source",
        options=["Upload file", "Record from microphone"],
        index=0,
        horizontal=True,
    )
    uploaded_file = None
    recorded_audio = None
    selected_audio: Any | None = None
    selected_audio_name = "input_audio.wav"
    selected_audio_snapshot: dict[str, Any] | None = None

    if input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload one audio file",
            type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "webm"],
            accept_multiple_files=False,
            key="upload_audio_file",
        )
        selected_audio = uploaded_file
        if uploaded_file is not None:
            selected_audio_name = str(getattr(uploaded_file, "name", "uploaded_audio.wav"))
            try:
                selected_audio_snapshot = _snapshot_audio_input(uploaded_file, selected_audio_name)
            except Exception as exc:
                st.error(f"Unable to read uploaded audio: {exc}")
                return
    else:
        mic_status, mic_message = _mic_context_message()
        if mic_status == "success":
            st.success(mic_message)
        elif mic_status == "warning":
            st.warning(mic_message)
        else:
            st.info(mic_message)
        st.caption("If the browser does not show a permission prompt, try Chrome/Edge and open the app on `localhost`.")
        recorded_audio = st.audio_input("Record audio from microphone", key="mic_audio_input")
        selected_audio_name = "recorded_audio.wav"
        if recorded_audio is not None:
            try:
                st.session_state["recorded_audio_snapshot"] = _snapshot_audio_input(recorded_audio, selected_audio_name)
            except Exception as exc:
                st.error(f"Unable to read recorded audio: {exc}")
                return

        cached_recording = st.session_state.get("recorded_audio_snapshot")
        if cached_recording:
            selected_audio_snapshot = dict(cached_recording)
            selected_audio = selected_audio_snapshot
            selected_audio_name = str(selected_audio_snapshot.get("name") or selected_audio_name)
            st.caption("Recorded audio preview")
            st.audio(
                selected_audio_snapshot["payload"],
                format=str(selected_audio_snapshot.get("content_type") or "audio/wav"),
            )
            if recorded_audio is None:
                st.caption("Using the latest recorded audio stored in this session.")

    run_clicked = st.button("Run End-to-End Pipeline", type="primary", use_container_width=True)

    if not run_clicked:
        return

    if selected_audio is None:
        if input_mode == "Upload file":
            st.error("Upload an audio file first.")
        else:
            st.error("Record audio first.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"st_{timestamp}"
    asr_run_id = f"asr_{tag}"
    pii_run_id = f"pii_{tag}"
    masked_run_id = f"masked_{tag}"
    eval_run_id = f"eval_{tag}"

    start = time.perf_counter()
    bundle_zip_path: Path | None = None

    try:
        audio_input_obj = selected_audio_snapshot if selected_audio_snapshot is not None else selected_audio
        audio_abs, audio_rel = _write_input_audio(audio_input_obj, tag, selected_audio_name)
        manifest_rel = _write_manifest(audio_rel, tag)

        asr_cfg = _build_asr_cfg(
            asr_run_id,
            manifest_rel,
            asr_device,
            asr_model,
            asr_language,
            mixed_language_enabled,
        )
        pii_cfg = _build_pii_cfg(pii_run_id, asr_run_id, ner_device)
        mask_cfg = _build_mask_cfg(masked_run_id, pii_run_id)
        eval_cfg = _build_eval_cfg(eval_run_id, asr_run_id, pii_run_id, masked_run_id)

        with st.spinner("Stage 1/4: Running ASR..."):
            asr_out = run_asr(asr_cfg)
        with st.spinner("Stage 2/4: Detecting PII..."):
            pii_out = run_pii(pii_cfg)
        with st.spinner("Stage 3/4: Applying mask stage..."):
            masked_out = run_mask(mask_cfg)
        with st.spinner("Stage 4/4: Building eval summary..."):
            eval_out = run_eval(eval_cfg)
        if package_artifacts:
            with st.spinner("Packaging run artifacts..."):
                bundle_zip_path = _bundle_artifacts_zip(
                    run_tag=tag,
                    source_audio_abs=audio_abs,
                    asr_out=Path(asr_out),
                    pii_out=Path(pii_out),
                    masked_out=Path(masked_out),
                    eval_out=Path(eval_out),
                    asr_cfg=asr_cfg,
                    pii_cfg=pii_cfg,
                    mask_cfg=mask_cfg,
                    eval_cfg=eval_cfg,
                )

    except Exception as exc:
        st.exception(exc)
        return

    elapsed = round(time.perf_counter() - start, 2)
    st.success(f"Pipeline completed in {elapsed} seconds.")
    _render_run_ids(asr_run_id, pii_run_id, masked_run_id, eval_run_id)
    if bundle_zip_path is not None and bundle_zip_path.exists():
        st.markdown("### Artifact Package")
        st.caption("Download all pipeline artifacts (ASR, PII, masked, eval) as one zip.")
        zip_bytes = bundle_zip_path.read_bytes()
        st.download_button(
            "Download Artifact ZIP",
            data=zip_bytes,
            file_name=bundle_zip_path.name,
            mime="application/zip",
            use_container_width=True,
        )
        st.code(_to_project_relative(bundle_zip_path), language="text")

    asr_segments = read_jsonl(Path(asr_out) / "segments.jsonl")
    pii_rows = read_jsonl(Path(pii_out) / "spans.jsonl")
    mask_audit = read_jsonl(Path(masked_out) / "audit.jsonl")
    asr_meta = _load_summary(Path(asr_out) / "run_meta.json")
    eval_summary = _load_summary(Path(eval_out) / "summary.json")
    pii_meta = _load_summary(Path(pii_out) / "run_meta.json")

    st.markdown("### Model Resolution")
    st.code(
        "\n".join(
            [
                f"ASR requested: {asr_meta.get('model_name_requested', asr_cfg['asr'].get('model_name'))}",
                f"ASR used:      {asr_meta.get('model_name', asr_cfg['asr'].get('model_name'))}",
                f"NER requested: {pii_meta.get('ner_model_name_requested', pii_cfg['pii']['ner'].get('model_name'))}",
                f"NER used:      {pii_meta.get('ner_model_name', pii_cfg['pii']['ner'].get('model_name'))}",
            ]
        ),
        language="text",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ASR segments", len(asr_segments))
    col2.metric("Rows with spans", len(pii_rows))
    col3.metric("Detected PII spans", int(pii_meta.get("total_detected_spans", 0)))
    col4.metric("Audio files processed", len(mask_audit))

    fallback_rows = [r for r in mask_audit if str(r.get("method", "")).startswith("fallback_copy")]
    if fallback_rows:
        st.warning(
            "Masking backend fell back to source-audio copy for one or more files. "
            "Check `mask_error` in audit details below."
        )

    word_aligned_total = int(sum(int(r.get("word_aligned_spans", 0) or 0) for r in mask_audit))
    ratio_aligned_total = int(sum(int(r.get("ratio_aligned_spans", 0) or 0) for r in mask_audit))
    if ratio_aligned_total > 0 and word_aligned_total == 0:
        st.warning(
            "Masking used char-ratio timing only (no ASR word timestamps found). "
            "Re-run from ASR so `segments.jsonl` includes `words` for better alignment."
        )
    elif ratio_aligned_total > 0:
        st.info(
            f"Masking alignment mix: {word_aligned_total} word-aligned spans and "
            f"{ratio_aligned_total} char-ratio-aligned spans."
        )

    st.markdown("### Transcript Output")
    if asr_segments:
        joined_original = " ".join(str(s.get("text", "")).strip() for s in asr_segments if str(s.get("text", "")).strip())

        by_segment = {str(r.get("segment_id")): r.get("spans", []) for r in pii_rows}
        joined_redacted = " ".join(
            _redact_text(str(seg.get("text", "")), by_segment.get(str(seg.get("segment_id")), []))
            for seg in asr_segments
            if str(seg.get("text", "")).strip()
        )

        left, right = st.columns(2)
        with left:
            st.caption("Original transcript")
            st.text_area("original_transcript", value=joined_original or "(empty)", height=180, label_visibility="collapsed")
        with right:
            st.caption("Redacted transcript")
            st.text_area("redacted_transcript", value=joined_redacted or "(empty)", height=180, label_visibility="collapsed")

    st.markdown("### Segment-Level Table")
    table_rows = _segments_table(asr_segments, pii_rows)
    if table_rows:
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No segment rows found.")

    st.markdown("### PII Type Counts")
    type_counts = pii_meta.get("types_detected", {})
    if type_counts:
        st.bar_chart(type_counts)
    else:
        st.info("No PII spans detected.")

    st.markdown("### Audio Preview")
    st.caption("Top player: source audio. Bottom player: masked output when available.")
    st.audio(str(audio_abs))

    if mask_audit and mask_audit[0].get("masked_audio"):
        masked_audio_abs = PROJECT_ROOT / str(mask_audit[0]["masked_audio"])
        if masked_audio_abs.exists():
            st.audio(str(masked_audio_abs))

    with st.expander("Mask Audit (first rows)"):
        st.json(mask_audit[:5])

    st.markdown("### Eval Summary")
    st.json(eval_summary)


if __name__ == "__main__":
    main()
