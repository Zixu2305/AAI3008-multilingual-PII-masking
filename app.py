"""Streamlit frontend for testing the hybrid PII detection pipeline."""

import html

import streamlit as st

from src.pii.ner import load_ner_pipeline
from src.pii.run import _detect_pii_for_text

# -- Colour map for PII types --------------------------------------------------

TYPE_COLOURS = {
    "NAME": "#ff6b6b",
    "PHONE": "#4ecdc4",
    "EMAIL": "#45b7d1",
    "ADDRESS": "#f9c74f",
    "ID": "#a78bfa",
}
DEFAULT_COLOUR = "#90be6d"

# -- Config defaults (matching configs/pii_eval_gold.yaml) ---------------------

RULES_CFG = {
    "phone": True,
    "email": True,
    "zh_phone": True,
    "id_number": True,
    "address": True,
    "en_spoken_phone": True,
    "partial_phone": True,
    "nric": True,
    "postal_code": True,
}

NER_CFG = {
    "label_map": {
        "PER": "NAME",
        "PERSON": "NAME",
        "LOC": "ADDRESS",
        "GPE": "ADDRESS",
        "ORG": "ORG",
    },
    "max_length": 256,
    "score_threshold": None,
}

NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"

MERGE_GAP_CHARS = 1
MIN_SPAN_CHARS = 1


# -- Load NER model (cached) --------------------------------------------------

@st.cache_resource
def get_ner_pipeline():
    return load_ner_pipeline(model_name=NER_MODEL, device_cfg="cpu")


# -- Helpers -------------------------------------------------------------------

def build_highlighted_html(text: str, spans: list[dict]) -> str:
    """Build HTML string with PII spans highlighted and colour-coded."""
    if not spans:
        return f"<p>{html.escape(text)}</p>"

    sorted_spans = sorted(spans, key=lambda s: s["start"])
    parts = []
    prev_end = 0

    for span in sorted_spans:
        start, end = span["start"], span["end"]
        if start < prev_end:
            continue
        pii_type = span.get("type", "UNKNOWN")
        colour = TYPE_COLOURS.get(pii_type, DEFAULT_COLOUR)

        parts.append(html.escape(text[prev_end:start]))
        parts.append(
            f'<mark style="background-color:{colour};padding:2px 4px;border-radius:4px;color:#1a1a2e;">'
            f'{html.escape(text[start:end])}'
            f'<sup style="font-size:0.7em;margin-left:2px;">{pii_type}</sup>'
            f"</mark>"
        )
        prev_end = end

    parts.append(html.escape(text[prev_end:]))
    return f'<div style="line-height:2;font-size:1.05em;">{"".join(parts)}</div>'


# -- App -----------------------------------------------------------------------

st.set_page_config(page_title="PII Detection Demo", layout="wide")
st.title("PII Detection Demo")
st.caption("Hybrid pipeline: regex rules + NER (XLM-RoBERTa) + postprocessing")

transcript = st.text_area(
    "Enter transcript",
    height=200,
    placeholder="e.g. My name is Daniel Tan, phone 9123 4567, email dan@test.com, reference number 5.",
)

if st.button("Detect PII", type="primary"):
    if not transcript.strip():
        st.warning("Please enter a transcript.")
    else:
        ner_pipe = get_ner_pipeline()
        spans = _detect_pii_for_text(
            text=transcript,
            enable_rules=True,
            rules_cfg=RULES_CFG,
            ner_pipe=ner_pipe,
            ner_cfg=NER_CFG,
            enable_llm=False,
            llm_cfg={},
            merge_gap_chars=MERGE_GAP_CHARS,
            min_span_chars=MIN_SPAN_CHARS,
        )

        st.subheader("Highlighted Transcript")

        # Colour legend
        legend_items = "  ".join(
            f'<span style="background-color:{c};padding:2px 6px;border-radius:4px;color:#1a1a2e;'
            f'font-size:0.85em;">{t}</span>'
            for t, c in TYPE_COLOURS.items()
        )
        st.markdown(legend_items, unsafe_allow_html=True)
        st.markdown(build_highlighted_html(transcript, spans), unsafe_allow_html=True)

        st.subheader("Detected Entities")
        if spans:
            rows = [
                {
                    "Type": s["type"],
                    "Text": s["text"],
                    "Start": s["start"],
                    "End": s["end"],
                    "Source": s.get("source", ""),
                }
                for s in spans
            ]
            st.table(rows)
        else:
            st.info("No PII entities detected.")
