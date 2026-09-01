"""RAD -- Requirement Ambiguity Detector. Minimal Streamlit UI.

Paste or upload a plain-text requirements doc (one requirement per line, or
free text split into sentences) and get back, per sentence: which of the 5
taxonomy categories are flagged, by which method, matched spans highlighted,
and a severity score.
"""
import html
import re
import sys
from pathlib import Path

import joblib
import spacy
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rules.detector import detect

CATEGORIES = [
    "vague_quantifier",
    "passive_no_actor",
    "unresolved_pronoun",
    "missing_measurable",
    "weak_conflicting_modality",
]
CATEGORY_LABELS = {
    "vague_quantifier": "Vague quantifier/adjective",
    "passive_no_actor": "Passive voice, no actor",
    "unresolved_pronoun": "Unresolved pronoun",
    "missing_measurable": "Missing measurable criteria",
    "weak_conflicting_modality": "Weak/conflicting modality",
}
CATEGORY_COLORS = {
    "vague_quantifier": "#f59e0b",
    "passive_no_actor": "#ef4444",
    "unresolved_pronoun": "#8b5cf6",
    "missing_measurable": "#0ea5e9",
    "weak_conflicting_modality": "#ec4899",
}

st.set_page_config(page_title="RAD -- Requirement Ambiguity Detector", layout="wide")


@st.cache_resource
def load_models():
    ml = joblib.load("data/processed/ml_model.joblib")
    hybrid = joblib.load("data/processed/hybrid_model.joblib")
    return ml, hybrid


@st.cache_resource
def load_sentencizer():
    nlp = spacy.load("en_core_web_sm")
    return nlp


def split_into_requirements(raw_text: str):
    """One requirement per non-empty line if the input looks line-delimited,
    else fall back to sentence splitting."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    if len(lines) > 1:
        return lines
    nlp = load_sentencizer()
    doc = nlp(raw_text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def rule_features_for_texts(texts, ml_pack):
    """Rebuild the rule-flag feature block hybrid needs, without re-importing
    hybrid.py's sklearn-specific plumbing (keeps the app lightweight)."""
    import numpy as np
    rows = []
    for t in texts:
        r = detect(t)
        rows.append([int(r[c]["flag"]) for c in CATEGORIES])
    return np.array(rows, dtype=float)


def predict_all(text, ml_pack, hybrid_pack):
    rule_result = detect(text)

    ml_vec, ml_models = ml_pack["vectorizer"], ml_pack["models"]
    X_ml = ml_vec.transform([text])
    ml_result = {cat: bool(ml_models[cat].predict(X_ml)[0]) for cat in CATEGORIES}

    from scipy.sparse import hstack, csr_matrix
    hy_vec, hy_models = hybrid_pack["vectorizer"], hybrid_pack["models"]
    X_text = hy_vec.transform([text])
    X_rules = rule_features_for_texts([text], hybrid_pack)
    X_hy = hstack([X_text, csr_matrix(X_rules)]).tocsr()
    hybrid_result = {cat: bool(hy_models[cat].predict(X_hy)[0]) for cat in CATEGORIES}

    return rule_result, ml_result, hybrid_result


def highlight_spans(text, rule_result):
    spans = []
    for cat in CATEGORIES:
        for span in rule_result[cat]["spans"]:
            idx = text.lower().find(span.lower())
            if idx != -1:
                spans.append((idx, idx + len(span), cat))
    spans.sort(key=lambda s: s[0])

    out = []
    cursor = 0
    for start, end, cat in spans:
        if start < cursor:
            continue
        out.append(html.escape(text[cursor:start]))
        color = CATEGORY_COLORS[cat]
        out.append(
            f'<mark style="background:{color}33;border-bottom:2px solid {color};padding:0 2px;" '
            f'title="{CATEGORY_LABELS[cat]}">{html.escape(text[start:end])}</mark>'
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


st.title("RAD -- Requirement Ambiguity Detector")
st.caption(
    "Paste requirements below (one per line works best). Each sentence is scored by "
    "the rule-based detector, the ML classifier, and the hybrid (rules-as-features) model."
)

default_text = """The system shall respond quickly to user requests.
The data shall be validated before processing.
When the sensor detects motion, it should update accordingly.
The system may notify the administrator. The system should notify the administrator within 5 seconds.
The system shall process 1000 transactions per second with 99.9% uptime."""

raw_text = st.text_area("Requirements text", value=default_text, height=180)

uploaded = st.file_uploader("...or upload a plain-text (.txt) file", type=["txt"])
if uploaded is not None:
    raw_text = uploaded.read().decode("utf-8")

method = st.radio("Scoring method to display", ["hybrid", "rules", "ml"], horizontal=True)

if st.button("Analyze", type="primary") and raw_text.strip():
    ml_pack, hybrid_pack = load_models()
    requirements = split_into_requirements(raw_text)

    st.subheader(f"Results ({len(requirements)} requirement(s))")

    for i, req in enumerate(requirements, 1):
        rule_result, ml_result, hybrid_result = predict_all(req, ml_pack, hybrid_pack)
        active_result = {"rules": rule_result, "ml": ml_result, "hybrid": hybrid_result}[method]

        if method == "rules":
            flagged = [c for c in CATEGORIES if rule_result[c]["flag"]]
        else:
            flagged = [c for c in CATEGORIES if active_result[c]]
        severity = len(flagged)

        with st.container(border=True):
            st.markdown(f"**#{i}**  &nbsp; severity: {'\U0001F534' * severity}{'⚪' * (5 - severity)} ({severity}/5)")
            st.markdown(highlight_spans(req, rule_result), unsafe_allow_html=True)
            if flagged:
                badges = "  ".join(
                    f'<span style="background:{CATEGORY_COLORS[c]}22;border:1px solid {CATEGORY_COLORS[c]};'
                    f'border-radius:12px;padding:2px 10px;font-size:0.85em;">{CATEGORY_LABELS[c]}</span>'
                    for c in flagged
                )
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown(":green[No ambiguity flagged.]")

            with st.expander("Compare all 3 methods for this sentence"):
                for name, result in [("Rules", rule_result), ("ML", ml_result), ("Hybrid", hybrid_result)]:
                    row = []
                    for c in CATEGORIES:
                        flag = result[c]["flag"] if name == "Rules" else result[c]
                        row.append(f"{CATEGORY_LABELS[c]}: {'Y' if flag else '.'}")
                    st.text(f"{name:8s} | " + "  ".join(row))

st.divider()
st.caption(
    "5-category taxonomy: vague quantifier/adjective, passive voice hiding the actor, "
    "unresolved pronoun reference, missing measurable/testable criteria, weak/conflicting modality. "
    "Rules = spaCy dependency parsing + editable lexicons. ML = TF-IDF + logistic regression. "
    "Hybrid = ML retrained with the rule flags as extra features."
)
