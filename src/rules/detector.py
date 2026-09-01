"""Rule-based ambiguity detector for software requirement statements.

Five independent checks map to the RAD taxonomy:
  1. vague_quantifier          - lexicon lookup (rules/vague_quantifiers.txt)
  2. passive_no_actor          - spaCy dependency parse (nsubjpass/agent)
  3. unresolved_pronoun        - antecedent-candidate heuristic (NOT full coreference resolution)
  4. missing_measurable        - performance adjective lexicon + absence of a digit in the sentence
  5. weak_conflicting_modality - modal-verb strength lexicon (rules/modal_strength.csv)

Every check returns {"flag": bool, "spans": [str, ...]} so callers know
*which* word(s) triggered it, not just a yes/no.
"""
import csv
import re
from pathlib import Path

import spacy

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"

_NUMBER_RE = re.compile(r"\d")

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _load_lexicon(filename: str) -> set:
    path = RULES_DIR / filename
    terms = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            terms.add(line)
    return terms


def _load_modal_strength() -> dict:
    path = RULES_DIR / "modal_strength.csv"
    strength = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            strength[row["modal"].strip().lower()] = int(row["tier"])
    return strength


_VAGUE_TERMS = _load_lexicon("vague_quantifiers.txt")
_PERF_TERMS = _load_lexicon("performance_adjectives.txt")
_REFERENT_NOUNS = _load_lexicon("referent_nouns.txt")
_MODAL_STRENGTH = _load_modal_strength()

# multi-word lexicon phrases need direct substring matching over the raw text
_VAGUE_PHRASES = sorted((t for t in _VAGUE_TERMS if " " in t), key=len, reverse=True)
_VAGUE_SINGLE = {t for t in _VAGUE_TERMS if " " not in t}

_PRONOUNS_ALWAYS = {"it", "its", "they", "them", "their", "theirs"}
_PRONOUNS_AGENT_LIKE = {"they", "them", "their", "theirs"}
_PRONOUNS_DEMONSTRATIVE = {"this", "that", "these", "those"}


def _check_vague_quantifier(doc) -> dict:
    spans = []
    lower_text = doc.text.lower()
    for phrase in _VAGUE_PHRASES:
        if phrase in lower_text:
            spans.append(phrase)
    for token in doc:
        lemma = token.lemma_.lower()
        word = token.text.lower()
        if word == "most" and token.i > 0 and doc[token.i - 1].lower_ == "at":
            continue  # "at most N" is a precise bound, not a vague quantifier
        if lemma in _VAGUE_SINGLE or word in _VAGUE_SINGLE:
            spans.append(token.text)
    spans = list(dict.fromkeys(spans))  # dedupe, keep order
    return {"flag": len(spans) > 0, "spans": spans}


def _check_passive_no_actor(doc) -> dict:
    spans = []
    for token in doc:
        if token.dep_ == "nsubjpass":
            verb = token.head
            has_agent = any(child.dep_ == "agent" for child in verb.children)
            if not has_agent:
                aux_tokens = [c for c in verb.children if c.dep_ in ("aux", "auxpass")]
                phrase_tokens = sorted(aux_tokens + [verb], key=lambda t: t.i)
                spans.append(" ".join(t.text for t in phrase_tokens))
    spans = list(dict.fromkeys(spans))
    return {"flag": len(spans) > 0, "spans": spans}


def _check_unresolved_pronoun(doc) -> dict:
    spans = []
    for sent in doc.sents:
        sent_start = sent.start
        for token in sent:
            lower = token.text.lower()
            is_target = False
            if lower in _PRONOUNS_ALWAYS and token.pos_ == "PRON":
                is_target = True
                # expletive/dummy "it" ("it should be possible to...", "it is
                # necessary that...") doesn't refer to anything -- it's a
                # placeholder subject, not what this category targets
                if lower == "it" and token.dep_ == "nsubj" and token.head.lemma_ == "be":
                    sibling_deps = {c.dep_ for c in token.head.children}
                    if "acomp" in sibling_deps and ("xcomp" in sibling_deps or "ccomp" in sibling_deps):
                        is_target = False
            elif (
                lower in _PRONOUNS_DEMONSTRATIVE
                and token.pos_ == "PRON"
                and token.dep_ != "det"
                and token.tag_ != "WDT"  # WDT = relative pronoun ("words THAT are...") -- always has a clear antecedent, not what this category targets
            ):
                is_target = True  # excludes complementizer "that" (pos_=SCONJ, e.g. "determine THAT X happens") too -- a grammatical connector, not a referring pronoun
            if not is_target:
                continue

            # candidate antecedents: noun-chunk roots appearing earlier in this sentence.
            # "they/them/their" almost always refers to a person/role in requirements
            # text, so narrow candidates to a referent-noun allowlist for those --
            # otherwise generic nouns like "product"/"ships" get wrongly counted as
            # competing antecedents alongside the real one (e.g. "the player").
            if lower in _PRONOUNS_AGENT_LIKE:
                candidates = {
                    chunk.root.lemma_.lower()
                    for chunk in doc.noun_chunks
                    if sent_start <= chunk.root.i < token.i
                    and chunk.root.lemma_.lower() in _REFERENT_NOUNS
                }
            else:
                candidates = {
                    chunk.root.lemma_.lower()
                    for chunk in doc.noun_chunks
                    if sent_start <= chunk.root.i < token.i
                }
            if len(candidates) != 1:
                spans.append(token.text)
    spans = list(dict.fromkeys(spans))
    return {"flag": len(spans) > 0, "spans": spans}


def _check_missing_measurable(doc) -> dict:
    spans = []
    for sent in doc.sents:
        has_number = bool(_NUMBER_RE.search(sent.text))
        if has_number:
            continue
        for token in sent:
            lemma = token.lemma_.lower()
            word = token.text.lower()
            if lemma in _PERF_TERMS or word in _PERF_TERMS:
                spans.append(token.text)
    spans = list(dict.fromkeys(spans))
    return {"flag": len(spans) > 0, "spans": spans}


def _check_weak_conflicting_modality(doc) -> dict:
    found = []  # (word, tier)
    for token in doc:
        word = token.text.lower()
        if token.tag_ == "MD" and word in _MODAL_STRENGTH:
            # a negated modal ("cannot", "should not") expresses a specific
            # negative-capability condition, not a competing obligation
            # strength for the requirement's main action -- counting it
            # toward "mixing" conflates negation with genuine ambiguity
            next_tok = doc[token.i + 1] if token.i + 1 < len(doc) else None
            if next_tok is not None and (next_tok.dep_ == "neg" or next_tok.lemma_ == "not"):
                continue
            found.append((token.text, _MODAL_STRENGTH[word]))
    tiers = {tier for _, tier in found}
    flag = len(tiers) > 1
    spans = list(dict.fromkeys(w for w, _ in found)) if flag else []
    return {"flag": flag, "spans": spans}


def detect(text: str) -> dict:
    """Run all 5 rule-based checks on a requirement string."""
    doc = _get_nlp()(text)
    return {
        "vague_quantifier": _check_vague_quantifier(doc),
        "passive_no_actor": _check_passive_no_actor(doc),
        "unresolved_pronoun": _check_unresolved_pronoun(doc),
        "missing_measurable": _check_missing_measurable(doc),
        "weak_conflicting_modality": _check_weak_conflicting_modality(doc),
    }


if __name__ == "__main__":
    examples = [
        "The system shall respond quickly to user requests.",
        "The data shall be validated before processing.",
        "The data shall be validated by the input handler before processing.",
        "When the sensor detects motion, it should update accordingly.",
        "The system may notify the administrator. The system should notify the administrator within 5 seconds.",
        "The product shall be intuitive and user-friendly for new users.",
        "The system shall process 1000 transactions per second with 99.9% uptime.",
    ]
    for ex in examples:
        print("\n>", ex)
        result = detect(ex)
        for cat, res in result.items():
            if res["flag"]:
                print(f"   [{cat}] {res['spans']}")
