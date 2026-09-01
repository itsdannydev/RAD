"""Zero-shot LLM baseline via local Ollama (llama3.1:8b) -- for comparison only.

Not the novelty of this project; kept deliberately simple: one prompt,
one call per requirement, JSON-constrained output.
"""
import json

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"

CATEGORIES = [
    "vague_quantifier",
    "passive_no_actor",
    "unresolved_pronoun",
    "missing_measurable",
    "weak_conflicting_modality",
]

SYSTEM_PROMPT = """You are a requirements-quality analyst. Classify a single software \
requirement sentence against exactly these 5 ambiguity categories. A requirement can match \
zero, one, or several categories.

1. vague_quantifier: a vague quantifier or adjective with no defined threshold \
(e.g. "appropriate", "sufficient", "fast", "user-friendly").
2. passive_no_actor: passive voice that hides who is responsible \
(e.g. "the data shall be validated" -- validated by whom?).
3. unresolved_pronoun: a pronoun ("it", "this", "they", "their"...) whose antecedent is unclear.
4. missing_measurable: a performance/quality claim with no number or unit given \
(e.g. "respond quickly" -- no number given).
5. weak_conflicting_modality: mixing modal verbs ("may", "should", "shall", "must") \
within the same requirement so the intended strength is unclear.

Respond with ONLY a JSON object with exactly these 5 boolean keys: \
vague_quantifier, passive_no_actor, unresolved_pronoun, missing_measurable, weak_conflicting_modality. \
No explanation, no markdown, just the JSON object."""


def classify(text: str, timeout=30) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Requirement: "{text}"'},
        ],
        "format": "json",
        "options": {"temperature": 0},
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}
    return {cat: bool(parsed.get(cat, False)) for cat in CATEGORIES}


if __name__ == "__main__":
    examples = [
        "The system shall respond quickly to user requests.",
        "The data shall be validated before processing.",
        "When the sensor detects motion, it should update accordingly.",
    ]
    for ex in examples:
        print(">", ex)
        print("  ", classify(ex))
