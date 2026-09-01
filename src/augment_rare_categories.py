"""Top up the labeling sample with extra examples of rare categories.

Random sampling reflects the true (low) base rate of missing_measurable and
weak_conflicting_modality, which risks a test split with zero positives for
those categories. We deliberately oversample them from the remaining pool
so the benchmark (Phase 6) has something to measure. This is a documented
scope decision, not an attempt to inflate overall performance.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_promise_nfr
from rules.detector import detect

TARGET_PER_RARE_CATEGORY = 15
RARE_CATEGORIES = ["missing_measurable", "weak_conflicting_modality"]
CATEGORIES = [
    "vague_quantifier",
    "passive_no_actor",
    "unresolved_pronoun",
    "missing_measurable",
    "weak_conflicting_modality",
]


def main():
    full = load_promise_nfr()
    current = pd.read_csv("data/processed/to_label.csv")
    remaining = full[~full["text"].isin(set(current["text"]))].copy()

    flags = {cat: [] for cat in RARE_CATEGORIES}
    for idx, row in remaining.iterrows():
        result = detect(row["text"])
        for cat in RARE_CATEGORIES:
            flags[cat].append(result[cat]["flag"])
    for cat in RARE_CATEGORIES:
        remaining[cat] = flags[cat]

    to_add_ids = set()
    for cat in RARE_CATEGORIES:
        hits = remaining[remaining[cat]].head(TARGET_PER_RARE_CATEGORY)
        to_add_ids.update(hits["req_id"])
        print(f"{cat}: found {remaining[cat].sum()} candidates in remaining pool, adding {len(hits)}")

    extra = remaining[remaining["req_id"].isin(to_add_ids)][["req_id", "text"]].reset_index(drop=True)
    for cat in CATEGORIES:
        extra[cat] = ""

    combined = pd.concat([current, extra], ignore_index=True)
    combined.to_csv("data/processed/to_label.csv", index=False)
    print(f"\ntotal labeling sample size now: {len(combined)} (added {len(extra)})")


if __name__ == "__main__":
    main()
