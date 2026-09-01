"""Run the rule-based detector over the labeling sample to pre-fill labels.

This isn't the final label set -- it's a fast draft a human then corrects,
instead of hand-labeling 120 items from a blank slate.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules.detector import detect

CATEGORIES = [
    "vague_quantifier",
    "passive_no_actor",
    "unresolved_pronoun",
    "missing_measurable",
    "weak_conflicting_modality",
]


def main():
    df = pd.read_csv("data/processed/to_label.csv")
    spans_col = {cat: [] for cat in CATEGORIES}

    for cat in CATEGORIES:
        df[cat] = False

    for idx, row in df.iterrows():
        result = detect(row["text"])
        for cat in CATEGORIES:
            df.at[idx, cat] = result[cat]["flag"]
            spans_col[cat].append("; ".join(result[cat]["spans"]))

    for cat in CATEGORIES:
        df[f"{cat}_spans"] = spans_col[cat]

    df.to_csv("data/processed/prelabeled.csv", index=False)

    print("Flag rate per category (out of", len(df), "):")
    for cat in CATEGORIES:
        print(f"  {cat}: {df[cat].sum()} ({100*df[cat].mean():.1f}%)")

    n_flagged_any = (df[CATEGORIES].sum(axis=1) > 0).sum()
    n_clean = len(df) - n_flagged_any
    print(f"\nrequirements with >=1 flag: {n_flagged_any}")
    print(f"requirements with 0 flags (clean): {n_clean}")
    avg_flags = df[CATEGORIES].sum(axis=1).mean()
    print(f"avg categories flagged per requirement: {avg_flags:.2f}")


if __name__ == "__main__":
    main()
