"""Pick a diverse, reproducible sample of requirements to hand-label."""
import numpy as np
import pandas as pd

from data import load_promise_nfr

N_SAMPLE = 120
SEED = 42

CATEGORIES = [
    "vague_quantifier",
    "passive_no_actor",
    "unresolved_pronoun",
    "missing_measurable",
    "weak_conflicting_modality",
]


def main():
    df = load_promise_nfr()

    # bucket by length so short and long requirements are both represented
    df["len_bucket"] = pd.qcut(df["text"].str.split().str.len(), 4, labels=False, duplicates="drop")

    parts = []
    n_buckets = df["len_bucket"].nunique()
    per_bucket = N_SAMPLE // n_buckets
    for _, g in df.groupby("len_bucket"):
        parts.append(g.sample(n=min(len(g), per_bucket), random_state=SEED))
    sample = pd.concat(parts)

    # top up to N_SAMPLE if rounding left us short
    if len(sample) < N_SAMPLE:
        remaining = df.drop(sample.index)
        topup = remaining.sample(n=min(len(remaining), N_SAMPLE - len(sample)), random_state=SEED)
        sample = pd.concat([sample, topup])

    sample = sample.drop(columns=["len_bucket", "nfr"]).reset_index(drop=True)
    for cat in CATEGORIES:
        sample[cat] = ""  # blank for human/rule pre-fill

    sample.to_csv("data/processed/to_label.csv", index=False)
    print(f"wrote {len(sample)} rows to data/processed/to_label.csv")


if __name__ == "__main__":
    main()
