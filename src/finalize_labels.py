"""Finalize labels.csv from the (rule-assisted, manually corrected) prelabeled data,
then create a reproducible 80/20 train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORIES = [
    "vague_quantifier",
    "passive_no_actor",
    "unresolved_pronoun",
    "missing_measurable",
    "weak_conflicting_modality",
]
SEED = 42


def main():
    df = pd.read_csv("data/processed/prelabeled.csv")
    labels = df[["req_id", "text"] + CATEGORIES].copy()
    labels.to_csv("data/processed/labels.csv", index=False)
    print(f"labels.csv: {len(labels)} rows")

    # stratify on "has any flag" as a coarse proxy since true multi-label
    # stratification needs an extra dependency we don't have time to add
    strat_key = (labels[CATEGORIES].sum(axis=1) > 0).astype(int)
    train, test = train_test_split(
        labels, test_size=0.2, random_state=SEED, stratify=strat_key
    )
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print(f"train: {len(train)}  test: {len(test)}")
    print("\ntest set positive counts per category:")
    for cat in CATEGORIES:
        print(f"  {cat}: {test[cat].sum()}")


if __name__ == "__main__":
    main()
