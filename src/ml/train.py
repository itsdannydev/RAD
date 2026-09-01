"""TF-IDF + Logistic Regression multi-label ambiguity classifier.

One independent binary LR classifier per category (OneVsRest), sharing a
single TF-IDF vectorizer. Interpretable by design: each category's top
positive-weight features are inspectable directly from the LR coefficients.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score

CATEGORIES = [
    "vague_quantifier",
    "passive_no_actor",
    "unresolved_pronoun",
    "missing_measurable",
    "weak_conflicting_modality",
]
SEED = 42


def build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )


def train_models(train_df: pd.DataFrame, vectorizer=None):
    if vectorizer is None:
        vectorizer = build_vectorizer()
        X = vectorizer.fit_transform(train_df["text"])
    else:
        X = vectorizer.transform(train_df["text"])

    models = {}
    for cat in CATEGORIES:
        y = train_df[cat].astype(int).values
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X, y)
        models[cat] = clf
    return vectorizer, models


def cross_validate(train_df: pd.DataFrame, n_splits=5):
    """Quick CV sanity check given the small dataset -- not a rigorous
    multi-label stratified CV, just per-category StratifiedKFold."""
    vectorizer = build_vectorizer()
    X_full = vectorizer.fit_transform(train_df["text"])

    print(f"\n{n_splits}-fold CV (per category, on train split only):")
    print(f"{'category':<28}{'precision':>10}{'recall':>10}{'f1':>10}")
    for cat in CATEGORIES:
        y = train_df[cat].astype(int).values
        n_pos = y.sum()
        if n_pos < n_splits:
            print(f"{cat:<28}  (skipped, only {n_pos} positives for {n_splits}-fold)")
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        preds = cross_val_predict(clf, X_full, y, cv=skf)
        p = precision_score(y, preds, zero_division=0)
        r = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"{cat:<28}{p:>10.2f}{r:>10.2f}{f1:>10.2f}")


def top_features(vectorizer, models, cat, n=12):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = models[cat].coef_[0]
    top_pos_idx = np.argsort(coefs)[-n:][::-1]
    top_neg_idx = np.argsort(coefs)[:n]
    pos = [(feature_names[i], round(coefs[i], 3)) for i in top_pos_idx]
    neg = [(feature_names[i], round(coefs[i], 3)) for i in top_neg_idx]
    return pos, neg


def main():
    train_df = pd.read_csv("data/processed/train.csv")

    cross_validate(train_df)

    vectorizer, models = train_models(train_df)
    joblib.dump({"vectorizer": vectorizer, "models": models}, "data/processed/ml_model.joblib")
    print("\nsaved model -> data/processed/ml_model.joblib")

    print("\nTop weighted features per category (fit on full train split):")
    for cat in CATEGORIES:
        pos, _ = top_features(vectorizer, models, cat)
        terms = ", ".join(f"{t}({w:+.2f})" for t, w in pos[:8])
        print(f"  {cat}: {terms}")


if __name__ == "__main__":
    main()
