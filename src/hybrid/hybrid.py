"""Hybrid scorer: rule-based flags are fed in as extra binary features
alongside TF-IDF text features, so the model learns per-category *when to
trust* the rule signal vs. the raw text pattern -- rather than a fixed
weighted average of the two methods' scores.

Why feature-stacking over a weighted average:
  - A fixed weight (e.g. 0.5*rule + 0.5*ml) applies the same trust level to
    every category and every example. But the rules are much more precise
    for some categories (e.g. passive_no_actor, backed by a real dependency
    parse) than others (e.g. unresolved_pronoun, a coarse heuristic) -- a
    single global weight can't express that.
  - Feature-stacking lets logistic regression learn a *per-category*
    coefficient on the rule flag itself, from data: if the rule is a
    strong, low-noise signal for a category, its learned weight will be
    large; if it's noisy, the model can lean on the text features instead
    (or even learn a negative-leaning correction if the rule
    over-triggers, as we saw with the pronoun heuristic).
  - Trade-off: this needs labeled training data to learn those weights
    (a weighted average needs none), and with only ~135 training rows the
    learned coefficients on the 5 rule features carry real variance --
    worth stating plainly in the report rather than overselling precision.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rules.detector import detect
from ml.train import build_vectorizer, CATEGORIES

RULE_FEATURE_NAMES = [f"rule_{c}" for c in CATEGORIES]


def rule_features_for(texts) -> np.ndarray:
    rows = []
    for text in texts:
        result = detect(text)
        rows.append([int(result[c]["flag"]) for c in CATEGORIES])
    return np.array(rows, dtype=float)


DISAGREEMENT_WEIGHT = 20  # how much more to weight rows where the rule flag
# disagrees with the label, during hybrid training. These rows are rare in a
# small rule-bootstrapped dataset but are the ONLY signal that teaches the
# model when to override the rule feature rather than defer to it -- at
# weight=1 they're statistically drowned out by ~130+ agreement rows and the
# model just learns to copy the rule (see notes/label-circularity below).


def train_hybrid(train_df):
    vectorizer = build_vectorizer()
    X_text = vectorizer.fit_transform(train_df["text"])
    X_rules = rule_features_for(train_df["text"])
    X = hstack([X_text, csr_matrix(X_rules)]).tocsr()

    models = {}
    for i, cat in enumerate(CATEGORIES):
        y = train_df[cat].astype(int).values
        rule_col = X_rules[:, i].astype(int)
        disagree = y != rule_col
        sample_weight = np.ones(len(y))
        sample_weight[disagree] = DISAGREEMENT_WEIGHT

        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
        clf.fit(X, y, sample_weight=sample_weight)
        models[cat] = clf
    return vectorizer, models


def predict_hybrid(texts, vectorizer, models) -> dict:
    X_text = vectorizer.transform(texts)
    X_rules = rule_features_for(texts)
    X = hstack([X_text, csr_matrix(X_rules)]).tocsr()
    preds = {}
    for cat in CATEGORIES:
        preds[cat] = models[cat].predict(X)
    return preds


def rule_feature_weight(vectorizer, models, cat):
    """Learned coefficient the hybrid model puts on category cat's OWN rule
    flag (the last len(RULE_FEATURE_NAMES) columns of the feature matrix)."""
    n_text_features = len(vectorizer.get_feature_names_out())
    own_rule_idx = n_text_features + CATEGORIES.index(cat)
    return models[cat].coef_[0][own_rule_idx]


if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv("data/processed/train.csv")
    vectorizer, models = train_hybrid(train_df)
    print("learned weight on each category's own rule flag:")
    for cat in CATEGORIES:
        w = rule_feature_weight(vectorizer, models, cat)
        print(f"  {cat}: {w:+.2f}")
