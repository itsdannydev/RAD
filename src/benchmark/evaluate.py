"""Benchmark harness: rules-only vs ML-only vs LLM-only vs hybrid on the
same held-out test set. Outputs per-category P/R/F1 and a comparison table.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rules.detector import detect
from ml.train import CATEGORIES, train_models
from hybrid.hybrid import train_hybrid, predict_hybrid
from llm.ollama_classifier import classify as llm_classify

LLM_CACHE_PATH = "data/processed/llm_test_predictions.json"


def get_rules_predictions(texts):
    preds = {cat: [] for cat in CATEGORIES}
    for text in texts:
        result = detect(text)
        for cat in CATEGORIES:
            preds[cat].append(int(result[cat]["flag"]))
    return preds


def get_ml_predictions(texts, vectorizer, models):
    X = vectorizer.transform(texts)
    return {cat: models[cat].predict(X) for cat in CATEGORIES}


def get_llm_predictions(texts, req_ids):
    cache_path = Path(LLM_CACHE_PATH)
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    preds = {cat: [] for cat in CATEGORIES}
    for req_id, text in zip(req_ids, texts):
        if req_id in cache:
            result = cache[req_id]
        else:
            result = llm_classify(text)
            cache[req_id] = result
            cache_path.write_text(json.dumps(cache, indent=2))
        for cat in CATEGORIES:
            preds[cat].append(int(result[cat]))
    return preds


def score(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    # Ground truth for scoring is an INDEPENDENT human pass over the test texts
    # (test_human_verified.csv), not the rule-bootstrapped labels in test.csv.
    # Those labels were derived from the rule detector's own output, so scoring
    # rules-only against them would be circular (rules "predicting" themselves).
    test_texts_df = pd.read_csv("data/processed/test.csv")[["req_id", "text"]]
    verified = pd.read_csv("data/processed/test_human_verified.csv")
    test_df = test_texts_df.merge(verified, on="req_id", how="inner")
    assert len(test_df) == len(test_texts_df), "verified labels missing rows vs test.csv"

    print("training ML-only model...")
    ml_vectorizer, ml_models = train_models(train_df)

    print("training hybrid model...")
    hy_vectorizer, hy_models = train_hybrid(train_df)

    print("running rules...")
    rules_preds = get_rules_predictions(test_df["text"])

    print("running ML...")
    ml_preds = get_ml_predictions(test_df["text"], ml_vectorizer, ml_models)

    print("running LLM (Ollama, cached after first run)...")
    llm_preds = get_llm_predictions(test_df["text"], test_df["req_id"])

    print("running hybrid...")
    hybrid_preds_raw = predict_hybrid(test_df["text"], hy_vectorizer, hy_models)
    hybrid_preds = {cat: hybrid_preds_raw[cat] for cat in CATEGORIES}

    methods = {
        "rules": rules_preds,
        "ml": ml_preds,
        "llm": llm_preds,
        "hybrid": hybrid_preds,
    }

    results = {}
    for method_name, preds in methods.items():
        results[method_name] = {}
        for cat in CATEGORIES:
            y_true = test_df[cat].astype(int).values
            results[method_name][cat] = score(y_true, preds[cat])

    # ---- comparison table ----
    print("\n" + "=" * 100)
    print(f"{'category':<28}{'method':<10}{'P':>8}{'R':>8}{'F1':>8}")
    print("-" * 100)
    for cat in CATEGORIES:
        for method_name in methods:
            m = results[method_name][cat]
            print(f"{cat:<28}{method_name:<10}{m['precision']:>8.2f}{m['recall']:>8.2f}{m['f1']:>8.2f}")
        print()

    print("=" * 100)
    print(f"{'METHOD':<10}{'macro-avg P':>14}{'macro-avg R':>14}{'macro-avg F1':>14}")
    for method_name in methods:
        avg_p = sum(results[method_name][c]["precision"] for c in CATEGORIES) / len(CATEGORIES)
        avg_r = sum(results[method_name][c]["recall"] for c in CATEGORIES) / len(CATEGORIES)
        avg_f1 = sum(results[method_name][c]["f1"] for c in CATEGORIES) / len(CATEGORIES)
        print(f"{method_name:<10}{avg_p:>14.2f}{avg_r:>14.2f}{avg_f1:>14.2f}")

    # ---- save everything for later inspection / the streamlit app / report ----
    out = {
        "results": results,
        "predictions": {
            method_name: {cat: [int(v) for v in preds[cat]] for cat in CATEGORIES}
            for method_name, preds in methods.items()
        },
        "test_req_ids": test_df["req_id"].tolist(),
        "test_texts": test_df["text"].tolist(),
        "test_labels": {cat: test_df[cat].astype(int).tolist() for cat in CATEGORIES},
    }
    Path("data/processed/benchmark_results.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> data/processed/benchmark_results.json")

    # persist trained models for the Streamlit app
    joblib.dump({"vectorizer": ml_vectorizer, "models": ml_models}, "data/processed/ml_model.joblib")
    joblib.dump({"vectorizer": hy_vectorizer, "models": hy_models}, "data/processed/hybrid_model.joblib")


if __name__ == "__main__":
    main()
