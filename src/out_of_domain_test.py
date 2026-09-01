"""Out-of-domain stress test: fresh requirement-style sentences for a fitness
wearable / IoT product -- a domain PROMISE NFR never covers -- run through
all 4 methods with NO retraining, to check the detector generalizes beyond
its training distribution rather than just memorizing PROMISE phrasing.
"""
import sys
from pathlib import Path

import joblib
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules.detector import detect
from ml.train import CATEGORIES
from llm.ollama_classifier import classify as llm_classify

SENTENCES = [
    "The wearable device shall provide a seamless and intuitive user experience.",
    "Heart rate data shall be encrypted before transmission to the cloud.",
    "The app synchronizes with the wearable and the smart scale. It should update the dashboard within 2 seconds.",
    "The mobile app shall load quickly when launched.",
    "The system should sync data in the background, but critical alerts shall be delivered instantly.",
    "The wearable shall transmit heart rate data to the companion app every 5 seconds via Bluetooth Low Energy.",
    "Battery level readings shall be validated by the device firmware before being displayed.",
    "The device shall respond to touch input in an acceptable time.",
    "Only the registered user shall be able to unlock the device using their fingerprint.",
    "The device shall achieve a battery life of at least 48 hours under typical usage conditions.",
]


def rule_feature_row(text):
    result = detect(text)
    return [int(result[c]["flag"]) for c in CATEGORIES]


def main():
    ml_pack = joblib.load("data/processed/ml_model.joblib")
    hybrid_pack = joblib.load("data/processed/hybrid_model.joblib")

    for text in SENTENCES:
        rule_result = detect(text)
        rule_flags = {c: rule_result[c]["flag"] for c in CATEGORIES}

        X_ml = ml_pack["vectorizer"].transform([text])
        ml_flags = {c: bool(ml_pack["models"][c].predict(X_ml)[0]) for c in CATEGORIES}

        X_text = hybrid_pack["vectorizer"].transform([text])
        X_rules = csr_matrix([rule_feature_row(text)])
        X_hy = hstack([X_text, X_rules]).tocsr()
        hybrid_flags = {c: bool(hybrid_pack["models"][c].predict(X_hy)[0]) for c in CATEGORIES}

        llm_flags = llm_classify(text)

        print("=" * 100)
        print(text)
        for cat in CATEGORIES:
            spans = rule_result[cat]["spans"]
            span_str = f" [{', '.join(spans)}]" if spans else ""
            print(
                f"  {cat:28s} rules={str(rule_flags[cat]):5s}{span_str:25s}"
                f" ml={str(ml_flags[cat]):5s} llm={str(llm_flags[cat]):5s} hybrid={str(hybrid_flags[cat]):5s}"
            )


if __name__ == "__main__":
    main()
