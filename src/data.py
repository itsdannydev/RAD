"""Load and clean the PROMISE NFR dataset.

Source file has a handful of rows where the original requirement text
contained a literal tab character (originally a list separator), which
breaks the expected 2-column (RequirementText, NFR) structure. We repair
those by treating the final tab-split token as the label and rejoining
everything before it as the text.
"""
import pandas as pd


def load_promise_nfr(path: str = "data/raw/promise_nfr.csv") -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()  # skip "RequirementText\tNFR"
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            last = parts[-1].strip()
            if last in ("0", "1"):
                # last token is a valid label; everything before it is text
                text = " ".join(p.strip() for p in parts[:-1]).strip()
                nfr = int(last)
            else:
                # label missing/malformed upstream (~a handful of rows);
                # the whole line is text we can still use, just unlabeled for NFR
                text = " ".join(p.strip() for p in parts).strip()
                nfr = None
            text = " ".join(text.split())  # collapse repeated whitespace
            rows.append({"text": text, "nfr": nfr})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df.insert(0, "req_id", [f"REQ-{i:04d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = load_promise_nfr()
    print("shape:", df.shape)
    print(df.head(10).to_string())
    print("word count stats:")
    print(df["text"].str.split().str.len().describe())
