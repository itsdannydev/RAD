# RAD — Requirement Ambiguity Detector

Detects ambiguous requirement statements against a 5-category taxonomy using
three independent methods (rule-based, ML, LLM zero-shot) plus a hybrid that
feeds rule flags into the ML classifier as extra features.

## Taxonomy

Each requirement is scored against 5 independent categories (a requirement
can match zero, one, or several):

1. **Vague quantifier/adjective** — no defined threshold (e.g. "appropriate," "fast," "user-friendly").
2. **Passive voice, no actor** — hides who is responsible (e.g. "the data shall be validated" — by whom?).
3. **Unresolved pronoun reference** — unclear antecedent (e.g. "it should update accordingly").
4. **Missing measurable/testable criteria** — a performance claim with no number/unit (e.g. "respond quickly").
5. **Weak/conflicting modality** — mixing "may"/"should"/"shall"/"must" so intended strength is unclear.

## Architecture

| Component | File | Approach |
|---|---|---|
| Rule-based detector | `src/rules/detector.py` | spaCy dependency parsing + editable lexicons (`rules/*.txt`, `rules/*.csv`) |
| ML classifier | `src/ml/train.py` | TF-IDF + logistic regression, one-vs-rest, interpretable coefficients |
| LLM baseline | `src/llm/ollama_classifier.py` | Local Ollama (`llama3.1:8b`), zero-shot, JSON-constrained prompt |
| Hybrid | `src/hybrid/hybrid.py` | Rule flags stacked as extra binary features into the ML classifier |
| Benchmark | `src/benchmark/evaluate.py` | All 4 methods, same held-out test set, per-category P/R/F1 |
| UI | `app/streamlit_app.py` | Paste/upload text → per-sentence flags, highlighted spans, severity |

Data pipeline: `src/data.py` (load PROMISE NFR) → `src/sample_for_labeling.py`
+ `src/augment_rare_categories.py` (build labeling batch) →
`src/prelabel.py` (rule-assisted pre-fill) → `src/finalize_labels.py`
(labels.csv + 80/20 split).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

You also need [Ollama](https://ollama.com) running locally with the model pulled,
for the LLM baseline and benchmark (not needed for rules/ML/hybrid/UI alone):

```bash
ollama serve &          # if not already running
ollama pull llama3.1:8b
```

## Running things

**Regenerate everything from scratch** (data → labels → train/test split):
```bash
python3 src/data.py                        # sanity check the raw dataset loads
python3 src/sample_for_labeling.py          # sample requirements to label
python3 src/augment_rare_categories.py      # oversample rare categories
python3 src/prelabel.py                     # rule-assisted pre-labeling
python3 src/finalize_labels.py              # labels.csv + train/test split
```
Note: `labels.csv`/`train.csv` already have 2 manually-corrected rows
(`REQ-0160`, `REQ-0175` — see Limitations) that re-running `finalize_labels.py`
will silently drop. Re-apply them if you regenerate labels (search this repo's
history for the "reapply manual corrections" step, or just re-flip
`unresolved_pronoun` to `False` for those two rows).

**Try the rule-based detector directly:**
```bash
python3 -m src.rules.detector
```

**Train the ML classifier** (prints 5-fold CV + top features per category):
```bash
python3 -m src.ml.train
```

**Train the hybrid and inspect learned rule-feature weights:**
```bash
python3 -m src.hybrid.hybrid
```

**Run the full benchmark** (rules vs ML vs LLM vs hybrid; requires Ollama running):
```bash
python3 -m src.benchmark.evaluate
```
LLM predictions on the test set are cached in
`data/processed/llm_test_predictions.json` — delete it to force fresh calls.

## Running the Streamlit app

This is the deliverable most people will actually click through, so here's
the full walkthrough.

### Prerequisites

The app itself only needs the rule detector + the two trained model files —
**it does not call the LLM/Ollama at all**, so you don't need `ollama serve`
running just to use the UI. It does need:

1. The Python environment set up (`python3 -m venv .venv`, `pip install -r
   requirements.txt`, `python -m spacy download en_core_web_sm` — see Setup
   above) if you haven't already.
2. Two trained model files to exist on disk:
   - `data/processed/ml_model.joblib`
   - `data/processed/hybrid_model.joblib`

   Both are already committed in this repo, so if you haven't deleted
   `data/processed/`, you're ready to go — skip to "Start the app" below.

   If either file is **missing** (e.g. you wiped `data/processed/` or cloned
   a stripped-down copy), regenerate them with one of:
   ```bash
   python3 -m src.benchmark.evaluate   # regenerates BOTH files (needs Ollama running, see Setup)
   ```
   or, to get just the ML one without needing Ollama at all:
   ```bash
   python3 -m src.ml.train             # regenerates only ml_model.joblib
   ```
   (`src/hybrid/hybrid.py` run standalone trains a hybrid model in memory to
   print its learned weights, but doesn't save it to disk — only
   `src/benchmark/evaluate.py`'s full run does. If you only need the app
   working and don't care about re-running the benchmark, this distinction
   doesn't matter as long as both `.joblib` files already exist.)

### Start the app

```bash
cd /home/dannydev/my_drive/projects/rad
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

You'll see terminal output like:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```
Open that URL in a browser (Streamlit does not auto-open one in this
environment). The first load takes a few seconds while it loads spaCy and
the two models into memory (cached after that via `@st.cache_resource`, so
subsequent interactions are fast).

If you'd rather run it in the background and keep using the terminal:
```bash
nohup streamlit run app/streamlit_app.py --server.headless true > /tmp/streamlit.log 2>&1 &
```
then check `/tmp/streamlit.log` if anything looks wrong.

### Using the app

1. **Requirements text box** — paste your requirements here. One requirement
   per line works best (each line is scored independently). If you paste a
   single block of prose with no line breaks, it falls back to splitting on
   sentence boundaries via spaCy.
2. **...or upload a plain-text (.txt) file** — alternative to pasting;
   overrides whatever is in the text box. Only `.txt` is supported by
   design (see the Phase-1 scope decision to skip PDF/docx parsing).
3. **Scoring method to display** (radio: `hybrid` / `rules` / `ml`) — controls
   which method's flags decide the badges and severity score shown per
   sentence. Span highlighting always comes from the rule detector
   regardless of this choice, since only the rule-based method returns exact
   matched-text spans (that's inherent to the approach — ML/hybrid only
   output a category-level yes/no).
4. Click **Analyze**. For each requirement you'll get:
   - The sentence with ambiguous spans highlighted (colored by category —
     hover to see which category triggered it).
   - A severity indicator (🔴 × number of categories flagged, out of 5).
   - Colored badges naming each flagged category.
   - An expandable **"Compare all 3 methods for this sentence"** section
     showing rules/ML/hybrid side by side per category — useful for
     demoing exactly the kind of method disagreement discussed in the
     Results/Methodology sections above.

### Stopping / restarting

- Foreground run: `Ctrl+C` in the terminal it's running in.
- Background run (via `nohup` above): find and stop it with
  ```bash
  pkill -f "streamlit run app/streamlit_app.py"
  ```
- To pick up code changes in `app/streamlit_app.py`, Streamlit auto-detects
  the file change and prompts you to rerun (or just refresh the browser tab).
  Changes to `src/rules/detector.py` or the `rules/*.txt` lexicons are picked
  up on the next analysis too (no cache holds those). Changes that would
  affect the trained models themselves require re-running
  `src/benchmark.evaluate` (or `src/ml/train.py`) to regenerate the
  `.joblib` files, then restarting the app so `@st.cache_resource` reloads them.

### Troubleshooting

- **`FileNotFoundError: data/processed/ml_model.joblib`** — see
  Prerequisites above; you need to train/regenerate the model files first.
- **`Address already in use` / port 8501 busy** — another Streamlit instance
  is already running (possibly from an earlier session). Either open
  http://localhost:8501 directly (it's probably already serving what you
  want), or run `pkill -f streamlit` first, or pass a different port:
  `streamlit run app/streamlit_app.py --server.port 8502`.
- **`OSError: [E050] Can't find model 'en_core_web_sm'`** — run
  `python -m spacy download en_core_web_sm` inside the activated venv.
- **Blank page / connection refused in browser** — the server likely hasn't
  finished starting yet (give it a few seconds), or you're not running this
  from the same machine the browser is on (use the `Network URL` Streamlit
  prints instead of `localhost` if so).

## Results

Benchmarked on a 34-item held-out test set, with ground truth **independently
hand-verified against the taxonomy** (not derived from rule output — see
Methodology note below):

| Category | Method | P | R | F1 |
|---|---|---|---|---|
| vague_quantifier | rules | 0.86 | 0.86 | **0.86** |
| | ml | 0.50 | 0.29 | 0.36 |
| | llm | 0.67 | 0.57 | 0.62 |
| | hybrid | 0.86 | 0.86 | **0.86** |
| passive_no_actor | rules | 0.83 | 0.71 | **0.77** |
| | ml | 0.67 | 0.57 | 0.62 |
| | llm | 0.25 | 1.00 | 0.40 |
| | hybrid | 0.83 | 0.71 | **0.77** |
| unresolved_pronoun | rules | 0.25 | 1.00 | 0.40 |
| | ml | 1.00 | 1.00 | **1.00** |
| | llm | 0.17 | 1.00 | 0.29 |
| | hybrid | 0.33 | 1.00 | 0.50 |
| missing_measurable | rules | 1.00 | 0.60 | **0.75** |
| | ml | 0.50 | 0.20 | 0.29 |
| | llm | 0.75 | 0.60 | 0.67 |
| | hybrid | 1.00 | 0.60 | **0.75** |
| weak_conflicting_modality | all 4 methods | 1.00 | 1.00 | 1.00 |

**Macro-average:** rules 0.76, ML 0.65, LLM 0.59, **hybrid 0.78**.

`unresolved_pronoun` and `weak_conflicting_modality` each have very few
positive examples in this 34-item test set — treat those two rows as
high-variance, not precise.

## Methodology note (important for the report)

The labeling process was: rule-detector pre-fill → manual correction pass,
compressed to fit a ~5-hour build budget. This created a real risk that
surfaced during development and is worth stating explicitly:

- **Early results showed rules-only and hybrid scoring a perfect 1.00 F1 on
  every category.** This wasn't a win — the test labels had been bootstrapped
  from the rule detector's own output, so rules were just agreeing with
  themselves (label circularity). Fixed by independently re-judging the
  34 test-set sentences against the taxonomy text, blind to what the rule
  detector said (see `data/processed/test_human_verified.csv`).
- **Separately, the hybrid was found to reproduce rules-only predictions
  byte-for-byte** on the corrected test set too. Root cause: 0 of 135 training
  rows had any disagreement between the rule flag and the label, so logistic
  regression had no signal to ever learn to override the rule. Fixed by
  finding genuine rule/label disagreements (e.g. "This will allow the X
  application to..." — a clause-referring "This" the pronoun heuristic
  mis-flags) and upweighting those rows during training
  (`DISAGREEMENT_WEIGHT` in `src/hybrid/hybrid.py`). After this fix the hybrid
  generalized the pattern from 2 training examples to a held-out case the
  rules got wrong (`REQ-0159`).
- Training labels remain largely rule-derived (only a small number of rows
  were independently corrected); this is a real threat to validity, not
  hidden — state it as a limitation rather than presenting the benchmark
  numbers as a clean, independent evaluation of the rules.

## Known limitations

- **Pronoun resolution is a heuristic**, not real coreference resolution: it
  counts candidate noun-chunk antecedents (with a referent-noun allowlist for
  "they/them/their") rather than doing semantic disambiguation. Known
  remaining false positives include cases like "This provides the feed of
  recycled parts data" (clear antecedent, still occasionally over-flagged).
- **Lexicons are hand-curated and incomplete** (`rules/vague_quantifiers.txt`,
  `rules/performance_adjectives.txt`, `rules/referent_nouns.txt`,
  `rules/modal_strength.csv`) — extend them for other domains/corpora.
- **Small dataset overall** (169 labeled items, 135 train / 34 test) — CV and
  test metrics carry real variance; this was a deliberate scope cut for a
  ~5-hour build, not an oversight.
- **LLM baseline is deliberately simple** (single zero-shot prompt, no
  few-shot examples or chain-of-thought) — it's a comparison point, not tuned
  for best possible performance.

## Project structure

```
rules/                          editable lexicons (vague words, performance
                                 adjectives, referent nouns, modal strengths)
data/raw/promise_nfr.csv        source dataset (PROMISE NFR, 622 unique reqs)
data/processed/
  to_label.csv                  sampled batch for labeling
  prelabeled.csv                rule-detector pre-fill (all fields + spans)
  labels.csv                    final labels (169 rows)
  train.csv / test.csv          80/20 split
  test_human_verified.csv       independent test-set ground truth
  ml_model.joblib               trained TF-IDF+LR model
  hybrid_model.joblib           trained hybrid model
  benchmark_results.json        full benchmark predictions + scores
  llm_test_predictions.json     cached LLM baseline predictions
src/
  data.py                       PROMISE NFR loader
  sample_for_labeling.py        stratified sample for hand-labeling
  augment_rare_categories.py    oversample rare categories
  prelabel.py                   run rule detector over labeling batch
  finalize_labels.py            labels.csv + train/test split
  rules/detector.py             rule-based detector (5 checks)
  ml/train.py                   TF-IDF + logistic regression
  llm/ollama_classifier.py      Ollama zero-shot baseline
  hybrid/hybrid.py              feature-stacked hybrid
  benchmark/evaluate.py         benchmark harness
app/streamlit_app.py            Streamlit UI
```
