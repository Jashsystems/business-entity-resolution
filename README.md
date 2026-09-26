# Business Entity Resolution

Match businesses listed in a primary directory (**Source 1**) against two
other business directories (**Source 2** and **Source 3**), when the same
real-world business may be listed under slightly different names,
addresses, or spellings across all three sources.

For example, Source 1 might list `"O'Reilly's Pub"` at `"100 Main St"`,
while Source 2 lists the same place as `"Oreillys Pub"` at
`"100 Main Street"`. The goal of this project is to automatically find
these pairs (and reject look-alikes that are actually different
businesses) across millions of rows.

## Project purpose

Given:
- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv` — three
  business directories for training
- `train_ground_truth.tsv` — the correct Source-2/Source-3 matches for
  every Source-1 training row
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` — the same
  three directories for testing, **with no ground truth provided**

the pipeline trains a matcher on the training data and produces
`predictions.tsv`: for every Source-1 test entity, the list of
Source-2/Source-3 entities the model believes are the same business.

## Project structure

```text
business-entity-resolution/
├── run.py                     # entry point: python run.py
├── requirements.txt
├── README.md
├── data/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   └── predictions.tsv        # written by the pipeline
└── src/
    ├── config.py               # Person 1 — paths, column names, constants
    ├── io.py                   # Person 1 — TSV loading
    ├── normalize.py             # Person 1 — Unicode-safe text normalization
    ├── blocking.py              # Person 2 — candidate generation
    ├── features.py               # Person 3 — pairwise feature engineering
    ├── model.py                  # Person 3 — pairwise classifier
    ├── evaluate.py                # Person 4 — metrics, threshold selection
    └── main.py                    # Person 4 — pipeline wiring
```

## What each component does

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for file paths, column names, source-id prefixes (`S1-`/`S2-`/`S3-`), and the random seed. Every other module reads from here instead of repeating literals. |
| `io.py` | Reads the raw TSVs. Keeps `entity_id` and all text fields as exact strings (never coerced to numbers), and represents missing values as `""` — never the literal text `"nan"`. |
| `normalize.py` | Produces `name_norm` / `address_norm` / `country_norm` columns: NFKC + casefold + Unicode-safe punctuation stripping that works correctly on non-Latin scripts (Hindi, Kannada, etc.) and accented text, without destroying them. |
| `blocking.py` | For every Source-1 row, retrieves a manageable set of Source-2/Source-3 candidates instead of comparing every row to every other row (which is infeasible at millions × millions scale). Combines four independent strategies (exact name, name+country, TF-IDF nearest-neighbor name similarity, and address-token blocking) to keep recall high. |
| `features.py` | Turns each candidate `(source1, candidate)` pair into a fixed-length numeric feature vector: name/address similarity scores, exact-match flags, postal/house-number agreement, missing-value indicators, and a few interaction features. |
| `model.py` | A pairwise binary classifier (LightGBM → XGBoost → scikit-learn `HistGradientBoostingClassifier`, whichever is installed) that turns a feature vector into `P(same business)`. It deliberately does **not** pick a decision threshold or force a fixed number of matches per entity — that is `evaluate.py`'s job. |
| `evaluate.py` | Precision/recall/F0.5, computed **per Source-1 entity** and macro-averaged; searches candidate probability thresholds on validation data only; turns a chosen threshold into the final predictions table. |
| `main.py` | Wires all of the above into one pipeline, in the correct order, with the train/validation split done before threshold selection so no test data ever influences a modeling decision. |
| `run.py` | `python run.py` — the only command you need to run the whole thing. |

## Person 1, Person 2, Person 3, and Person 4 responsibilities

- **Person 1 — Data & Normalization**: `config.py`, `io.py`, `normalize.py`.
  Loading TSVs correctly (Unicode, missing values, no numeric coercion of
  ids) and producing normalized text that's safe to compare across
  scripts and languages.
- **Person 2 — Blocking**: `blocking.py`. Turning "2.2M × 5M+ possible
  pairs" into a small, high-recall candidate set per Source-1 entity.
- **Person 3 — Features & Model**: `features.py`, `model.py`. Turning
  candidate pairs into numeric features and training the classifier that
  scores them.
- **Person 4 — Evaluation & Integration** (this part of the repo):
  `evaluate.py`, `main.py`, `run.py`. Deciding what "correct" means
  (precision/recall/F0.5 per Source-1 entity), picking the probability
  threshold that separates a predicted match from a non-match using
  validation data only, wiring every teammate's module into one
  pipeline, and producing the final `predictions.tsv`.

## Requirements

- Python 3.9 or newer
- See `requirements.txt`. In short:
  - **Required**: `pandas`, `numpy`, `scikit-learn` (scikit-learn is a
    hard dependency of `blocking.py`, not optional)
  - **Optional, faster if installed, otherwise an automatic fallback is
    used**: `rapidfuzz` (faster string similarity), `lightgbm` or
    `xgboost` (faster/stronger model backend than the scikit-learn
    fallback)

## Installation

```bash
# 1. Clone/enter the project
cd business-entity-resolution

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional, recommended for real-size data) install the faster backends
pip install rapidfuzz lightgbm
```

To leave the virtual environment later: `deactivate`.

## Data directory structure

By default the pipeline looks for data under `<project root>/data/`:

```text
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Every `*_source*.tsv` file must have these tab-separated columns:
`entity_id`, `business_name`, `business_address`, `country`.

`train_ground_truth.tsv` must have: `source1_entity_id`,
`matched_entity_ids` (a comma-separated list of Source-2/Source-3 ids, or
an empty string if that Source-1 entity has no true match).

If your data lives somewhere else, set the `BER_DATA_DIR` environment
variable instead of editing any code:

```bash
export BER_DATA_DIR=/path/to/your/data
python run.py
```

Output location is similarly configurable via `BER_OUTPUT_DIR` (default:
`<project root>/output/`).

## How to run the project

```bash
python run.py
```

This runs the full pipeline end to end: load → normalize → generate
candidates → build features → train → select threshold → retrain on all
labeled data → score the test set → write `output/predictions.tsv`.

Progress is logged to stdout stage by stage (with row counts and elapsed
time per stage), since the real dataset has multiple millions of rows
per file and a silent multi-minute stage would otherwise look identical
to a hang.

## How to run the blocking self-test

`blocking.py` ships its own self-contained test using a handful of
synthetic rows (no real data files needed):

```bash
python -m src.blocking
```

Expect to see `All assertions passed.` at the end. `normalize.py` and
`evaluate.py` have the same kind of self-test, runnable the same way:

```bash
python -m src.normalize
python -m src.evaluate
```

## How evaluation works

Matching is evaluated **per Source-1 entity**, not as one global
precision/recall over every candidate pair. For each Source-1 entity:

1. Take the set of candidate ids the model predicted as matches (its
   probability was at or above the chosen threshold).
2. Compare against the true match set from ground truth.
3. Compute that entity's own precision, recall, and F0.5.

These per-entity F0.5 scores are then averaged across all Source-1
entities (**macro-averaging**) to get one overall score. This matters
because some Source-1 entities have far more candidates than others (a
very common business name generates many candidate pairs); a pooled
score would let those entities dominate the result, while macro
averaging weights every Source-1 entity equally, which better reflects
"did we get this business right?" for every business individually —
including the ~5.6% of entities with zero true matches, where the
correct prediction is an empty set.

## Why F0.5 is used

F0.5 weights precision twice as heavily as recall
(`F_beta = (1 + beta²) · P · R / (beta² · P + R)`, `beta = 0.5`). In this
project, a **false positive** — claiming two different businesses are
the same — is treated as a worse mistake than a **false negative** —
missing one of several true matches for a business that has other
correctly-found matches. F0.5 reflects that asymmetry; plain F1 would
weight both mistakes equally, and recall-oriented F2 would reward
over-predicting matches.

## How threshold selection works

The model (`model.py`) outputs a raw probability per candidate pair and
deliberately does **not** decide what counts as a match — that decision
(the probability cutoff) is made by `evaluate.py`, and only ever on the
**validation split**:

1. Training data is split by `source1_entity_id` (not by row) into a
   train set and a validation set, so every candidate pair for a given
   Source-1 entity stays entirely on one side of the split.
2. A model is trained on the train split only.
3. That model scores the validation split.
4. `evaluate.py` tries a grid of probability thresholds against the
   validation predictions and picks whichever threshold gives the
   highest macro F0.5 **on validation data**.
5. Only after that threshold is locked in does the pipeline retrain a
   final model on *all* labeled training data (train + validation
   combined) and apply the already-chosen threshold to the test set.

Test data and test ground truth are never used to pick the threshold —
in fact test ground truth is never loaded by this pipeline at all, by
design.

## How to generate predictions

Predictions are a side effect of running the full pipeline:

```bash
python run.py
```

There is no separate "predict only" command in this version of the
project — `main.py`'s `run_pipeline()` always trains, selects a
threshold, and predicts in one call, since a saved model on disk would
still need its accompanying threshold and feature-building code to be
useful. If you want to reuse a trained model without retraining, see
`model.PairwiseMatcher.save` / `.load` in `model.py`, and call
`evaluate.build_predictions(...)` directly with pre-computed test
features.

## Where predictions.tsv is saved

`output/predictions.tsv` by default (`config.PREDICTIONS_OUTPUT_PATH`,
i.e. `<BER_OUTPUT_DIR or <project root>/output>/predictions.tsv`). The
parent directory is created automatically if it does not exist.

## Expected output format

Tab-separated, one row per **test** Source-1 entity — including entities
with zero predicted matches — in the same shape as
`train_ground_truth.tsv`:

```text
source1_entity_id	matched_entity_ids
S1-1	S2-4821,S3-1193
S1-2	
S1-3	S2-77
```

- `matched_entity_ids` is a comma-separated list of predicted
  Source-2/Source-3 entity ids, sorted for determinism.
- An empty string means the model predicts that Source-1 entity has no
  match in Source 2 or Source 3 (a valid, expected outcome — see the
  training ground truth's own ~5.6% zero-match rate).
- Every `entity_id` from `test_source1.tsv` appears exactly once, even
  if blocking produced no candidates for it at all.