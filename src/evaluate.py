"""
evaluate.py  (Person 4's component)
====================================

Evaluation and final-prediction-selection logic for the Business Entity
Resolution pipeline.

This module owns everything downstream of the model's raw probabilities:
    - precision / recall / F-beta (F0.5) at pair level
    - per-source1_entity_id evaluation (the metric the project actually
      cares about is computed PER S1 entity, then averaged -- a global/
      micro precision-recall would let a few very common S1 entities
      dominate the score)
    - macro F0.5 across S1 entities
    - threshold search + best-threshold selection (VALIDATION DATA ONLY)
    - turning a chosen threshold into an actual match selection
    - building the final predictions.tsv-shaped output

--------------------------------------------------------------------------
Why F0.5, and why per-entity macro-averaging
--------------------------------------------------------------------------
F0.5 weights precision twice as heavily as recall. In this project, a
false positive (claiming S1 and a candidate are the same business when
they are not) is a worse failure than a false negative (missing one of
several true matches for an S1 entity) -- so precision is emphasized.

Macro-averaging over source1_entity_id (rather than one global precision/
recall over every pair) matches how the project is actually graded: each
S1 entity's match set is right or wrong on its own, regardless of how
many candidate pairs blocking happened to generate for it. Without
macro-averaging, S1 entities with unusually many candidates (e.g. very
common business names) would dominate a pooled/micro score.

--------------------------------------------------------------------------
Handling S1 entities with zero true matches (~5.6% of training S1 rows
per dataset_profile.txt) and S1 entities blocking failed to generate any
candidates for
--------------------------------------------------------------------------
Both cases are real and must not be dropped from evaluation or silently
scored as 0:
    - true=empty, predicted=empty  -> precision=1, recall=1, F0.5=1
      (correctly predicting "no match" is a correct prediction)
    - true=empty, predicted=nonempty -> F0.5=0 (every prediction is wrong)
    - true=nonempty, predicted=empty -> F0.5=0 (recall=0)
    - true=nonempty, predicted=nonempty -> ordinary precision/recall

The true-match COUNT per S1 entity is always taken from the ground truth
table directly (via features.build_training_labels), never merely from
however many label==1 rows happen to survive in a features/candidate
DataFrame. This matters because a true match that blocking never
retrieved as a candidate cannot appear as a label==1 row at all -- if we
counted true matches from the candidate/feature frame instead, blocking
misses would be invisible to recall instead of correctly counted as
false negatives.

Public API
----------
    precision_recall_fbeta(tp, fp, fn, beta=0.5)
    true_match_counts(ground_truth_df, group_ids, ...)
    evaluate_at_threshold(scored_df, true_counts, threshold, ...)
    search_thresholds(scored_df, true_counts, thresholds=None, ...)
    select_best_threshold(scored_df, ground_truth_df, group_ids, ...)
    select_matches(scored_df, threshold, ...)
    build_predictions(scored_df, all_source1_ids, threshold, ...)
    save_predictions(predictions_df, path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import config
from . import features

# ==========================================================================
# Column-name configuration (mirrors config.py / features.py; overridable
# via kwargs so this module never hard-codes a disagreement with them)
# ==========================================================================

S1_COL = config.S1_PAIR_COL
CAND_COL = config.CAND_PAIR_COL
LABEL_COL = "label"
PROBA_COL = "proba"

GT_S1_COL = config.GT_S1_COL
GT_MATCH_COL = config.GT_MATCH_COL

# Default beta for F-beta. The project spec calls for F0.5 (precision
# weighted twice as heavily as recall) everywhere in this module.
DEFAULT_BETA = 0.5

# Default threshold search grid. 99 candidate thresholds is fine-grained
# enough to find a good operating point without being so large that
# search_thresholds becomes the bottleneck of the pipeline; each candidate
# threshold costs one boolean filter + two groupby('source1_entity_id')
# calls over the validation frame, which is cheap relative to feature
# building / model training even at multi-million-row scale.
DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


# ==========================================================================
# Core metric: precision / recall / F-beta, vectorized and 0/0-safe
# ==========================================================================

def precision_recall_fbeta(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    *,
    beta: float = DEFAULT_BETA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized precision / recall / F-beta from per-group tp/fp/fn
    counts. Accepts scalars or numpy arrays (same shape). Never raises or
    produces NaN -- every 0/0 case is resolved by the convention below,
    chosen so that "correctly predicting no match" scores as a perfect
    F-beta rather than an undefined one:

        tp+fp == 0 (no predictions made for this group):
            precision := 1.0 if (tp+fn == 0) else 0.0
            (no predictions and no true matches -> vacuously correct;
             no predictions but true matches existed -> wrong, but this
             case's F-beta is forced to 0 by recall=0 regardless)
        tp+fn == 0 (no true matches for this group):
            recall := 1.0
            (nothing to find, so nothing was missed)
        precision + recall == 0:
            f_beta := 0.0
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    pred_total = tp + fp
    true_total = tp + fn

    precision = np.where(pred_total > 0, np.divide(tp, pred_total, out=np.zeros_like(tp), where=pred_total > 0), 0.0)
    # No predictions AND no true matches -> vacuously perfect precision.
    precision = np.where((pred_total == 0) & (true_total == 0), 1.0, precision)

    recall = np.where(true_total > 0, np.divide(tp, true_total, out=np.zeros_like(tp), where=true_total > 0), 1.0)

    beta2 = beta * beta
    denom = beta2 * precision + recall
    f_beta = np.where(denom > 0, (1 + beta2) * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)

    return precision, recall, f_beta


# ==========================================================================
# True-match counts per S1 entity (from ground truth, NOT from candidates)
# ==========================================================================

def true_match_counts(
    ground_truth_df: pd.DataFrame,
    group_ids: Iterable,
    *,
    gt_s1_col: str = GT_S1_COL,
    gt_match_col: str = GT_MATCH_COL,
) -> pd.Series:
    """Number of true matches per source1_entity_id, from ground truth
    directly. Reindexed onto `group_ids` (fill 0) so every requested S1
    id gets an entry -- including S1 ids with zero true matches AND S1
    ids blocking produced no candidates for at all (both must still
    participate in macro evaluation, see module docstring).
    """
    positives = features.build_training_labels(
        ground_truth_df, gt_s1_col=gt_s1_col, gt_match_col=gt_match_col
    )
    counts = positives.groupby(config.S1_PAIR_COL).size()
    group_ids = pd.Index(group_ids)
    return counts.reindex(group_ids, fill_value=0).astype(np.int64)


# ==========================================================================
# Per-entity confusion counts at a given threshold
# ==========================================================================

def _group_confusion_counts(
    scored_df: pd.DataFrame,
    true_counts: pd.Series,
    threshold: float,
    *,
    s1_col: str = S1_COL,
    label_col: str = LABEL_COL,
    proba_col: str = PROBA_COL,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """tp / fp / fn per source1_entity_id at one threshold, indexed
    exactly by true_counts.index (so every group in the evaluation set is
    represented, even ones with zero surviving candidate rows at this
    threshold -- or zero candidate rows at all).
    """
    idx = true_counts.index
    pred = scored_df.loc[scored_df[proba_col] >= threshold]

    pred_total = pred.groupby(s1_col).size().reindex(idx, fill_value=0)
    tp = (
        pred.loc[pred[label_col] == 1]
        .groupby(s1_col)
        .size()
        .reindex(idx, fill_value=0)
    )
    fp = pred_total - tp
    fn = true_counts - tp
    return tp.astype(np.int64), fp.astype(np.int64), fn.astype(np.int64)


def evaluate_at_threshold(
    scored_df: pd.DataFrame,
    true_counts: pd.Series,
    threshold: float,
    *,
    s1_col: str = S1_COL,
    label_col: str = LABEL_COL,
    proba_col: str = PROBA_COL,
    beta: float = DEFAULT_BETA,
) -> pd.DataFrame:
    """Per-source1_entity_id precision/recall/F-beta at one threshold.

    Returns a DataFrame indexed by source1_entity_id with columns
    [tp, fp, fn, precision, recall, f_beta], one row per id in
    true_counts.index.
    """
    tp, fp, fn = _group_confusion_counts(
        scored_df, true_counts, threshold, s1_col=s1_col, label_col=label_col, proba_col=proba_col
    )
    precision, recall, f_beta = precision_recall_fbeta(tp.values, fp.values, fn.values, beta=beta)
    return pd.DataFrame(
        {
            "tp": tp.values,
            "fp": fp.values,
            "fn": fn.values,
            "precision": precision,
            "recall": recall,
            "f_beta": f_beta,
        },
        index=true_counts.index,
    )


def macro_fbeta_at_threshold(
    scored_df: pd.DataFrame,
    true_counts: pd.Series,
    threshold: float,
    *,
    s1_col: str = S1_COL,
    label_col: str = LABEL_COL,
    proba_col: str = PROBA_COL,
    beta: float = DEFAULT_BETA,
) -> dict:
    """Convenience wrapper: macro-averaged precision/recall/F-beta (a
    single scalar summary) at one threshold, without materializing the
    full per-entity DataFrame. Used by search_thresholds for speed.
    """
    tp, fp, fn = _group_confusion_counts(
        scored_df, true_counts, threshold, s1_col=s1_col, label_col=label_col, proba_col=proba_col
    )
    precision, recall, f_beta = precision_recall_fbeta(tp.values, fp.values, fn.values, beta=beta)
    return {
        "threshold": threshold,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f_beta": float(np.mean(f_beta)),
        "n_entities": int(len(true_counts)),
    }


# ==========================================================================
# Threshold search + best-threshold selection
# ==========================================================================

def search_thresholds(
    scored_df: pd.DataFrame,
    true_counts: pd.Series,
    *,
    thresholds: Optional[np.ndarray] = None,
    s1_col: str = S1_COL,
    label_col: str = LABEL_COL,
    proba_col: str = PROBA_COL,
    beta: float = DEFAULT_BETA,
) -> pd.DataFrame:
    """Macro precision/recall/F-beta for every candidate threshold.

    `scored_df` and `true_counts` MUST come from the validation split
    only -- this function has no way to enforce that itself, so the
    caller (select_best_threshold / main.py) is responsible for never
    passing test data here. See module docstring / main.py for the
    leakage-prevention contract.

    Returns a DataFrame with columns
    [threshold, macro_precision, macro_recall, macro_f_beta, n_entities],
    one row per candidate threshold, sorted by threshold ascending.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_GRID
    rows = [
        macro_fbeta_at_threshold(
            scored_df, true_counts, float(t),
            s1_col=s1_col, label_col=label_col, proba_col=proba_col, beta=beta,
        )
        for t in thresholds
    ]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_best_threshold(
    scored_val_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    val_group_ids: Iterable,
    *,
    thresholds: Optional[np.ndarray] = None,
    s1_col: str = S1_COL,
    label_col: str = LABEL_COL,
    proba_col: str = PROBA_COL,
    gt_s1_col: str = GT_S1_COL,
    gt_match_col: str = GT_MATCH_COL,
    beta: float = DEFAULT_BETA,
) -> Tuple[float, pd.DataFrame]:
    """Select the threshold that maximizes macro F-beta on VALIDATION
    data only.

    Parameters
    ----------
    scored_val_df : validation candidate pairs with columns
        [s1_col, label_col, proba_col] (the output of attach_labels(...)
        plus a model.predict_proba(...) column). Must NOT include any
        test data -- this is the leakage-prevention boundary of the
        whole pipeline: the threshold is a hyperparameter, and tuning it
        on test data would leak test information into the pipeline's
        decisions.
    ground_truth_df : the TRAINING ground truth table (validation ids are
        a subset of the grouped train/val split over training data;
        there is no "validation ground truth" file of its own).
    val_group_ids : the source1_entity_id values assigned to the
        validation split by model.grouped_train_val_split(...).

    Returns
    -------
    (best_threshold, search_results_df) -- search_results_df is the full
    per-threshold table from search_thresholds(...), so callers/log
    output can show the whole curve, not just the argmax.
    """
    tcounts = true_match_counts(ground_truth_df, val_group_ids, gt_s1_col=gt_s1_col, gt_match_col=gt_match_col)
    results = search_thresholds(
        scored_val_df, tcounts,
        thresholds=thresholds, s1_col=s1_col, label_col=label_col, proba_col=proba_col, beta=beta,
    )
    if results.empty:
        raise ValueError("search_thresholds returned no rows; check the thresholds grid")
    best_row = results.loc[results["macro_f_beta"].idxmax()]
    return float(best_row["threshold"]), results


# ==========================================================================
# Match selection + final predictions.tsv-shaped output
# ==========================================================================

def select_matches(
    scored_df: pd.DataFrame,
    threshold: float,
    *,
    proba_col: str = PROBA_COL,
) -> pd.DataFrame:
    """Filter scored candidate pairs down to the ones selected as matches
    at `threshold` (proba >= threshold). No ranking/top-1 logic is
    applied -- an S1 entity may legitimately have zero, one, or several
    matches (per dataset_profile.txt, real S1 entities have anywhere from
    0 to 11 true matches), so every pair clearing the threshold is kept.
    """
    return scored_df.loc[scored_df[proba_col] >= threshold].reset_index(drop=True)


def build_predictions(
    scored_df: pd.DataFrame,
    all_source1_ids: Iterable,
    threshold: float,
    *,
    s1_col: str = S1_COL,
    cand_col: str = CAND_COL,
    proba_col: str = PROBA_COL,
    out_s1_col: str = GT_S1_COL,
    out_match_col: str = GT_MATCH_COL,
) -> pd.DataFrame:
    """Build the final predictions table in the project's required output
    format: one row per S1 entity (source1_entity_id, matched_entity_ids
    as a comma-joined string, "" when there are no predicted matches) --
    the SAME shape as the ground truth table loaded by
    io.load_ground_truth, which is what makes predictions.tsv directly
    comparable to ground truth.

    Every id in `all_source1_ids` gets exactly one output row, including
    ids blocking produced zero candidates for at all -- those get "".
    """
    matched = select_matches(scored_df, threshold, proba_col=proba_col)
    joined = (
        matched.groupby(s1_col)[cand_col]
        .apply(lambda ids: ",".join(sorted(ids)))
    )
    all_ids = pd.Index(pd.unique(pd.Series(list(all_source1_ids))))
    result = pd.DataFrame({out_s1_col: all_ids})
    result[out_match_col] = result[out_s1_col].map(joined).fillna("")
    return result


def save_predictions(
    predictions_df: pd.DataFrame,
    path: Union[str, Path],
    *,
    sep: str = config.TSV_SEPARATOR,
) -> None:
    """Write the final predictions table as a TSV, matching io.py's
    read conventions (plain tab-separated, string values, no index).
    Creates the parent directory if it does not already exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(path, sep=sep, index=False)


# ==========================================================================
# Self-test / validation (run with: python -m src.evaluate)
# ==========================================================================

if __name__ == "__main__":
    # Small synthetic scored validation set exercising every edge case
    # called out in the module docstring: exact match, partial recall,
    # a false positive, a true-zero-match entity, and an entity blocking
    # produced no candidates for at all (so it never appears in
    # scored_df, only in true_match_counts via ground truth).
    scored = pd.DataFrame(
        {
            S1_COL: ["S1-1", "S1-1", "S1-2", "S1-2", "S1-3", "S1-4"],
            CAND_COL: ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S2-6"],
            LABEL_COL: [1, 0, 1, 1, 0, 0],
            PROBA_COL: [0.92, 0.10, 0.81, 0.55, 0.70, 0.95],
        }
    )
    # Ground truth: S1-1 has 1 true match (S2-1), S1-2 has 2 (S2-3,S2-4),
    # S1-3 has 0 true matches, S1-4 has 0 true matches (but the model
    # scored a candidate 0.95 for it -- a false positive above most
    # thresholds), S1-5 has 1 true match but blocking never retrieved a
    # candidate for it at all (missing from `scored` entirely).
    gt = pd.DataFrame(
        {
            GT_S1_COL: ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
            GT_MATCH_COL: ["S2-1", "S2-3,S2-4", "", "", "S2-99"],
        }
    )
    val_ids = ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5"]

    tcounts = true_match_counts(gt, val_ids)
    assert tcounts.loc["S1-1"] == 1
    assert tcounts.loc["S1-2"] == 2
    assert tcounts.loc["S1-3"] == 0
    assert tcounts.loc["S1-5"] == 1  # true match exists even with no candidate row

    per_entity = evaluate_at_threshold(scored, tcounts, threshold=0.5)
    # S1-1: pred={S2-1(tp),S2-2 filtered out at 0.5 since 0.10<0.5} -> tp=1,fp=0,fn=0 -> P=1,R=1,F=1
    assert per_entity.loc["S1-1", "tp"] == 1
    assert per_entity.loc["S1-1", "fp"] == 0
    assert per_entity.loc["S1-1", "fn"] == 0
    # S1-2: both candidates clear 0.5 and both are true positives -> perfect
    assert per_entity.loc["S1-2", "tp"] == 2
    assert per_entity.loc["S1-2", "f_beta"] == 1.0
    # S1-3: no true matches, one candidate scored 0.70 clears threshold -> false positive -> F=0
    assert per_entity.loc["S1-3", "fp"] == 1
    assert per_entity.loc["S1-3", "f_beta"] == 0.0
    # S1-4: no true matches, candidate at 0.95 clears threshold -> false positive -> F=0
    assert per_entity.loc["S1-4", "fp"] == 1
    assert per_entity.loc["S1-4", "f_beta"] == 0.0
    # S1-5: true match exists, no candidate row at all -> recall=0 -> F=0
    assert per_entity.loc["S1-5", "fn"] == 1
    assert per_entity.loc["S1-5", "f_beta"] == 0.0

    # Threshold search should prefer a threshold that keeps S1-1/S1-2's
    # true positives while rejecting S1-3/S1-4's false positives (which
    # score 0.70 and 0.95) -- e.g. somewhere above 0.81 up to 0.92 keeps
    # S1-1 (0.92) and drops S1-3 (0.70), but a max grid of 0.01 steps will
    # find the best trade-off automatically; just check it beats 0.5.
    best_t, curve = select_best_threshold(scored, gt, val_ids, thresholds=np.round(np.arange(0.05, 1.0, 0.05), 2))
    score_at_best = curve.loc[np.isclose(curve["threshold"], best_t), "macro_f_beta"].iloc[0]
    score_at_half = macro_fbeta_at_threshold(scored, tcounts, 0.5)["macro_f_beta"]
    assert score_at_best >= score_at_half

    preds = build_predictions(scored, val_ids, threshold=best_t)
    assert set(preds[GT_S1_COL]) == set(val_ids)
    assert preds.loc[preds[GT_S1_COL] == "S1-5", GT_MATCH_COL].iloc[0] == ""

    print("=== evaluate.py self-test ===")
    print(curve.to_string(index=False))
    print(f"\nbest_threshold = {best_t}")
    print(per_entity)
    print("\nAll assertions passed.")