"""
main.py  (Person 4's component)
================================

End-to-end pipeline wiring for the Business Entity Resolution project.
This module contains NO modeling/feature/blocking/normalization logic of
its own -- it only calls the existing teammate modules in the right
order, with the right leakage-prevention boundaries:

    io          -> load raw TSVs
    normalize   -> Unicode-safe *_norm columns (Person 1)
    blocking    -> candidate generation (Person 2)
    features    -> candidate pairs -> numeric feature matrix (Person 3)
    model       -> pairwise classifier (Person 3)
    evaluate    -> threshold selection + final prediction shape (Person 4)

Pipeline (see also README.md):

    load train S1/S2/S3 + ground truth, load test S1/S2/S3
        -> normalize every source table
        -> generate train candidates (S1xS2 union S1xS3)
        -> generate test candidates  (S1xS2 union S1xS3)
        -> build train features, attach ground-truth labels
        -> grouped train/val split on source1_entity_id
        -> train an initial model on the train split only
        -> score the val split, select best threshold via macro F0.5
           (VALIDATION DATA ONLY -- see evaluate.select_best_threshold)
        -> retrain a final model on ALL labeled training data
           (train+val combined -- safe now that the threshold decision
           has already been locked in on validation alone)
        -> build test features, score with the final model
        -> apply the validation-selected threshold to test scores
        -> write predictions.tsv in the ground-truth-shaped format

Run with:  python run.py   (see run.py -- it just calls run_pipeline()
below and does not duplicate any of this logic).
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pandas as pd

from . import config
from . import io
from . import normalize
from . import blocking
from . import features
from . import model
from . import evaluate


# ==========================================================================
# Pipeline-local tunables
# ==========================================================================
# These are NOT modeling/feature/blocking parameters (those all live in
# their owning modules' own defaults) -- they are pipeline-wiring choices
# that belong to Person 4's integration layer.

# Fraction of training source1_entity_id groups held out for threshold
# selection. Passed straight through to model.grouped_train_val_split,
# whose own default (0.15) would be used anyway if this were omitted --
# named here so the leakage boundary is visible in one place.
VAL_FRACTION = 0.15

# Max candidate matches retrieved per S1 row by blocking's TF-IDF stage
# (blocking.generate_candidates(..., top_k=...)). Passed through
# unchanged to Person 2's function; named here rather than passed as a
# bare literal at each call site.
BLOCKING_TOP_K = 50


def _log(msg: str, *, start_time: float = None) -> float:
    """Timestamped progress logging. The full pipeline runs over
    multi-million-row source tables (see data/dataset_profile.txt), so
    stage-by-stage progress output matters for anyone watching a real run
    -- silence for tens of minutes with no output looks identical to a
    hang. Returns the current time so callers can chain elapsed-time
    reporting: `t = _log("stage", start_time=t)`.
    """
    now = time.time()
    if start_time is not None:
        print(f"[{now:.0f}] {msg} ({now - start_time:.1f}s)")
    else:
        print(f"[{now:.0f}] {msg}")
    return now


# ==========================================================================
# Stage 1: load
# ==========================================================================

def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train S1/S2/S3 and ground truth from the paths in config.py."""
    s1 = io.load_source(config.TRAIN_SOURCE1_PATH, source_prefix=config.S1_PREFIX)
    s2 = io.load_source(config.TRAIN_SOURCE2_PATH, source_prefix=config.S2_PREFIX)
    s3 = io.load_source(config.TRAIN_SOURCE3_PATH, source_prefix=config.S3_PREFIX)
    gt = io.load_ground_truth(config.TRAIN_GROUND_TRUTH_PATH)
    return s1, s2, s3, gt


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load test S1/S2/S3 from the paths in config.py. Test ground truth
    is never loaded by this pipeline -- it does not exist as a file the
    pipeline is allowed to see (per the "do not use test ground truth"
    requirement); scoring against it, if desired, is a separate,
    out-of-band step for whoever holds the answer key.
    """
    s1 = io.load_source(config.TEST_SOURCE1_PATH, source_prefix=config.S1_PREFIX)
    s2 = io.load_source(config.TEST_SOURCE2_PATH, source_prefix=config.S2_PREFIX)
    s3 = io.load_source(config.TEST_SOURCE3_PATH, source_prefix=config.S3_PREFIX)
    return s1, s2, s3


# ==========================================================================
# Stage 2: normalize
# ==========================================================================

def normalize_all(*sources: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Apply normalize.normalize_source to every given source table.
    Per normalize.py's own integration note, this MUST run before
    blocking/features -- both of the latter assume *_norm columns exist
    and blocking.generate_candidates raises ValueError otherwise.
    """
    return tuple(normalize.normalize_source(s) for s in sources)


# ==========================================================================
# Stage 3: candidate generation (blocking)
# ==========================================================================

def generate_all_candidates(
    s1_norm: pd.DataFrame, s2_norm: pd.DataFrame, s3_norm: pd.DataFrame, *, top_k: int = BLOCKING_TOP_K
) -> pd.DataFrame:
    """Union of candidates generated against source2 and source3
    separately (blocking.generate_candidates has no S2/S3-specific
    branching -- it is called once per candidate table, per its own
    docstring), deduplicated on (source1_entity_id, candidate_entity_id).
    S2 and S3 ids never collide (disjoint "S2-"/"S3-" prefixes), so a
    plain concat + drop_duplicates is sufficient and exact.
    """
    from_s2 = blocking.generate_candidates(s1_norm, s2_norm, top_k=top_k)
    from_s3 = blocking.generate_candidates(s1_norm, s3_norm, top_k=top_k)
    parts = [p for p in (from_s2, from_s3) if not p.empty]
    if not parts:
        return pd.DataFrame({config.S1_PAIR_COL: pd.Series(dtype=object),
                              config.CAND_PAIR_COL: pd.Series(dtype=object)})
    combined = pd.concat(parts, axis=0, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=[config.S1_PAIR_COL, config.CAND_PAIR_COL], ignore_index=True
    )
    return combined


# ==========================================================================
# Stage 4: features (normalized-text wiring -- see normalize.py's
# integration note: already_normalized=True + the *_norm column names,
# never the default raw-text fallback path)
# ==========================================================================

def build_features_for(
    pairs_df: pd.DataFrame, s1_norm: pd.DataFrame, s2_norm: pd.DataFrame, s3_norm: pd.DataFrame
) -> pd.DataFrame:
    if pairs_df.empty:
        # Handle empty candidate sets safely: build_features already
        # returns a correctly-shaped empty frame for an empty pairs_df,
        # so no special casing is needed beyond letting it through.
        pass
    return features.build_features(
        pairs_df, s1_norm, s2_norm, s3_norm,
        already_normalized=True,
        name_col=config.NAME_NORM_COL,
        addr_col=config.ADDR_NORM_COL,
        country_col=config.COUNTRY_NORM_COL,
    )


# ==========================================================================
# Stage 5: train / validation split (leakage prevention)
# ==========================================================================

def split_train_val(
    labeled_df: pd.DataFrame, *, val_frac: float = VAL_FRACTION, random_state: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Group-wise split on source1_entity_id (never a row-wise split --
    that would leak, since multiple candidate pairs share an S1 id and
    the per-entity evaluation would then mix train and val pairs for the
    same entity). Returns (train_df, val_df, train_ids, val_ids).
    """
    train_ids, val_ids = model.grouped_train_val_split(
        labeled_df, group_col=config.S1_PAIR_COL, val_frac=val_frac, random_state=random_state
    )
    train_df = labeled_df[labeled_df[config.S1_PAIR_COL].isin(train_ids)].reset_index(drop=True)
    val_df = labeled_df[labeled_df[config.S1_PAIR_COL].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df, train_ids, val_ids


# ==========================================================================
# Stage 6: train a classifier on a labeled+sampled frame
# ==========================================================================

def fit_classifier(labeled_df: pd.DataFrame, *, random_state: int = config.RANDOM_SEED) -> model.PairwiseMatcher:
    """Sample (positives + hard/ordinary negatives, per model.py) then
    fit a PairwiseMatcher. Shared by both the initial (train-split-only)
    model and the final (train+val) model -- identical logic, different
    input frame, so it is written once here rather than duplicated.
    """
    sampled = model.sample_training_pairs(labeled_df, group_col=config.S1_PAIR_COL, random_state=random_state)
    clf = model.train_model(
        sampled[features.FEATURE_COLUMNS], sampled["label"],
        feature_columns=features.FEATURE_COLUMNS, random_state=random_state,
    )
    return clf


# ==========================================================================
# Full pipeline
# ==========================================================================

def run_pipeline() -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """Run the complete pipeline end to end and write predictions.tsv.

    Returns (predictions_df, best_threshold, threshold_search_results) so
    callers (run.py, tests, notebooks) can inspect what happened without
    re-reading the file back off disk.
    """
    t0 = t = _log("Loading training data...")
    train_s1, train_s2, train_s3, train_gt = load_train_data()
    t = _log(
        f"Loaded train S1={len(train_s1):,} S2={len(train_s2):,} S3={len(train_s3):,} "
        f"ground_truth={len(train_gt):,}", start_time=t,
    )

    t = _log("Loading test data...")
    test_s1, test_s2, test_s3 = load_test_data()
    t = _log(f"Loaded test S1={len(test_s1):,} S2={len(test_s2):,} S3={len(test_s3):,}", start_time=t)

    t = _log("Normalizing all source tables...")
    train_s1n, train_s2n, train_s3n = normalize_all(train_s1, train_s2, train_s3)
    test_s1n, test_s2n, test_s3n = normalize_all(test_s1, test_s2, test_s3)
    t = _log("Normalization complete.", start_time=t)

    t = _log("Generating training candidates (blocking)...")
    train_pairs = generate_all_candidates(train_s1n, train_s2n, train_s3n)
    t = _log(f"Training candidate pairs: {len(train_pairs):,}", start_time=t)

    t = _log("Generating test candidates (blocking)...")
    test_pairs = generate_all_candidates(test_s1n, test_s2n, test_s3n)
    t = _log(f"Test candidate pairs: {len(test_pairs):,}", start_time=t)

    t = _log("Building training features...")
    train_features = build_features_for(train_pairs, train_s1n, train_s2n, train_s3n)
    t = _log(f"Training feature rows: {len(train_features):,}", start_time=t)

    t = _log("Attaching ground-truth labels...")
    labeled = features.attach_labels(train_features, train_gt)
    n_pos = int((labeled["label"] == 1).sum())
    t = _log(f"Labeled rows: {len(labeled):,} (positives: {n_pos:,})", start_time=t)
    if n_pos == 0:
        raise RuntimeError(
            "No positive pairs survived blocking + labeling. Check that "
            "blocking.generate_candidates is retrieving true matches "
            "(see blocking.py's recall-oriented design) before training."
        )

    t = _log("Splitting train/validation by source1_entity_id...")
    train_df, val_df, train_ids, val_ids = split_train_val(labeled)
    t = _log(
        f"Train groups: {len(train_ids):,} rows: {len(train_df):,} | "
        f"Val groups: {len(val_ids):,} rows: {len(val_df):,}", start_time=t,
    )

    t = _log("Training initial model (train split only)...")
    initial_clf = fit_classifier(train_df)
    t = _log(f"Initial model trained (backend={initial_clf.backend_name}).", start_time=t)

    t = _log("Scoring validation split...")
    val_scored = val_df[[config.S1_PAIR_COL, config.CAND_PAIR_COL, "label"]].copy()
    val_scored[evaluate.PROBA_COL] = initial_clf.predict_proba(val_df[features.FEATURE_COLUMNS])
    t = _log("Validation scored.", start_time=t)

    t = _log("Selecting best threshold via validation macro F0.5 (test data NOT used)...")
    best_threshold, threshold_curve = evaluate.select_best_threshold(val_scored, train_gt, val_ids)
    best_row = threshold_curve.loc[np.isclose(threshold_curve["threshold"], best_threshold)].iloc[0]
    t = _log(
        f"Best threshold={best_threshold:.2f} "
        f"(macro F0.5={best_row['macro_f_beta']:.4f}, "
        f"precision={best_row['macro_precision']:.4f}, "
        f"recall={best_row['macro_recall']:.4f})", start_time=t,
    )

    t = _log("Training final model on all labeled training data (train+val)...")
    final_clf = fit_classifier(labeled)
    t = _log(f"Final model trained (backend={final_clf.backend_name}).", start_time=t)

    t = _log("Building test features...")
    test_features = build_features_for(test_pairs, test_s1n, test_s2n, test_s3n)
    t = _log(f"Test feature rows: {len(test_features):,}", start_time=t)

    t = _log("Scoring test candidates with final model...")
    test_scored = test_features[[config.S1_PAIR_COL, config.CAND_PAIR_COL]].copy()
    if len(test_features):
        test_scored[evaluate.PROBA_COL] = final_clf.predict_proba(test_features[features.FEATURE_COLUMNS])
    else:
        # Handle empty candidate sets safely: no test candidates at all
        # (e.g. a tiny smoke-test run) still yields a well-formed,
        # empty-but-typed proba column rather than a crash.
        test_scored[evaluate.PROBA_COL] = pd.Series(dtype=np.float32)
    t = _log("Test candidates scored.", start_time=t)

    t = _log("Applying validation-selected threshold and building final predictions...")
    predictions = evaluate.build_predictions(test_scored, test_s1[config.ID_COL], best_threshold)
    n_with_matches = int((predictions[config.GT_MATCH_COL] != "").sum())
    t = _log(
        f"Predictions built: {len(predictions):,} S1 entities "
        f"({n_with_matches:,} with >=1 predicted match).", start_time=t,
    )

    evaluate.save_predictions(predictions, config.PREDICTIONS_OUTPUT_PATH)
    _log(f"Saved predictions to {config.PREDICTIONS_OUTPUT_PATH}", start_time=t0)

    return predictions, best_threshold, threshold_curve


if __name__ == "__main__":
    run_pipeline()