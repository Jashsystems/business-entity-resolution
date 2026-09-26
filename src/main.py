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

Pipeline:

    load train S1/S2/S3 + ground truth, load test S1/S2/S3
        -> normalize every source table
        -> generate train candidates (S1xS2 union S1xS3)
        -> generate test candidates  (S1xS2 union S1xS3)
        -> build train features, attach ground-truth labels
        -> grouped train/val split on source1_entity_id
        -> train an initial model on the train split only
        -> score the val split, select best threshold via macro F0.5
        -> retrain a final model on ALL labeled training data
        -> build test features, score with the final model
        -> apply the validation-selected threshold to test scores
        -> write matching_results.tsv
        -> write candidate_pairs.tsv

Run with:

    python run.py
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

VAL_FRACTION = 0.15

BLOCKING_TOP_K = 50


def _log(msg: str, *, start_time: float = None) -> float:
    """Timestamped progress logging."""
    now = time.time()

    if start_time is not None:
        print(f"[{now:.0f}] {msg} ({now - start_time:.1f}s)")
    else:
        print(f"[{now:.0f}] {msg}")

    return now


# ==========================================================================
# Stage 1: load
# ==========================================================================

def load_train_data() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load train S1/S2/S3 and ground truth from config paths."""

    s1 = io.load_source(
        config.TRAIN_SOURCE1_PATH,
        source_prefix=config.S1_PREFIX,
    )

    s2 = io.load_source(
        config.TRAIN_SOURCE2_PATH,
        source_prefix=config.S2_PREFIX,
    )

    s3 = io.load_source(
        config.TRAIN_SOURCE3_PATH,
        source_prefix=config.S3_PREFIX,
    )

    gt = io.load_ground_truth(
        config.TRAIN_GROUND_TRUTH_PATH,
    )

    return s1, s2, s3, gt


def load_test_data() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load test S1/S2/S3.

    Test ground truth is never loaded by this pipeline.
    """

    s1 = io.load_source(
        config.TEST_SOURCE1_PATH,
        source_prefix=config.S1_PREFIX,
    )

    s2 = io.load_source(
        config.TEST_SOURCE2_PATH,
        source_prefix=config.S2_PREFIX,
    )

    s3 = io.load_source(
        config.TEST_SOURCE3_PATH,
        source_prefix=config.S3_PREFIX,
    )

    return s1, s2, s3


# ==========================================================================
# Stage 2: normalize
# ==========================================================================

def normalize_all(*sources: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Normalize every source table."""

    return tuple(
        normalize.normalize_source(source)
        for source in sources
    )


# ==========================================================================
# Stage 3: candidate generation
# ==========================================================================

def generate_all_candidates(
    s1_norm: pd.DataFrame,
    s2_norm: pd.DataFrame,
    s3_norm: pd.DataFrame,
    *,
    top_k: int = BLOCKING_TOP_K,
) -> pd.DataFrame:
    """Generate candidates against S2 and S3 and union them."""

    from_s2 = blocking.generate_candidates(
        s1_norm,
        s2_norm,
        top_k=top_k,
    )

    from_s3 = blocking.generate_candidates(
        s1_norm,
        s3_norm,
        top_k=top_k,
    )

    parts = [
        part
        for part in (from_s2, from_s3)
        if not part.empty
    ]

    if not parts:
        return pd.DataFrame(
            {
                config.S1_PAIR_COL: pd.Series(dtype=object),
                config.CAND_PAIR_COL: pd.Series(dtype=object),
            }
        )

    combined = pd.concat(
        parts,
        axis=0,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            config.S1_PAIR_COL,
            config.CAND_PAIR_COL,
        ],
        ignore_index=True,
    )

    return combined


# ==========================================================================
# Stage 4: features
# ==========================================================================

def build_features_for(
    pairs_df: pd.DataFrame,
    s1_norm: pd.DataFrame,
    s2_norm: pd.DataFrame,
    s3_norm: pd.DataFrame,
) -> pd.DataFrame:
    """Build numeric features using the already-normalized columns."""

    return features.build_features(
        pairs_df,
        s1_norm,
        s2_norm,
        s3_norm,
        already_normalized=True,
        name_col=config.NAME_NORM_COL,
        addr_col=config.ADDR_NORM_COL,
        country_col=config.COUNTRY_NORM_COL,
    )


# ==========================================================================
# Stage 5: train / validation split
# ==========================================================================

def split_train_val(
    labeled_df: pd.DataFrame,
    *,
    val_frac: float = VAL_FRACTION,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """Split by source1_entity_id to prevent entity leakage."""

    train_ids, val_ids = model.grouped_train_val_split(
        labeled_df,
        group_col=config.S1_PAIR_COL,
        val_frac=val_frac,
        random_state=random_state,
    )

    train_df = labeled_df[
        labeled_df[config.S1_PAIR_COL].isin(train_ids)
    ].reset_index(drop=True)

    val_df = labeled_df[
        labeled_df[config.S1_PAIR_COL].isin(val_ids)
    ].reset_index(drop=True)

    return train_df, val_df, train_ids, val_ids


# ==========================================================================
# Stage 6: train classifier
# ==========================================================================

def fit_classifier(
    labeled_df: pd.DataFrame,
    *,
    random_state: int = config.RANDOM_SEED,
) -> model.PairwiseMatcher:
    """Sample training pairs and train the pairwise classifier."""

    sampled = model.sample_training_pairs(
        labeled_df,
        group_col=config.S1_PAIR_COL,
        random_state=random_state,
    )

    clf = model.train_model(
        sampled[features.FEATURE_COLUMNS],
        sampled["label"],
        feature_columns=features.FEATURE_COLUMNS,
        random_state=random_state,
    )

    return clf


# ==========================================================================
# Full pipeline
# ==========================================================================

def run_pipeline() -> Tuple[
    pd.DataFrame,
    float,
    pd.DataFrame,
]:
    """Run the complete pipeline.

    Returns:
        predictions:
            Final predictions dataframe.

        best_threshold:
            Threshold selected using validation macro F0.5.

        threshold_curve:
            Threshold search results.
    """

    # ----------------------------------------------------------------------
    # Load training data
    # ----------------------------------------------------------------------

    t0 = t = _log("Loading training data...")

    train_s1, train_s2, train_s3, train_gt = load_train_data()

    t = _log(
        f"Loaded train "
        f"S1={len(train_s1):,} "
        f"S2={len(train_s2):,} "
        f"S3={len(train_s3):,} "
        f"ground_truth={len(train_gt):,}",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Load test data
    # ----------------------------------------------------------------------

    t = _log("Loading test data...")

    test_s1, test_s2, test_s3 = load_test_data()

    t = _log(
        f"Loaded test "
        f"S1={len(test_s1):,} "
        f"S2={len(test_s2):,} "
        f"S3={len(test_s3):,}",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Normalize
    # ----------------------------------------------------------------------

    t = _log("Normalizing all source tables...")

    train_s1n, train_s2n, train_s3n = normalize_all(
        train_s1,
        train_s2,
        train_s3,
    )

    test_s1n, test_s2n, test_s3n = normalize_all(
        test_s1,
        test_s2,
        test_s3,
    )

    t = _log(
        "Normalization complete.",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Generate training candidates
    # ----------------------------------------------------------------------

    t = _log(
        "Generating training candidates (blocking)..."
    )

    train_pairs = generate_all_candidates(
        train_s1n,
        train_s2n,
        train_s3n,
    )

    t = _log(
        f"Training candidate pairs: {len(train_pairs):,}",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Generate test candidates
    # ----------------------------------------------------------------------

    t = _log(
        "Generating test candidates (blocking)..."
    )

    test_pairs = generate_all_candidates(
        test_s1n,
        test_s2n,
        test_s3n,
    )

    t = _log(
        f"Test candidate pairs: {len(test_pairs):,}",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Build training features
    # ----------------------------------------------------------------------

    t = _log(
        "Building training features..."
    )

    train_features = build_features_for(
        train_pairs,
        train_s1n,
        train_s2n,
        train_s3n,
    )

    t = _log(
        f"Training feature rows: {len(train_features):,}",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Attach ground-truth labels
    # ----------------------------------------------------------------------

    t = _log(
        "Attaching ground-truth labels..."
    )

    labeled = features.attach_labels(
        train_features,
        train_gt,
    )

    n_pos = int(
        (labeled["label"] == 1).sum()
    )

    t = _log(
        f"Labeled rows: {len(labeled):,} "
        f"(positives: {n_pos:,})",
        start_time=t,
    )

    if n_pos == 0:
        raise RuntimeError(
            "No positive pairs survived blocking + labeling. "
            "Check that blocking.generate_candidates is retrieving "
            "true matches before training."
        )

    # ----------------------------------------------------------------------
    # Train / validation split
    # ----------------------------------------------------------------------

    t = _log(
        "Splitting train/validation by source1_entity_id..."
    )

    train_df, val_df, train_ids, val_ids = split_train_val(
        labeled
    )

    t = _log(
        f"Train groups: {len(train_ids):,} "
        f"rows: {len(train_df):,} | "
        f"Val groups: {len(val_ids):,} "
        f"rows: {len(val_df):,}",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Initial model
    # ----------------------------------------------------------------------

    t = _log(
        "Training initial model (train split only)..."
    )

    initial_clf = fit_classifier(
        train_df
    )

    t = _log(
        f"Initial model trained "
        f"(backend={initial_clf.backend_name}).",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Validation scoring
    # ----------------------------------------------------------------------

    t = _log(
        "Scoring validation split..."
    )

    val_scored = val_df[
        [
            config.S1_PAIR_COL,
            config.CAND_PAIR_COL,
            "label",
        ]
    ].copy()

    val_scored[evaluate.PROBA_COL] = (
        initial_clf.predict_proba(
            val_df[features.FEATURE_COLUMNS]
        )
    )

    t = _log(
        "Validation scored.",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Threshold selection
    # ----------------------------------------------------------------------

    t = _log(
        "Selecting best threshold via validation macro F0.5 "
        "(test data NOT used)..."
    )

    best_threshold, threshold_curve = (
        evaluate.select_best_threshold(
            val_scored,
            train_gt,
            val_ids,
        )
    )

    best_row = threshold_curve.loc[
        np.isclose(
            threshold_curve["threshold"],
            best_threshold,
        )
    ].iloc[0]

    t = _log(
        f"Best threshold={best_threshold:.2f} "
        f"(macro F0.5={best_row['macro_f_beta']:.4f}, "
        f"precision={best_row['macro_precision']:.4f}, "
        f"recall={best_row['macro_recall']:.4f})",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Final model
    # ----------------------------------------------------------------------

    t = _log(
        "Training final model on all labeled training data "
        "(train+val)..."
    )

    final_clf = fit_classifier(
        labeled
    )

    t = _log(
        f"Final model trained "
        f"(backend={final_clf.backend_name}).",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Build test features
    # ----------------------------------------------------------------------

    t = _log(
        "Building test features..."
    )

    test_features = build_features_for(
        test_pairs,
        test_s1n,
        test_s2n,
        test_s3n,
    )

    t = _log(
        f"Test feature rows: {len(test_features):,}",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Score test candidates
    # ----------------------------------------------------------------------

    t = _log(
        "Scoring test candidates with final model..."
    )

    test_scored = test_features[
        [
            config.S1_PAIR_COL,
            config.CAND_PAIR_COL,
        ]
    ].copy()

    if len(test_features):
        test_scored[evaluate.PROBA_COL] = (
            final_clf.predict_proba(
                test_features[features.FEATURE_COLUMNS]
            )
        )
    else:
        test_scored[evaluate.PROBA_COL] = pd.Series(
            dtype=np.float32
        )

    t = _log(
        "Test candidates scored.",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Build final predictions
    # ----------------------------------------------------------------------

    t = _log(
        "Applying validation-selected threshold "
        "and building final predictions..."
    )

    predictions = evaluate.build_predictions(
        test_scored,
        test_s1[config.ID_COL],
        best_threshold,
    )

    n_with_matches = int(
        (
            predictions[config.GT_MATCH_COL] != ""
        ).sum()
    )

    t = _log(
        f"Predictions built: {len(predictions):,} S1 entities "
        f"({n_with_matches:,} with >=1 predicted match).",
        start_time=t,
    )

    # ----------------------------------------------------------------------
    # Save required output: matching_results.tsv
    # ----------------------------------------------------------------------

    evaluate.save_predictions(
        predictions,
        config.PREDICTIONS_OUTPUT_PATH,
    )

    # ----------------------------------------------------------------------
    # Save required output: candidate_pairs.tsv
    #
    # test_pairs is exactly the final candidate set that is passed into
    # feature construction and therefore into the matching model.
    # ----------------------------------------------------------------------

    candidate_pairs_path = (
        config.OUTPUT_DIR / "candidate_pairs.tsv"
    )

    test_pairs.to_csv(
        candidate_pairs_path,
        sep=config.TSV_SEPARATOR,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Final logging
    # ----------------------------------------------------------------------

    _log(
        f"Saved predictions to "
        f"{config.PREDICTIONS_OUTPUT_PATH}",
        start_time=t0,
    )

    _log(
        f"Saved candidate pairs to "
        f"{candidate_pairs_path}",
        start_time=t0,
    )

    return (
        predictions,
        best_threshold,
        threshold_curve,
    )


if __name__ == "__main__":
    run_pipeline()