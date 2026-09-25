"""
model.py  (Person 3's component)
=================================

A pairwise binary classifier: for each (S1, candidate) pair described by
the feature vector from features.py, produce P(pair is a true match).

This module does NOT:
    - pick a probability threshold
    - force exactly one match per S1
    - guarantee at least one match per S1
All of that is Person 4's job (threshold tuning / final selection).

Public API
----------

    grouped_train_val_split(...)   split by source1_entity_id, no leakage
    sample_training_pairs(...)     positives + hard/ordinary negatives
    PairwiseMatcher                train / predict_proba / save / load

Typical usage (by Person 4, illustrative only):

    from src import features, model

    feats = features.build_features(pairs_df, s1_df, s2_df, s3_df)
    labeled = features.attach_labels(feats, ground_truth_df)

    train_ids, val_ids = model.grouped_train_val_split(
        labeled, group_col="source1_entity_id", val_frac=0.15
    )
    train_df = labeled[labeled["source1_entity_id"].isin(train_ids)]
    val_df = labeled[labeled["source1_entity_id"].isin(val_ids)]

    train_sampled = model.sample_training_pairs(train_df, group_col="source1_entity_id")

    clf = model.PairwiseMatcher(feature_columns=features.FEATURE_COLUMNS)
    clf.fit(train_sampled[features.FEATURE_COLUMNS], train_sampled["label"])

    val_scores = clf.predict_proba(val_df[features.FEATURE_COLUMNS])
    # Person 4 then thresholds val_scores against val_df["label"], grouped
    # by source1_entity_id, to optimize macro F0.5.
"""

from __future__ import annotations

import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ==========================================================================
# Grouped train/validation split (leakage prevention)
# ==========================================================================

def grouped_train_val_split(
    df: pd.DataFrame,
    *,
    group_col: str = "source1_entity_id",
    val_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split unique S1 group ids into train/val sets so that every candidate
    pair for a given S1 entity stays entirely in one split.

    Returns (train_group_ids, val_group_ids). Caller filters their frame
    with df[group_col].isin(train_group_ids) / .isin(val_group_ids).

    Uses sklearn's GroupShuffleSplit when available (handles large group
    counts efficiently); otherwise falls back to a plain shuffled split of
    the unique group ids, which is equivalent here since we only need a
    group-level train/val partition, not a stratified one.
    """
    groups = df[group_col].unique()
    rng = np.random.RandomState(random_state)
    try:
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
        # GroupShuffleSplit wants X and groups aligned 1:1 with rows, but we
        # only need the *unique* group partition, so we run it over the
        # unique-group array itself (X is a dummy of the same length).
        dummy_X = np.zeros((len(groups), 1))
        train_idx, val_idx = next(gss.split(dummy_X, groups=groups))
        return groups[train_idx], groups[val_idx]
    except ImportError:  # pragma: no cover
        rng.shuffle(groups)
        n_val = int(len(groups) * val_frac)
        return groups[n_val:], groups[:n_val]


# ==========================================================================
# Negative sampling
# ==========================================================================

def sample_training_pairs(
    labeled_df: pd.DataFrame,
    *,
    group_col: str = "source1_entity_id",
    label_col: str = "label",
    hard_negative_proxy_col: str = "name_char_sim",
    neg_pos_ratio: float = 4.0,
    hard_negative_frac: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a memory-conscious, informative training set from labeled
    candidate pairs.

    All positive pairs are kept. Negatives are subsampled *per S1 group*
    (never a Cartesian construction -- negatives already come only from
    blocking's candidate set) up to `neg_pos_ratio` times the positive
    count, split between:

      - hard negatives: negatives with the highest `hard_negative_proxy_col`
        value (default: name similarity) -- these are the "same/similar
        name, wrong entity" cases the model most needs to learn to reject.
      - ordinary negatives: the remaining negatives, sampled uniformly at
        random, so the model still sees easy/typical negatives too.

    If a group has few negatives, all of them are kept (no attempt to
    manufacture negatives that blocking did not produce).
    """
    rng = np.random.RandomState(random_state)

    pos = labeled_df[labeled_df[label_col] == 1]
    neg = labeled_df[labeled_df[label_col] == 0]

    n_pos = len(pos)
    if n_pos == 0:
        # No positives at all (e.g. a debug/smoke-test slice): fall back to
        # returning everything rather than an empty/undefined sample.
        return labeled_df

    target_neg_total = int(n_pos * neg_pos_ratio)
    if len(neg) <= target_neg_total:
        return pd.concat([pos, neg], axis=0, ignore_index=True)

    n_hard = int(target_neg_total * hard_negative_frac)
    n_ordinary = target_neg_total - n_hard

    # Hard negatives: top-scoring proxy value per group, capped globally.
    neg_sorted = neg.sort_values(hard_negative_proxy_col, ascending=False)
    hard_negs = neg_sorted.head(n_hard)

    remaining = neg_sorted.iloc[n_hard:]
    if len(remaining) > n_ordinary:
        sampled_idx = rng.choice(remaining.index.values, size=n_ordinary, replace=False)
        ordinary_negs = remaining.loc[sampled_idx]
    else:
        ordinary_negs = remaining

    return pd.concat([pos, hard_negs, ordinary_negs], axis=0, ignore_index=True)


# ==========================================================================
# Model backend selection
# ==========================================================================
#
# Preference order: LightGBM > XGBoost > sklearn HistGradientBoostingClassifier.
# The first two are efficient gradient boosting implementations well suited
# to tabular numeric features at multi-million-row scale and handle NaN
# natively (relevant since missingness is meaningful here, not an error --
# see features.py's *_missing_* columns, which the trees can split on
# directly). HistGradientBoostingClassifier is the guaranteed-available
# fallback (ships with scikit-learn) and also natively supports NaN.

def _make_default_backend(random_state: int = 42):
    try:
        import lightgbm as lgb

        return "lightgbm", lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass
    try:
        import xgboost as xgb

        return "xgboost", xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier

    return "sklearn_hgb", HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=63,
        random_state=random_state,
    )


# ==========================================================================
# PairwiseMatcher
# ==========================================================================

class PairwiseMatcher:
    """Thin wrapper around a gradient-boosting classifier that pins down
    the feature column order/name so training and inference can never
    silently misalign, and that exposes plain probabilities (no baked-in
    thresholding) as required by the precision-heavy, variable-match-count
    evaluation.
    """

    def __init__(self, feature_columns: list, backend=None, backend_name: Optional[str] = None,
                 random_state: int = 42):
        self.feature_columns = list(feature_columns)
        self.random_state = random_state
        if backend is None:
            backend_name, backend = _make_default_backend(random_state)
        self.backend_name = backend_name
        self.model = backend
        self._fitted = False

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        X = self._select_columns(X)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.model.fit(X.values, np.asarray(y), **fit_kwargs)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(match) for each row, as a 1-D float array."""
        if not self._fitted:
            raise RuntimeError("PairwiseMatcher.fit(...) must be called before predict_proba(...)")
        X = self._select_columns(X)
        proba = self.model.predict_proba(X.values)
        # class order is [neg, pos] for every backend used here
        return proba[:, 1]

    def _select_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        # Fixed column order every call -- this is what prevents silent
        # train/inference feature misalignment.
        return X[self.feature_columns]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "feature_columns": self.feature_columns,
                    "backend_name": self.backend_name,
                    "model": self.model,
                    "random_state": self.random_state,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "PairwiseMatcher":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(
            feature_columns=payload["feature_columns"],
            backend=payload["model"],
            backend_name=payload["backend_name"],
            random_state=payload.get("random_state", 42),
        )
        obj._fitted = True
        return obj


# ==========================================================================
# Convenience functional wrappers (in case Person 4 prefers functions over
# the class -- both call the same underlying logic)
# ==========================================================================

def train_model(X: pd.DataFrame, y, feature_columns: list, random_state: int = 42,
                 sample_weight=None) -> PairwiseMatcher:
    clf = PairwiseMatcher(feature_columns=feature_columns, random_state=random_state)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def predict_proba(clf: PairwiseMatcher, X: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(X)


def save_model(clf: PairwiseMatcher, path: str) -> None:
    clf.save(path)


def load_model(path: str) -> PairwiseMatcher:
    return PairwiseMatcher.load(path)


# ==========================================================================
# INTERFACE ASSUMPTIONS
# ==========================================================================
#
# 1. `labeled_df` passed into grouped_train_val_split / sample_training_pairs
#    is the output of features.attach_labels(...): the feature columns from
#    features.FEATURE_COLUMNS, plus source1_entity_id / candidate_entity_id
#    / label. Column names are configurable via kwargs if blocking/labels
#    use different names.
#
# 2. This module trains one global classifier over all candidate pairs
#    (mixing S2 and S3 candidates). If Person 4's evaluation shows S2 and S3
#    candidates need materially different decision boundaries, the same
#    PairwiseMatcher class can be instantiated twice (one per source) with
#    no change to features.py.
#
# 3. Model persistence uses pickle for simplicity/portability across the
#    three supported backends (lightgbm/xgboost/sklearn expose different
#    "native" save formats). If the wider repo standardizes on joblib or a
#    specific model registry format, swap PairwiseMatcher.save/load's
#    internals only -- the public signatures do not need to change.
