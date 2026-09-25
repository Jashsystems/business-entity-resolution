"""
features.py  (Person 3's component)
====================================

Turns candidate (S1, S2/S3) pairs produced by blocking (Person 2) into a
numerical feature matrix for the pairwise classifier in model.py.

--------------------------------------------------------------------------
ASSUMED INTERFACE (see "INTERFACE ASSUMPTIONS" below for how to adapt this
to the real repository once config.py / io.py / normalize.py / blocking.py
exist)
--------------------------------------------------------------------------

Source tables (S1, S2, S3) are pandas DataFrames with the raw columns
documented in the project spec:

    entity_id, business_name, business_address, country

Candidate pairs (from Person 2's blocking stage) are a long-format
DataFrame with one row per candidate pair:

    source1_entity_id, candidate_entity_id

`candidate_entity_id` may reference either an S2 or an S3 record; the
"S2-" / "S3-" prefix in the id is used to route the lookup to the right
source table. This is exactly why the S1/S2/S3 id convention exists.

Public API
----------

    FEATURE_COLUMNS            stable, ordered list of feature names
    build_features(...)        candidate pairs -> feature DataFrame
    build_training_labels(...) ground truth -> long-format positive pairs
    attach_labels(...)         feature DataFrame + ground truth -> +label col

Everything here is read-only with respect to the source/candidate/ground
truth tables it is given; it does not perform blocking, does not search
S2/S3, and does not create candidate pairs.
"""

from __future__ import annotations

import re
import unicodedata
import difflib
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# rapidfuzz is much faster than difflib for token_sort/token_set style
# similarity on millions of pairs. It is optional: if the repository's
# environment has it installed we use it, otherwise we fall back to a
# pure-stdlib implementation so this module never hard-fails.
try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAVE_RAPIDFUZZ = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_RAPIDFUZZ = False


# ==========================================================================
# Column-name configuration (override via build_features(...) kwargs if the
# real pipeline uses different names -- see INTERFACE ASSUMPTIONS at bottom)
# ==========================================================================

ID_COL = "entity_id"
NAME_COL = "business_name"
ADDR_COL = "business_address"
COUNTRY_COL = "country"

S1_PAIR_COL = "source1_entity_id"
CAND_PAIR_COL = "candidate_entity_id"

GT_S1_COL = "source1_entity_id"
GT_MATCH_COL = "matched_entity_ids"


# ==========================================================================
# Stable, ordered feature list
# ==========================================================================

FEATURE_COLUMNS = [
    # --- name features ---
    "name_missing_s1",
    "name_missing_cand",
    "name_exact_match",
    "name_char_sim",
    "name_token_sort_sim",
    "name_token_set_sim",
    "name_token_jaccard",
    "name_token_containment",
    "name_len_diff",
    # --- address features ---
    "address_missing_s1",
    "address_missing_cand",
    "address_exact_match",
    "address_char_sim",
    "address_token_jaccard",
    "address_len_diff",
    "address_numeric_jaccard",
    "postal_both_present",
    "postal_match",
    "house_number_both_present",
    "house_number_match",
    # --- country features ---
    "country_missing_s1",
    "country_missing_cand",
    "country_match",
    # --- combined / interaction features ---
    "name_address_sim_product",
    "name_high_address_low_conflict",
    "name_low_address_high_support",
]


# ==========================================================================
# Text normalization fallback
# ==========================================================================
#
# Person 1 owns the project's normalization pipeline (normalize.py). If the
# source tables passed in already contain normalized text (e.g. because the
# pipeline calls normalize.py before handing rows to us), we use it as-is --
# we do not want two competing normalization implementations disagreeing
# with each other.
#
# The functions below are only a *fallback*, used when the incoming text
# still looks like raw/untouched text. They exist so this module is
# self-contained and testable in isolation, and so it never crashes on
# messy Unicode (accents, encoding artifacts, non-Latin scripts).

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z\s]")
_DIGIT_RUN_RE = re.compile(r"\d+")


def _safe_str(x) -> str:
    """Coerce any scalar (including NaN/None) to a plain string, never raising."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x)


def normalize_text(x) -> str:
    """Lightweight, Unicode-safe fallback normalization.

    - NFKC-normalizes Unicode (fixes many encoding-artifact / compatibility
      issues without destroying non-Latin scripts).
    - Lowercases.
    - Strips punctuation to whitespace (keeps alphanumerics from *any*
      script, since \\w in Python's re with str is Unicode-aware).
    - Collapses whitespace.

    Never raises; unparseable/garbled input degrades to an empty or
    partial string rather than crashing feature generation.
    """
    s = _safe_str(x)
    if not s:
        return ""
    try:
        s = unicodedata.normalize("NFKC", s)
    except Exception:
        pass
    s = s.lower()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _tokens(s: str) -> list:
    if not s:
        return []
    return s.split(" ")


# ==========================================================================
# Similarity primitives
# ==========================================================================

def _char_sim(a: str, b: str) -> float:
    """Character-level similarity in [0, 1]. 0 if either side is empty."""
    if not a or not b:
        return 0.0
    if _HAVE_RAPIDFUZZ:
        return _rf_fuzz.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _token_sort_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAVE_RAPIDFUZZ:
        return _rf_fuzz.token_sort_ratio(a, b) / 100.0
    ta, tb = " ".join(sorted(_tokens(a))), " ".join(sorted(_tokens(b)))
    return difflib.SequenceMatcher(None, ta, tb).ratio()


def _token_set_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAVE_RAPIDFUZZ:
        return _rf_fuzz.token_set_ratio(a, b) / 100.0
    sa, sb = set(_tokens(a)), set(_tokens(b))
    inter = sa & sb
    common = " ".join(sorted(inter))
    rest_a = " ".join(sorted(sa - sb))
    rest_b = " ".join(sorted(sb - sa))
    ta = (common + " " + rest_a).strip()
    tb = (common + " " + rest_b).strip()
    return difflib.SequenceMatcher(None, ta, tb).ratio()


def _jaccard(sa: set, sb: set) -> float:
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _containment(sa: set, sb: set) -> float:
    """Fraction of the smaller token set contained in the larger one."""
    if not sa or not sb:
        return 0.0
    smaller, larger = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(smaller & larger) / len(smaller)


def _extract_digit_runs(s: str) -> set:
    if not s:
        return set()
    return set(_DIGIT_RUN_RE.findall(s))


def _extract_postal_candidate(s: str) -> Optional[str]:
    """Heuristic, country-agnostic postal-code guess: the longest digit run
    of length 4-6 (covers US ZIP, Indian PIN, French postal codes, etc.
    without hard-coding any country-specific format)."""
    runs = [r for r in _DIGIT_RUN_RE.findall(s or "") if 4 <= len(r) <= 6]
    if not runs:
        return None
    return max(runs, key=len)


def _extract_house_number_candidate(s: str) -> Optional[str]:
    """Heuristic: the first digit run in the address, typically the
    house/street number in most Western and Indian address formats."""
    m = _DIGIT_RUN_RE.search(s or "")
    return m.group(0) if m else None


# ==========================================================================
# Row-level feature computation (vectorized outer loop via pandas merge,
# per-row similarity kept in tight Python loops over already-aligned
# numpy/list arrays rather than DataFrame.apply, which is measurably
# faster at millions-of-rows scale).
# ==========================================================================

def _compute_feature_block(
    name_s1: Sequence[str], name_c: Sequence[str],
    addr_s1: Sequence[str], addr_c: Sequence[str],
    country_s1: Sequence[str], country_c: Sequence[str],
) -> pd.DataFrame:
    n = len(name_s1)
    out = {col: np.zeros(n, dtype=np.float32) for col in FEATURE_COLUMNS}

    for i in range(n):
        ns1, nc = name_s1[i], name_c[i]
        as1, ac = addr_s1[i], addr_c[i]
        cs1, cc = country_s1[i], country_c[i]

        ns1_missing, nc_missing = (len(ns1) == 0), (len(nc) == 0)
        as1_missing, ac_missing = (len(as1) == 0), (len(ac) == 0)
        cs1_missing, cc_missing = (len(cs1) == 0), (len(cc) == 0)

        # ---- name ----
        out["name_missing_s1"][i] = ns1_missing
        out["name_missing_cand"][i] = nc_missing
        if not ns1_missing and not nc_missing:
            name_exact = float(ns1 == nc)
            name_char = _char_sim(ns1, nc)
            name_tsort = _token_sort_sim(ns1, nc)
            name_tset = _token_set_sim(ns1, nc)
            toks1, toks2 = set(_tokens(ns1)), set(_tokens(nc))
            name_jac = _jaccard(toks1, toks2)
            name_cont = _containment(toks1, toks2)
            name_len_diff = abs(len(ns1) - len(nc)) / max(len(ns1), len(nc), 1)
        else:
            name_exact = 0.0
            name_char = name_tsort = name_tset = name_jac = name_cont = 0.0
            name_len_diff = 1.0
        out["name_exact_match"][i] = name_exact
        out["name_char_sim"][i] = name_char
        out["name_token_sort_sim"][i] = name_tsort
        out["name_token_set_sim"][i] = name_tset
        out["name_token_jaccard"][i] = name_jac
        out["name_token_containment"][i] = name_cont
        out["name_len_diff"][i] = name_len_diff

        # ---- address ----
        out["address_missing_s1"][i] = as1_missing
        out["address_missing_cand"][i] = ac_missing
        if not as1_missing and not ac_missing:
            addr_exact = float(as1 == ac)
            addr_char = _char_sim(as1, ac)
            atoks1, atoks2 = set(_tokens(as1)), set(_tokens(ac))
            addr_jac = _jaccard(atoks1, atoks2)
            addr_len_diff = abs(len(as1) - len(ac)) / max(len(as1), len(ac), 1)

            nums1, nums2 = _extract_digit_runs(as1), _extract_digit_runs(ac)
            addr_num_jac = _jaccard(nums1, nums2)

            postal1, postal2 = _extract_postal_candidate(as1), _extract_postal_candidate(ac)
            if postal1 and postal2:
                postal_both = 1.0
                postal_match = float(postal1 == postal2)
            else:
                postal_both = 0.0
                postal_match = 0.0

            house1, house2 = _extract_house_number_candidate(as1), _extract_house_number_candidate(ac)
            if house1 and house2:
                house_both = 1.0
                house_match = float(house1 == house2)
            else:
                house_both = 0.0
                house_match = 0.0
        else:
            addr_exact = 0.0
            addr_char = addr_jac = 0.0
            addr_len_diff = 1.0
            addr_num_jac = 0.0
            postal_both = postal_match = 0.0
            house_both = house_match = 0.0
        out["address_exact_match"][i] = addr_exact
        out["address_char_sim"][i] = addr_char
        out["address_token_jaccard"][i] = addr_jac
        out["address_len_diff"][i] = addr_len_diff
        out["address_numeric_jaccard"][i] = addr_num_jac
        out["postal_both_present"][i] = postal_both
        out["postal_match"][i] = postal_match
        out["house_number_both_present"][i] = house_both
        out["house_number_match"][i] = house_match

        # ---- country ----
        out["country_missing_s1"][i] = cs1_missing
        out["country_missing_cand"][i] = cc_missing
        if not cs1_missing and not cc_missing:
            out["country_match"][i] = float(cs1 == cc)
        else:
            out["country_match"][i] = 0.0

        # ---- combined ----
        out["name_address_sim_product"][i] = name_char * addr_char
        # High name similarity but addresses both present and clearly
        # conflicting: classic hard-negative pattern (common chain name,
        # different location).
        out["name_high_address_low_conflict"][i] = float(
            name_char >= 0.85 and not as1_missing and not ac_missing and addr_char <= 0.4
        )
        # Weaker name but strong corroborating address evidence: still
        # useful positive signal (e.g. abbreviations/typos in the name).
        out["name_low_address_high_support"][i] = float(
            name_char <= 0.6 and addr_char >= 0.8
        )

    return pd.DataFrame(out)


# ==========================================================================
# Public: build_features
# ==========================================================================

def build_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    *,
    id_col: str = ID_COL,
    name_col: str = NAME_COL,
    addr_col: str = ADDR_COL,
    country_col: str = COUNTRY_COL,
    s1_pair_col: str = S1_PAIR_COL,
    cand_pair_col: str = CAND_PAIR_COL,
    chunksize: int = 200_000,
    already_normalized: bool = False,
) -> pd.DataFrame:
    """Turn candidate pairs into a numerical feature matrix.

    Parameters
    ----------
    pairs_df : DataFrame with columns [s1_pair_col, cand_pair_col], one row
        per candidate pair produced by blocking. `cand_pair_col` values are
        S2 or S3 entity ids (routed by "S2-"/"S3-" prefix).
    s1_df, s2_df, s3_df : raw source tables with columns
        [id_col, name_col, addr_col, country_col].
    chunksize : candidate pairs are processed in chunks of this many rows
        to bound peak memory; increase/decrease per available RAM.
    already_normalized : set True once Person 1's normalize.py is wired in
        and the incoming name/address/country columns are already
        normalized -- this skips the internal fallback normalization pass
        (keeps a single source of truth for normalization logic).

    Returns
    -------
    DataFrame with columns [s1_pair_col, cand_pair_col] + FEATURE_COLUMNS,
    same row order as `pairs_df`, dtype float32 for feature columns (keeps
    memory down at multi-million-row scale).
    """
    # Build lookup tables indexed by entity_id. Concatenating S2+S3 into one
    # candidate lookup lets a single merge resolve candidate attributes
    # regardless of source, without ever computing S1 x S2/S3 products.
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        cols = [id_col, name_col, addr_col, country_col]
        d = df[cols].drop_duplicates(subset=id_col).set_index(id_col)
        return d

    s1_lookup = _prep(s1_df)
    cand_lookup = pd.concat([_prep(s2_df), _prep(s3_df)], axis=0)
    # Duplicate ids across S2/S3 lookups would silently corrupt joins --
    # guard rather than fail deep inside pandas internals.
    if cand_lookup.index.duplicated().any():
        cand_lookup = cand_lookup[~cand_lookup.index.duplicated(keep="first")]

    out_chunks = []
    n = len(pairs_df)
    for start in range(0, n, chunksize):
        chunk = pairs_df.iloc[start:start + chunksize]

        left = chunk[[s1_pair_col, cand_pair_col]].merge(
            s1_lookup, left_on=s1_pair_col, right_index=True, how="left"
        ).rename(columns={name_col: "_n1", addr_col: "_a1", country_col: "_c1"})

        both = left.merge(
            cand_lookup, left_on=cand_pair_col, right_index=True, how="left"
        ).rename(columns={name_col: "_n2", addr_col: "_a2", country_col: "_c2"})

        if already_normalized:
            n1 = both["_n1"].map(_safe_str).tolist()
            n2 = both["_n2"].map(_safe_str).tolist()
            a1 = both["_a1"].map(_safe_str).tolist()
            a2 = both["_a2"].map(_safe_str).tolist()
            c1 = both["_c1"].map(_safe_str).tolist()
            c2 = both["_c2"].map(_safe_str).tolist()
        else:
            n1 = both["_n1"].map(normalize_text).tolist()
            n2 = both["_n2"].map(normalize_text).tolist()
            a1 = both["_a1"].map(normalize_text).tolist()
            a2 = both["_a2"].map(normalize_text).tolist()
            c1 = both["_c1"].map(normalize_text).tolist()
            c2 = both["_c2"].map(normalize_text).tolist()

        feat_block = _compute_feature_block(n1, n2, a1, a2, c1, c2)
        feat_block.index = chunk.index
        result_chunk = pd.concat(
            [chunk[[s1_pair_col, cand_pair_col]].reset_index(drop=True),
             feat_block.reset_index(drop=True)],
            axis=1,
        )
        out_chunks.append(result_chunk)

    if not out_chunks:
        return pd.DataFrame(columns=[s1_pair_col, cand_pair_col] + FEATURE_COLUMNS)

    return pd.concat(out_chunks, axis=0, ignore_index=True)


# ==========================================================================
# Public: label construction helpers
# ==========================================================================

def build_training_labels(
    ground_truth_df: pd.DataFrame,
    *,
    gt_s1_col: str = GT_S1_COL,
    gt_match_col: str = GT_MATCH_COL,
) -> pd.DataFrame:
    """Explode ground truth into a long-format table of positive pairs.

    Input: one row per S1 entity with a comma-separated matched_entity_ids
    string (empty/NaN => no matches).

    Output: DataFrame with columns [source1_entity_id, candidate_entity_id],
    one row per true positive pair. This is the *only* use of ground truth
    in this module: it defines labels, never a feature.
    """
    gt = ground_truth_df[[gt_s1_col, gt_match_col]].copy()
    gt[gt_match_col] = gt[gt_match_col].map(_safe_str)
    gt["_matches"] = gt[gt_match_col].str.split(",")
    gt = gt.explode("_matches")
    gt["_matches"] = gt["_matches"].str.strip()
    gt = gt[gt["_matches"] != ""]
    return gt.rename(columns={gt_s1_col: S1_PAIR_COL, "_matches": CAND_PAIR_COL})[
        [S1_PAIR_COL, CAND_PAIR_COL]
    ].reset_index(drop=True)


def attach_labels(
    features_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    *,
    s1_pair_col: str = S1_PAIR_COL,
    cand_pair_col: str = CAND_PAIR_COL,
    gt_s1_col: str = GT_S1_COL,
    gt_match_col: str = GT_MATCH_COL,
) -> pd.DataFrame:
    """Add a binary `label` column to a feature DataFrame by checking
    membership of each candidate pair in the exploded ground truth.

    A pair not present in ground truth for that S1 is labeled 0 (negative).
    This assumes `features_df` was built only from candidate pairs that
    blocking actually produced (never a synthetic all-pairs set), which is
    the documented contract of this module.
    """
    positives = build_training_labels(
        ground_truth_df, gt_s1_col=gt_s1_col, gt_match_col=gt_match_col
    )
    positives["label"] = 1
    merged = features_df.merge(
        positives,
        left_on=[s1_pair_col, cand_pair_col],
        right_on=[S1_PAIR_COL, CAND_PAIR_COL],
        how="left",
        suffixes=("", "_gt"),
    )
    merged["label"] = merged["label"].fillna(0).astype(np.int8)
    drop_cols = [c for c in (S1_PAIR_COL, CAND_PAIR_COL) if c not in (s1_pair_col, cand_pair_col) and c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def get_feature_names() -> list:
    """Return the stable, ordered list of feature column names."""
    return list(FEATURE_COLUMNS)


# ==========================================================================
# INTERFACE ASSUMPTIONS (read before wiring this into main.py)
# ==========================================================================
#
# 1. Raw source tables use columns: entity_id, business_name,
#    business_address, country. This matches the project spec's documented
#    TSV schema exactly (section 2), so it should not need adapting.
#
# 2. Candidate pairs from blocking are a LONG-format table with columns
#    source1_entity_id, candidate_entity_id (one row per pair). If
#    blocking.py instead already emits a merged/wide table (S1 attributes
#    + candidate attributes in the same row), skip the merge step inside
#    build_features and feed columns directly into
#    _compute_feature_block(...) -- the row-level feature logic is
#    independent of how the join happens.
#
# 3. If normalize.py's normalized text is already present on the source
#    tables under the SAME column names (business_name/business_address/
#    country), call build_features(..., already_normalized=True) to skip
#    the fallback normalization pass here and avoid double-normalizing.
#    If normalize.py exposes it under different column names, pass those
#    via name_col/addr_col/country_col.
#
# 4. ground_truth_df is assumed to have source1_entity_id and
#    matched_entity_ids columns per the spec (section 2); adjust
#    gt_s1_col/gt_match_col if the loaded DataFrame names them differently.
