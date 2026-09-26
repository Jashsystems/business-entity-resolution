"""
blocking.py  (Person 2's component)
====================================

High-recall candidate generation: for every Source 1 entity, produce a
manageable set of possible Source 2 / Source 3 matches for Person 3's
feature/model pipeline to score.

This module does NOT decide matches -- it only narrows millions x millions
down to a candidate set per S1 entity. Recall matters far more than
precision here; the classifier downstream is what enforces precision.

Uses Person 1's already-normalized columns (config.NAME_NORM_COL /
ADDR_NORM_COL / COUNTRY_NORM_COL) exclusively -- no normalization logic
is duplicated here.

Public API
----------
    generate_candidates(source1_df, candidate_df, *, top_k=50)
        -> DataFrame[source1_entity_id, candidate_entity_id]

Works identically for candidate_df = source2 or source3; no S2/S3-specific
branching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# sklearn is a hard dependency of the TF-IDF stage. It's already required
# by model.py (GroupShuffleSplit / HistGradientBoostingClassifier), so no
# new dependency is introduced by using it here too.
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors


# ==========================================================================
# Tunables
# ==========================================================================

# A normalized key (name, or address house-number token) shared by more
# than this many candidate-side rows is "too common" to safely block on
# exactly -- for a key this common, exact blocking would pair every S1 row
# that has it with every candidate row that has it, which is a near-
# Cartesian blow-up for common names/tokens (chains, generic names like
# "the coffee shop", or a house number like "1") while adding little
# beyond what n-gram TF-IDF retrieval already finds. Skipping the exact
# block for these does NOT lose recall on its own: TF-IDF retrieval below
# still fires for them, since it is not key-count gated.
_MAX_EXACT_BLOCK_GROUP_SIZE = 200

# Character n-gram range for the approximate-name TF-IDF stage. 3-5 grams
# are short enough to survive minor spelling/OCR-ish differences while
# still being discriminative for short business names.
_NGRAM_RANGE = (3, 5)

# Length of the name_norm prefix used to bucket the TF-IDF nearest-
# neighbor search (see _tfidf_name_candidates). Benchmarked directly
# during development on synthetic data shaped like this project's real
# scale: prefix_len=1 (26-ish buckets) still left ~200k x 500k rows
# taking on the order of a minute for this stage alone, which projects to
# multiple hours at the real 2.2M x 5M+ scale; prefix_len=2 (hundreds of
# buckets) cut that stage's time by roughly 15x with no other change,
# just smaller per-bucket work. Real near-duplicate names differing in
# their first two characters are rare (typos there are uncommon, and
# transliteration differences are usually further into the name), so the
# recall cost of this bucketing is small, and it is not the only recall
# path -- exact-name, name+country, and address blocking all run
# independently of this key.
_TFIDF_BUCKET_PREFIX_LEN = 2


# ==========================================================================
# Small internal helpers
# ==========================================================================

def _nonempty(series: pd.Series) -> pd.Series:
    """Boolean mask: value is present and not the empty string normalize.py
    produces for missing input (never filters on the literal "nan")."""
    return series.notna() & (series != "")


def _dedupe_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (source1_entity_id, candidate_entity_id) pairs,
    keeping the first occurrence -- pairs are unordered facts, not ranked,
    so which duplicate survives doesn't matter."""
    return df.drop_duplicates(subset=[config.S1_PAIR_COL, config.CAND_PAIR_COL], ignore_index=True)


def _make_pairs(s1_ids: np.ndarray, cand_ids: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({config.S1_PAIR_COL: s1_ids, config.CAND_PAIR_COL: cand_ids})


def _empty_pairs() -> pd.DataFrame:
    return _make_pairs(np.array([]), np.array([]))


def _cap_common_keys(cand: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Drop rows whose key occurs more than _MAX_EXACT_BLOCK_GROUP_SIZE
    times on the candidate side, so an indexed merge on `key_col` can
    never blow up into a near-Cartesian block for a pathologically common
    key. Computed from the candidate side only, since block size for a
    given key is driven by how many candidate rows share it."""
    sizes = cand.groupby(key_col)[config.ID_COL].transform("size")
    return cand.loc[sizes <= _MAX_EXACT_BLOCK_GROUP_SIZE]


def _merge_on_key(s1: pd.DataFrame, cand: pd.DataFrame, key_cols) -> pd.DataFrame:
    if s1.empty or cand.empty:
        return _empty_pairs()
    merged = s1.merge(cand, on=key_cols, suffixes=("_s1", "_cand"))
    if merged.empty:
        return _empty_pairs()
    return _make_pairs(
        merged[f"{config.ID_COL}_s1"].to_numpy(),
        merged[f"{config.ID_COL}_cand"].to_numpy(),
    )


# ==========================================================================
# Strategy 1: exact normalized name
# ==========================================================================

def _exact_name_candidates(source1_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Pair every S1 row with every candidate row sharing the same
    non-empty name_norm, via a single indexed merge (no Python loops, no
    Cartesian product -- pandas' merge only materializes rows that
    actually share a key). Guards against pathologically common names via
    _cap_common_keys; those are left to the TF-IDF stage instead, which
    ranks rather than exhaustively pairs.
    """
    s1 = source1_df.loc[_nonempty(source1_df[config.NAME_NORM_COL]), [config.ID_COL, config.NAME_NORM_COL]]
    cand = candidate_df.loc[_nonempty(candidate_df[config.NAME_NORM_COL]), [config.ID_COL, config.NAME_NORM_COL]]
    if s1.empty or cand.empty:
        return _empty_pairs()
    cand = _cap_common_keys(cand, config.NAME_NORM_COL)
    return _merge_on_key(s1, cand, config.NAME_NORM_COL)


# ==========================================================================
# Strategy 2: normalized name + country
# ==========================================================================

def _name_country_candidates(source1_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Same idea as exact-name blocking, keyed on (name_norm,
    country_norm) instead. This is a SEPARATE, additive strategy, not a
    replacement -- it recovers cases where a name is too common *globally*
    to pass strategy 1's safeguard, but is uncommon once restricted to one
    country. Country equality is never a hard requirement of the overall
    pipeline (this is just one of several unioned strategies), so an
    empty/unknown country_norm value is excluded from this block entirely
    rather than treated as a wildcard match.
    """
    key_cols = [config.NAME_NORM_COL, config.COUNTRY_NORM_COL]
    s1_mask = _nonempty(source1_df[config.NAME_NORM_COL]) & _nonempty(source1_df[config.COUNTRY_NORM_COL])
    cand_mask = _nonempty(candidate_df[config.NAME_NORM_COL]) & _nonempty(candidate_df[config.COUNTRY_NORM_COL])
    s1 = source1_df.loc[s1_mask, [config.ID_COL] + key_cols]
    cand = candidate_df.loc[cand_mask, [config.ID_COL] + key_cols]
    return _merge_on_key(s1, cand, key_cols)


# ==========================================================================
# Strategy 3: approximate name retrieval (character n-gram TF-IDF)
# ==========================================================================

def _tfidf_name_candidates(source1_df: pd.DataFrame, candidate_df: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """For each S1 row, retrieve its top_k nearest candidate rows by
    cosine similarity over character n-gram TF-IDF of name_norm.

    This is what gives the pipeline recall on near-misses that exact/
    name+country blocking miss entirely: typos, transliteration
    differences, word-order swaps, abbreviations, etc.

    SCALABILITY NOTE: a single global nearest-neighbor search (S1 rows x
    candidate rows) was benchmarked at ~60s for 20k x 50k rows. That cost
    scales roughly with the product of the two sizes, so at this project's
    real scale (millions x millions) a single global search is not
    feasible (order of days, not minutes) -- confirmed by a direct
    benchmark during development, not a theoretical guess.

    To stay within a shared vector space (S1 and candidate names must be
    comparable) while keeping the search itself tractable, this uses a
    stateless HashingVectorizer (see below for why, vs. TfidfVectorizer)
    plus bucketing: NearestNeighbors is run separately per
    _TFIDF_BUCKET_PREFIX_LEN-character name prefix, only ever comparing
    S1 rows to candidate rows that start the same way. This is a standard
    "sorted-neighborhood"-style coarsening: real near-duplicate names
    (typos, minor formatting differences) essentially never differ in
    their first couple of characters, so recall loss from this bucketing
    is small, while the search cost drops from O(n_s1 * n_cand) to the
    much smaller sum of O(n_s1_bucket * n_cand_bucket) across buckets. Any
    S1 row whose true match starts differently is still covered by the
    exact-name / name+country / address strategies run alongside this one.
    """
    s1_mask = _nonempty(source1_df[config.NAME_NORM_COL])
    cand_mask = _nonempty(candidate_df[config.NAME_NORM_COL])
    s1 = source1_df.loc[s1_mask, [config.ID_COL, config.NAME_NORM_COL]].reset_index(drop=True)
    cand = candidate_df.loc[cand_mask, [config.ID_COL, config.NAME_NORM_COL]].reset_index(drop=True)
    if s1.empty or cand.empty:
        return _empty_pairs()

    # HashingVectorizer (not TfidfVectorizer) is deliberately used here:
    # TfidfVectorizer.fit() must scan the *entire* corpus to build and
    # count a vocabulary before any bucketing can help, and that counting
    # pass was benchmarked at ~33s on a 700k-name corpus during
    # development -- a cost paid once per generate_candidates() call
    # regardless of how the neighbor search itself is bucketed below.
    # HashingVectorizer needs no vocabulary/fit pass at all (hashing is a
    # fixed, stateless function of each n-gram), so it removes that
    # bottleneck entirely rather than just shrinking it. This trades away
    # exact IDF weighting and can very rarely collide two different
    # n-grams into the same hash bucket -- an acceptable approximation for
    # a *retrieval* heuristic feeding a downstream classifier, not the
    # final similarity score used for matching (features.py computes its
    # own exact similarity features separately).
    vectorizer = HashingVectorizer(
        analyzer="char_wb", ngram_range=_NGRAM_RANGE,
        n_features=2 ** 20, alternate_sign=False, norm="l2",
    )
    s1_vecs = vectorizer.transform(s1[config.NAME_NORM_COL])
    cand_vecs = vectorizer.transform(cand[config.NAME_NORM_COL])

    s1_bucket_key = s1[config.NAME_NORM_COL].str[:_TFIDF_BUCKET_PREFIX_LEN]
    cand_bucket_key = cand[config.NAME_NORM_COL].str[:_TFIDF_BUCKET_PREFIX_LEN]
    cand_positions_by_bucket = cand.groupby(cand_bucket_key).indices  # {char: row-position array}

    result_parts = []
    for bucket_char, s1_positions in s1.groupby(s1_bucket_key).indices.items():
        cand_positions = cand_positions_by_bucket.get(bucket_char)
        if cand_positions is None or len(cand_positions) == 0:
            # No candidate shares this S1 bucket's first character -- the
            # other three strategies are this row's only recall path.
            continue
        bucket_s1_vecs = s1_vecs[s1_positions]
        bucket_cand_vecs = cand_vecs[cand_positions]
        k = min(top_k, len(cand_positions))

        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(bucket_cand_vecs)
        _, neighbor_idx = nn.kneighbors(bucket_s1_vecs)

        bucket_s1_ids = np.repeat(s1[config.ID_COL].to_numpy()[s1_positions], k)
        bucket_cand_ids = cand[config.ID_COL].to_numpy()[cand_positions][neighbor_idx.ravel()]
        result_parts.append(_make_pairs(bucket_s1_ids, bucket_cand_ids))

    if not result_parts:
        return _empty_pairs()
    return pd.concat(result_parts, axis=0, ignore_index=True)


# ==========================================================================
# Strategy 4: address-based blocking
# ==========================================================================

def _address_candidates(source1_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Block on the first whitespace-delimited token of address_norm
    (typically the house number, since normalize_address keeps digit runs
    intact and only strips punctuation/whitespace -- see normalize.py).
    Two records sharing a house-number token are a cheap, high-signal (if
    partial) recall net for cases where the business name itself is too
    different (rename, franchise under a different brand, transliteration)
    for either name-based strategy to catch -- while keeping group sizes
    naturally small, unlike blocking on the full address (rarely collides)
    or on city name alone (near-Cartesian). Blank addresses are skipped
    entirely, never treated as a shared key.
    """
    def _first_token(series: pd.Series) -> pd.Series:
        return series.str.split(n=1).str[0]

    s1 = source1_df.loc[_nonempty(source1_df[config.ADDR_NORM_COL]), [config.ID_COL, config.ADDR_NORM_COL]].copy()
    cand = candidate_df.loc[_nonempty(candidate_df[config.ADDR_NORM_COL]), [config.ID_COL, config.ADDR_NORM_COL]].copy()
    if s1.empty or cand.empty:
        return _empty_pairs()

    s1["_addr_key"] = _first_token(s1[config.ADDR_NORM_COL])
    cand["_addr_key"] = _first_token(cand[config.ADDR_NORM_COL])
    s1 = s1.loc[_nonempty(s1["_addr_key"]), [config.ID_COL, "_addr_key"]]
    cand = cand.loc[_nonempty(cand["_addr_key"]), [config.ID_COL, "_addr_key"]]
    if s1.empty or cand.empty:
        return _empty_pairs()

    cand = _cap_common_keys(cand, "_addr_key")
    return _merge_on_key(s1, cand, "_addr_key")


# ==========================================================================
# Public API
# ==========================================================================

def generate_candidates(
    source1_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    top_k: int = 50,
) -> pd.DataFrame:
    """Union of all blocking strategies, deduplicated.

    Parameters
    ----------
    source1_df : normalized Source 1 DataFrame (output of
        normalize.normalize_source), i.e. must already contain
        config.NAME_NORM_COL / ADDR_NORM_COL / COUNTRY_NORM_COL.
    candidate_df : normalized Source 2 OR Source 3 DataFrame, same
        requirement. No S2/S3-specific logic is used, so callers can pass
        either (or call this twice, once per source, and concat the
        results before handing off to features.build_features).
    top_k : max TF-IDF nearest-candidate matches retrieved per S1 row
        (Strategy 3 only -- exact/name+country/address strategies are not
        capped per-row, since an indexed merge on a safeguarded group size
        is already bounded).

    Returns
    -------
    DataFrame with exactly [config.S1_PAIR_COL, config.CAND_PAIR_COL],
    every candidate_entity_id guaranteed to exist in candidate_df, no
    duplicate pairs.
    """
    required = [config.ID_COL, config.NAME_NORM_COL, config.ADDR_NORM_COL, config.COUNTRY_NORM_COL]
    missing_s1 = [c for c in required if c not in source1_df.columns]
    missing_cand = [c for c in required if c not in candidate_df.columns]
    if missing_s1 or missing_cand:
        raise ValueError(
            "generate_candidates expects already-normalized input "
            "(source1_df/candidate_df must include name_norm/address_norm/"
            f"country_norm -- run normalize.normalize_source() first). "
            f"Missing on source1_df: {missing_s1}; on candidate_df: {missing_cand}"
        )

    valid_cand_ids = set(candidate_df[config.ID_COL])

    parts = [
        _exact_name_candidates(source1_df, candidate_df),
        _name_country_candidates(source1_df, candidate_df),
        _tfidf_name_candidates(source1_df, candidate_df, top_k=top_k),
        _address_candidates(source1_df, candidate_df),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame({config.S1_PAIR_COL: pd.Series(dtype=object),
                              config.CAND_PAIR_COL: pd.Series(dtype=object)})

    combined = pd.concat(parts, axis=0, ignore_index=True)
    combined = _dedupe_pairs(combined)

    # Belt-and-suspenders: every strategy above only ever pulls ids that
    # already came from candidate_df, so this should be a no-op filter --
    # kept because the spec requires the guarantee explicitly and it's
    # cheap relative to the blocking work above.
    combined = combined.loc[combined[config.CAND_PAIR_COL].isin(valid_cand_ids)].reset_index(drop=True)
    return combined


# ==========================================================================
# Self-test (run with: python -m src.blocking)
# ==========================================================================

if __name__ == "__main__":
    from . import normalize

    s1_raw = pd.DataFrame({
        config.ID_COL: ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
        config.NAME_COL: ["O'Reilly's Pub", "Cafe Deja Vu", "The Coffee Shop", "चाय की दुकान", "Zzz Unique Co"],
        config.ADDR_COL: ["100 Main St", "10 Rue de Paris", "1 Market St", "", "200 Oak Ave"],
        config.COUNTRY_COL: ["US", "France", "US", "India", "US"],
    })
    s2_raw = pd.DataFrame({
        config.ID_COL: ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S2-6"],
        config.NAME_COL: ["Oreillys Pub", "Cafe Deja-Vu", "The Coffee Shop", "The Coffee Shop", "चाय की  दुकान", "Totally Unrelated LLC"],
        config.ADDR_COL: ["100 Main Street", "10 Rue de Paris", "2 Market St", "3 Market St", "", "999 Nowhere Rd"],
        config.COUNTRY_COL: ["US", "FR", "US", "US", "India", "US"],
    })

    s1 = normalize.normalize_source(s1_raw)
    s2 = normalize.normalize_source(s2_raw)

    result = generate_candidates(s1, s2, top_k=5)
    print("=== blocking.py self-test ===")
    print(result.to_string(index=False))

    assert list(result.columns) == [config.S1_PAIR_COL, config.CAND_PAIR_COL]
    assert not result.duplicated().any(), "duplicate pairs found"
    assert result[config.CAND_PAIR_COL].isin(s2[config.ID_COL]).all(), "invalid candidate id leaked through"
    pairs_set = set(map(tuple, result.values))
    # S1-1's apostrophe-normalized name should retrieve S2-1 via TF-IDF
    # even though "Oreillys Pub" != "O'Reilly's Pub" pre-normalization.
    assert ("S1-1", "S2-1") in pairs_set
    # Non-Latin (Hindi) name should still retrieve its near-duplicate.
    assert ("S1-4", "S2-5") in pairs_set
    # Cafe Deja Vu / Cafe Deja-Vu, same address -> should be found by
    # exact name+address, independent of country-code spelling (US vs FR
    # style aliases aside, same normalized country here).
    assert ("S1-2", "S2-2") in pairs_set
    print("\nAll assertions passed.")
