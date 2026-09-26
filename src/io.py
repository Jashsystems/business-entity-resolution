"""
io.py  (Person 1's component)
==============================

Reliable TSV loading for the Business Entity Resolution pipeline.

Responsibilities (and ONLY these):
    - read train/test source tables (S1/S2/S3) and ground truth from TSV
    - preserve entity_id (and all text columns) as plain strings
    - preserve Unicode
    - represent missing/empty fields as "" (never "nan"/"none"/"null")
    - validate that required columns are present
    - support chunked reading for the multi-million-row files

This module does NOT: merge sources, build candidate pairs, normalize
text, or do anything downstream of raw loading. See normalize.py for
normalization and blocking.py (Person 2) for candidate generation.

Public API
----------
    load_source(path, ...)          -> DataFrame or chunk iterator
    load_ground_truth(path, ...)    -> DataFrame or chunk iterator
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

import pandas as pd

from . import config


# ==========================================================================
# Internal helpers
# ==========================================================================

def _validate_columns(df: pd.DataFrame, required: list, path: Union[str, Path]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}; "
            f"found columns: {list(df.columns)}"
        )


def _read_csv_kwargs(dtype_cols: list, sep: str) -> dict:
    """Shared pd.read_csv kwargs that keep every listed column as plain
    Python strings and keep missing/empty cells as "" rather than NaN.

    keep_default_na=False + na_values=[] turns off pandas' automatic NA
    sniffing (which would otherwise treat blank cells, and tokens like the
    literal strings "NA"/"NaN"/"null" if they ever appear in the data, as
    NaN). Combined with dtype=str this means:
      - entity_id is never silently coerced to an integer/float
      - Unicode text passes through untouched
      - a genuinely empty business_address cell becomes "" directly,
        never the string "nan"
    """
    return dict(
        sep=sep,
        dtype={c: str for c in dtype_cols},
        keep_default_na=False,
        na_values=[],
        encoding=config.FILE_ENCODING,
    )


def _strip_and_fill(df: pd.DataFrame, dtype_cols: list) -> pd.DataFrame:
    """Belt-and-suspenders cleanup: even with keep_default_na=False, a
    column can end up with an actual NaN if pandas still infers a numeric
    dtype for an all-blank column in some edge case, or if upstream data
    contains a true null byte /  encoding artifact. Coerce any such
    leftover NaN/None to "" so downstream code never sees "nan" as text.
    Avoids unnecessary full-frame copies -- only touches the given columns.
    """
    for c in dtype_cols:
        if c in df.columns:
            col = df[c]
            if col.isna().any():
                df[c] = col.fillna("")
    return df


# ==========================================================================
# Public: load_source
# ==========================================================================

def load_source(
    path: Union[str, Path],
    *,
    source_prefix: str = None,
    chunksize: int = None,
    required_columns: list = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Load one source table (S1, S2, or S3) from a TSV file.

    Parameters
    ----------
    path : path to the source TSV.
    source_prefix : if given (e.g. config.S1_PREFIX), validates that every
        entity_id in the file starts with this prefix. Purely a sanity
        check -- does not alter the data. Skipped when chunksize is set
        (would defeat the point of chunked/low-memory reading); validate
        chunk-by-chunk yourself if needed.
    chunksize : if given, returns an iterator of DataFrames (one per
        chunk of this many rows) instead of loading the whole file into
        memory. Use for the ~5M-row S2/S3 files on constrained memory.
    required_columns : columns that must be present; defaults to
        config.REQUIRED_SOURCE_COLUMNS (entity_id, business_name,
        business_address, country).

    Returns
    -------
    A single DataFrame (chunksize=None) or a generator of DataFrames
    (chunksize set), each with:
        - entity_id as string, exact values preserved (no numeric coercion)
        - business_name, business_address, country as strings
        - missing/empty cells as "" (never "nan"/"none"/"null")
        - original row order preserved
        - no rows dropped, no dedup performed
    """
    path = Path(path)
    required = required_columns or config.REQUIRED_SOURCE_COLUMNS
    kwargs = _read_csv_kwargs(required, config.TSV_SEPARATOR)

    if chunksize is None:
        df = pd.read_csv(path, **kwargs)
        _validate_columns(df, required, path)
        df = _strip_and_fill(df, required)
        if source_prefix is not None:
            _check_prefix(df, config.ID_COL, source_prefix, path)
        return df

    def _chunk_gen() -> Iterator[pd.DataFrame]:
        reader = pd.read_csv(path, chunksize=chunksize, **kwargs)
        first = True
        for chunk in reader:
            if first:
                _validate_columns(chunk, required, path)
                first = False
            chunk = _strip_and_fill(chunk, required)
            yield chunk

    return _chunk_gen()


def _check_prefix(df: pd.DataFrame, id_col: str, prefix: str, path: Union[str, Path]) -> None:
    bad = ~df[id_col].str.startswith(prefix)
    n_bad = int(bad.sum())
    if n_bad:
        example = df.loc[bad, id_col].iloc[0]
        raise ValueError(
            f"{path}: {n_bad} entity_id value(s) do not start with "
            f"expected prefix {prefix!r} (e.g. {example!r})"
        )


# ==========================================================================
# Public: load_ground_truth
# ==========================================================================

def load_ground_truth(
    path: Union[str, Path],
    *,
    chunksize: int = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Load the ground truth TSV (source1_entity_id, matched_entity_ids).

    A row whose matched_entity_ids is empty means that S1 entity has zero
    true matches (this is a valid, expected case per the project spec --
    ~5.6% of training S1 entities). Empty is represented as "" here, not
    as NaN or the string "nan"; features.py's build_training_labels()
    already assumes this ("" splits to no matches after filtering).

    Parameters mirror load_source(): chunksize=None loads the whole file
    (ground truth is one row per S1 entity, same row count as source1, so
    this is generally fine to load in full); chunksize returns a chunk
    iterator for symmetry / very constrained memory.
    """
    path = Path(path)
    required = config.REQUIRED_GT_COLUMNS
    kwargs = _read_csv_kwargs(required, config.TSV_SEPARATOR)

    if chunksize is None:
        df = pd.read_csv(path, **kwargs)
        _validate_columns(df, required, path)
        df = _strip_and_fill(df, required)
        return df

    def _chunk_gen() -> Iterator[pd.DataFrame]:
        reader = pd.read_csv(path, chunksize=chunksize, **kwargs)
        first = True
        for chunk in reader:
            if first:
                _validate_columns(chunk, required, path)
                first = False
            chunk = _strip_and_fill(chunk, required)
            yield chunk

    return _chunk_gen()
