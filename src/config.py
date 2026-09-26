"""
config.py  (Person 1's component)
==================================

Centralized configuration for the Business Entity Resolution pipeline.
Nothing here does I/O or normalization itself -- this module is pure
constants/paths so every other module (io.py, normalize.py, blocking.py,
features.py, model.py, evaluate.py, main.py) reads from one place instead
of re-declaring column names / paths / seeds independently.

All paths are project-relative (pathlib), never hard-coded absolute paths.
Override via environment variable BER_DATA_DIR if the data lives somewhere
else (e.g. a different mount point in a grading environment) without
touching this file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ==========================================================================
# Project paths
# ==========================================================================

# Repository root = parent of the src/ directory this file lives in.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data directory can be relocated via env var without code changes.
DATA_DIR: Path = Path(os.environ.get("BER_DATA_DIR", PROJECT_ROOT / "data")).resolve()

TRAIN_DIR: Path = DATA_DIR / "train"
TEST_DIR: Path = DATA_DIR / "test"

OUTPUT_DIR: Path = Path(os.environ.get("BER_OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
MODEL_DIR: Path = OUTPUT_DIR / "models"

# Raw source file paths. If the actual repo lays files out flat (no
# train/ test/ subfolders), adjust these two blocks only -- every other
# module consumes the *_PATH constants below, never raw filenames.
TRAIN_SOURCE1_PATH: Path = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH: Path = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH: Path = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH: Path = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1_PATH: Path = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH: Path = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH: Path = TEST_DIR / "test_source3.tsv"

PREDICTIONS_OUTPUT_PATH: Path = OUTPUT_DIR / "matching_results.tsv"


# ==========================================================================
# Column names (single source of truth -- must match features.py's
# ID_COL/NAME_COL/ADDR_COL/COUNTRY_COL and model.py's group_col/label_col
# defaults; features.py already agrees with these exact strings)
# ==========================================================================

ID_COL: str = "entity_id"
NAME_COL: str = "business_name"
ADDR_COL: str = "business_address"
COUNTRY_COL: str = "country"

REQUIRED_SOURCE_COLUMNS: list = [ID_COL, NAME_COL, ADDR_COL, COUNTRY_COL]

# Normalized-column names produced by normalize.py's normalize_source().
NAME_NORM_COL: str = "name_norm"
ADDR_NORM_COL: str = "address_norm"
COUNTRY_NORM_COL: str = "country_norm"

# Ground truth columns (must match features.py's GT_S1_COL / GT_MATCH_COL).
GT_S1_COL: str = "source1_entity_id"
GT_MATCH_COL: str = "matched_entity_ids"
REQUIRED_GT_COLUMNS: list = [GT_S1_COL, GT_MATCH_COL]

# Candidate-pair columns (must match features.py's S1_PAIR_COL / CAND_PAIR_COL).
S1_PAIR_COL: str = "source1_entity_id"
CAND_PAIR_COL: str = "candidate_entity_id"


# ==========================================================================
# Source ID prefixes
# ==========================================================================
# NOTE: these route candidate ids to the right lookup table (see
# features.py's cand_lookup) -- they are structural, not a "country list",
# so hard-coding them is safe and intentional.

S1_PREFIX: str = "S1"
S2_PREFIX: str = "S2"
S3_PREFIX: str = "S3"

SOURCE_PREFIXES: dict = {
    "source1": S1_PREFIX,
    "source2": S2_PREFIX,
    "source3": S3_PREFIX,
}


# ==========================================================================
# I/O settings
# ==========================================================================

TSV_SEPARATOR: str = "\t"

# Default chunk size for chunked reading of the multi-million-row sources.
# 500k rows/chunk keeps peak memory bounded on the ~5.3M-row S3 files
# while still amortizing per-chunk pandas overhead reasonably well.
DEFAULT_CHUNK_SIZE: int = 500_000

# Encoding assumption for the provided TSVs. Kept as a named constant
# (rather than repeated literal "utf-8") so it is a one-line change if a
# grading environment ever supplies a different encoding.
FILE_ENCODING: str = "utf-8"


# ==========================================================================
# Normalization settings
# ==========================================================================

# Whether to strip apostrophe/quote characters entirely (no replacement
# space) during name/address normalization, e.g. "O'Reilly" -> "oreilly".
# This is the safer default for fuzzy/token matching across sources that
# may or may not include the apostrophe. Set False to instead treat
# apostrophes as a word separator (a space).
STRIP_APOSTROPHES: bool = True

# Legal-suffix stripping is OFF by default per the project spec ("do not
# blindly remove business suffixes ... unless there is a very strong
# reason and it is configurable"). If enabled, only the suffixes listed
# below are stripped, and only as a trailing token match, never a
# substring match, to avoid mangling names that legitimately contain
# these strings mid-name.
STRIP_LEGAL_SUFFIXES: bool = False
LEGAL_SUFFIXES: list = [
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "pvt", "private", "plc",
]

# Conservative, EXTENSIBLE country alias table used only to canonicalize
# well-known spelling variants seen in this project's data (US/USA/India
# so far, France appearing in test). This is NOT a closed-world mapping:
# normalize_country() falls through to a cleaned-but-unmapped value for
# any country not listed here, so unseen countries (or new ones added
# later) are still normalized (case/whitespace/Unicode) and remain usable
# -- they just aren't canonicalized to a shared code. Extend this dict as
# needed; normalize.py's logic never needs to change.
COUNTRY_ALIASES: dict = {
    "us": "us", "usa": "us", "u.s.": "us", "u.s.a.": "us",
    "united states": "us", "united states of america": "us",
    "in": "in", "india": "in",
    "fr": "fr", "france": "fr",
}


# ==========================================================================
# Reproducibility
# ==========================================================================

RANDOM_SEED: int = 42
