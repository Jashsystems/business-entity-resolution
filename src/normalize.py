"""
normalize.py  (Person 1's component)
=====================================

Deterministic, Unicode-safe normalization of business_name,
business_address, and country.

Design constraints from the project spec (see comments inline for how
each is satisfied):
    - Unicode/non-Latin text must survive (Hindi, Kannada, French accents,
      etc.) -- no [^0-9a-zA-Z\\s] style ASCII-only stripping.
    - Do not blindly delete legal suffixes (Inc/LLC/Ltd/Pvt) -- off by
      default, configurable in config.py.
    - Country normalization must stay open-world -- France (and any other
      future country) must normalize safely without a closed mapping.
    - Missing values must become "" , never "nan"/"none"/"null".
    - Vectorized where practical; no df.apply over millions of rows.

Public API
----------
    normalize_name(value)      -> str
    normalize_address(value)   -> str
    normalize_country(value)   -> str
    normalize_source(df, ...)  -> df with *_norm columns added

==========================================================================
INTEGRATION NOTE FOR main.py (Person 4 / whoever wires the pipeline)
==========================================================================
features.py ships its OWN fallback normalizer (features.normalize_text),
used only when build_features(...) is called with the default
already_normalized=False. That fallback uses an ASCII-only regex
([^0-9a-zA-Z\\s]) and will silently destroy non-Latin text (Hindi,
Kannada, etc.) and mangle accented text (e.g. "Café" -> "caf") -- exactly
what this module exists to avoid.

main.py MUST call normalize_source() on every source table first, then
call build_features(..., already_normalized=True, name_col=config.NAME_NORM_COL,
addr_col=config.ADDR_NORM_COL, country_col=config.COUNTRY_NORM_COL).
Skipping this silently bypasses all of the Unicode-safety work below.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config

# ==========================================================================
# Missing-value handling
# ==========================================================================

def _safe_str(x) -> str:
    """Coerce any scalar to a plain string, never producing "nan"/"None"/
    "null" text and never raising on odd input (NaN, None, non-str)."""
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


# ==========================================================================
# Shared character-level helpers
# ==========================================================================

# Curly/backtick/prime quote variants -> treated as apostrophes before the
# apostrophe-handling step below, so "O’Reilly", "O`Reilly", "O´Reilly"
# all normalize the same way as "O'Reilly".
_APOSTROPHE_VARIANTS = "\u2018\u2019\u02bc\u0060\u00b4"
_APOSTROPHE_TRANSLATION = str.maketrans({c: "'" for c in _APOSTROPHE_VARIANTS})

# Whitespace collapse.
_WS_RE = re.compile(r"\s+")

# Punctuation/separator stripping -- ASCII fast path only. IMPORTANT: this
# is deliberately NOT used directly on non-ASCII text. Python's `\w` is
# Unicode-aware for *letters* (it matches Devanagari/Kannada/etc. base
# letters fine) but is defined via str.isalnum(), which does NOT count
# Unicode combining marks (category Mn/Mc -- e.g. Devanagari vowel signs
# and virama, which are separate codepoints, not precomposed) as word
# characters. A naive `[^\w\s]` substitution therefore silently strips
# vowel signs out of Indic-script text ("चाय" -> "च य", losing the matra)
# -- exactly the kind of non-Latin destruction the spec warns against.
# See _strip_non_word() below for the Unicode-category-aware path used
# for any non-ASCII input.
#
# `_` is stripped explicitly (`|_`) even though `\w` normally treats it as
# a word character: the accurate/non-ASCII path below keeps only Unicode
# categories L/M/N (letters, marks, numbers), which excludes underscore
# (category Pc). Without the explicit `|_` here, "Foo_Bar" would normalize
# differently depending on whether some *other*, unrelated character in
# the same string happened to be non-ASCII -- an inconsistency caught in
# review, not by the original self-test.
_NON_WORD_RE = re.compile(r"[^\w\s]|_", flags=re.UNICODE)


def _strip_non_word(s: str) -> str:
    """Replace punctuation/symbols with a single space, keeping letters,
    digits, and Unicode combining marks -- safe for every script.

    Fast path: pure-ASCII strings (the large majority of this dataset's
    business names/addresses) use the compiled regex above, which is a
    single C-level pass over the string.

    Accurate path: any string containing non-ASCII characters (accented
    French text, Devanagari, Kannada, etc.) is scanned character-by-
    character using unicodedata.category(), keeping category groups L
    (letters), M (combining marks), and N (numbers), and turning
    everything else into a space. This is the only way to correctly keep
    combining marks, which `\\w` does not recognize as word characters.
    """
    if s.isascii():
        return _NON_WORD_RE.sub(" ", s)
    out = []
    for ch in s:
        if ch.isspace():
            out.append(" ")
        elif unicodedata.category(ch)[0] in ("L", "M", "N"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)

def _nfkc(s: str) -> str:
    try:
        return unicodedata.normalize("NFKC", s)
    except Exception:
        # Never let a pathological/garbled string crash normalization;
        # degrade to the original text rather than raising.
        return s


def _strip_apostrophes(s: str, enabled: bool) -> str:
    s = s.translate(_APOSTROPHE_TRANSLATION)
    if enabled:
        # Delete (not replace-with-space): "o'reilly" -> "oreilly", so it
        # can still match a variant of the same name with no apostrophe
        # at all, which is the common case across independently-sourced
        # business listings.
        s = s.replace("'", "")
    else:
        s = s.replace("'", " ")
    return s


def _strip_legal_suffix(tokens: list) -> list:
    """Drop a single trailing legal-form token (inc/llc/ltd/...), only as
    a whole trailing token match -- never a substring match -- so a name
    that legitimately contains e.g. "Coinc" or "Cornerstone" is untouched.
    Only called when config.STRIP_LEGAL_SUFFIXES is True.
    """
    if tokens and tokens[-1] in config.LEGAL_SUFFIXES:
        return tokens[:-1]
    return tokens


# ==========================================================================
# Public: normalize_name
# ==========================================================================

def normalize_name(value) -> str:
    """Normalize a business_name value.

    Pipeline: coerce-missing-safe -> NFKC -> casefold -> apostrophe
    handling -> Unicode-safe punctuation-to-space -> whitespace collapse
    -> (optional, off by default) trailing legal-suffix strip.

    Non-Latin scripts (Hindi, Kannada, etc.) pass through: casefold() and
    the \\w-based punctuation filter are both Unicode-aware, so this never
    transliterates to ASCII and never deletes non-Latin letters.
    """
    s = _safe_str(value)
    if not s:
        return ""
    s = _nfkc(s)
    s = s.casefold()
    s = _strip_apostrophes(s, config.STRIP_APOSTROPHES)
    s = _strip_non_word(s)
    s = _WS_RE.sub(" ", s).strip()
    if config.STRIP_LEGAL_SUFFIXES and s:
        tokens = _strip_legal_suffix(s.split(" "))
        s = " ".join(tokens)
    return s


# ==========================================================================
# Public: normalize_address
# ==========================================================================

def normalize_address(value) -> str:
    """Normalize a business_address value.

    Same core pipeline as normalize_name, but WITHOUT legal-suffix
    stripping (not applicable to addresses). House numbers, unit numbers,
    and postal codes are digit runs and are deliberately kept intact --
    feature extraction (postal/house-number matching) happens downstream
    in features.py, which expects them present in the normalized text.

    (An earlier version of this function pre-collapsed repeated ","/";"
    separators before punctuation stripping. That step was a no-op: the
    punctuation-to-space pass below already turns every comma/semicolon
    into whitespace, which is then collapsed anyway -- so it was removed
    rather than kept as dead code.)
    """
    s = _safe_str(value)
    if not s:
        return ""
    s = _nfkc(s)
    s = s.casefold()
    s = _strip_apostrophes(s, config.STRIP_APOSTROPHES)
    s = _strip_non_word(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ==========================================================================
# Public: normalize_country
# ==========================================================================

def normalize_country(value) -> str:
    """Normalize a country value conservatively and OPEN-WORLD.

    1. Generic, safe cleanup: NFKC, casefold, strip periods (so "U.S.A."
       and "USA" clean to the same pre-alias string), collapse whitespace.
    2. Look the cleaned value up in config.COUNTRY_ALIASES, a small,
       extensible table of KNOWN spelling variants seen in this project's
       data (currently US/India/France). If found, return the canonical
       code.
    3. If NOT found -- an unseen country, e.g. one that only appears in a
       later test set -- return the cleaned-but-unmapped string as-is,
       rather than collapsing it to "unknown"/empty/some default. This is
       what keeps the function open-world: a country the model has never
       seen still gets a stable, deterministic normalized value and never
       spuriously matches an unrelated country because both were dumped
       into the same "unknown" bucket.

    Missing country -> "" (handled like any other missing field; never
    treated as a match against another missing country by this function
    itself -- that decision belongs to features.py's *_missing_* logic).
    """
    s = _safe_str(value)
    if not s:
        return ""
    s = _nfkc(s)
    s = s.casefold()
    s = s.replace(".", "")
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    return config.COUNTRY_ALIASES.get(s, s)


# ==========================================================================
# Public: normalize_source
# ==========================================================================

def normalize_source(
    df: pd.DataFrame,
    *,
    id_col: str = config.ID_COL,
    name_col: str = config.NAME_COL,
    addr_col: str = config.ADDR_COL,
    country_col: str = config.COUNTRY_COL,
) -> pd.DataFrame:
    """Add name_norm / address_norm / country_norm columns to a source
    DataFrame. Original columns (entity_id, business_name,
    business_address, country) are left completely intact -- this never
    mutates or drops the raw fields, since Person 3's fallback path and
    manual QC both want the original text available too.

    Vectorization note: NFKC normalization and casefold have no
    Series-level C-accelerated path in pandas/numpy (they are inherently
    per-character Unicode table lookups), so this uses a single
    Series.map(...) pass per column -- one Python-level call per row, but
    only ONE pass doing all the work per column, rather than chaining
    several separate .str.xxx() passes (each of which would re-scan the
    full column). At ~5M rows this runs in low tens of seconds per column
    on typical hardware, which is acceptable for a one-time preprocessing
    step; do not replace with df.apply(axis=1) (that iterates row-wise
    across *all* columns per call and is substantially slower here).

    Returns a new DataFrame (does not mutate the input in place).
    """
    out = df.copy(deep=False)
    out[config.NAME_NORM_COL] = out[name_col].map(normalize_name)
    out[config.ADDR_NORM_COL] = out[addr_col].map(normalize_address)
    out[config.COUNTRY_NORM_COL] = out[country_col].map(normalize_country)
    return out


# ==========================================================================
# Self-test / validation (run with: python -m src.normalize)
# ==========================================================================

if __name__ == "__main__":
    cases = [
        ("O'Reilly's Barbershop", "name"),
        ("O’REILLY’S BARBERSHOP", "name"),
        ("1795 Westchester Drive, High Point, NC", "address"),
        ("1795 Westchester Dr., High Point NC", "address"),
        ("Café Déjà Vu", "name"),
        ("Société Générale — Succursale de Lyon", "name"),
        ("चाय की दुकान", "name"),           # Hindi
        ("ಕಾಫಿ ಅಂಗಡಿ", "name"),              # Kannada
        ("", "name"),
        (None, "address"),
        ("US", "country"),
        ("U.S.A.", "country"),
        ("United States", "country"),
        ("France", "country"),
        ("Freedonia", "country"),           # unseen/unknown country
    ]
    fn = {"name": normalize_name, "address": normalize_address, "country": normalize_country}
    print("=== normalize.py self-test ===")
    for value, kind in cases:
        result = fn[kind](value)
        print(f"[{kind:7s}] {value!r:45s} -> {result!r}")

    # Determinism check.
    assert normalize_name("O'Reilly's Barbershop") == normalize_name("O'Reilly's Barbershop")
    # Non-Latin survives -- including combining vowel signs, which a
    # naive \w-based filter would silently strip (that was a real bug
    # caught by this exact assertion during development; see
    # _strip_non_word()'s docstring).
    assert normalize_name("चाय की दुकान") == "चाय की दुकान"
    assert normalize_name("ಕಾಫಿ ಅಂಗಡಿ") == "ಕಾಫಿ ಅಂಗಡಿ"
    # Missing never becomes the literal string "nan".
    assert normalize_name(None) == ""
    assert normalize_address(float("nan")) == ""
    # NOTE on the two address examples above ("Westchester Drive" vs
    # "Westchester Dr."): normalize_address does NOT expand/collapse
    # abbreviations like Dr./Drive -- that gap is intentionally left to
    # fuzzy matching (features.py's char/token similarity features), not
    # solved here, per the spec's "don't destroy useful information"
    # constraint. What normalize_address *does* guarantee for that pair
    # is deterministic Unicode/case/punctuation/whitespace cleanup.
    assert normalize_address("1795 Westchester Drive, High Point, NC") == \
        "1795 westchester drive high point nc"
    assert normalize_address("1795 Westchester Dr., High Point NC") == \
        "1795 westchester dr high point nc"
    # Unknown country still normalizes deterministically, not to "unknown".
    assert normalize_country("Freedonia") == "freedonia"
    assert normalize_country("France") == "fr"
    print("\nAll assertions passed.")
