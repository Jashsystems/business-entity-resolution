"""
Dataset profiler for the Business Entity Resolution competition.

Usage:
    python src/dataset_profile.py

Optional:
    python src/dataset_profile.py --data-dir data
    python src/dataset_profile.py --chunk-size 100000
    python src/dataset_profile.py --output dataset_profile.txt

The profiler:
- Reads TSV files incrementally.
- Does not load the complete 2GB dataset into RAM.
- Reports schema, row counts, missingness, lengths, countries,
  ID uniqueness, duplicate normalized names/addresses, and
  ground-truth match-count distributions.
- Never prints actual business names or addresses.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


# ============================================================
# Configuration
# ============================================================

EXPECTED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

DEFAULT_CHUNK_SIZE = 100_000

TEXT_COLUMNS = [
    "business_name",
    "business_address",
    "country",
]


# ============================================================
# Basic utilities
# ============================================================

def normalize_text(value: object) -> str:
    """
    Conservative normalization used ONLY for profiling duplicate
    rates. This is not necessarily the final competition normalization.
    """
    if value is None:
        return ""

    text = str(value)

    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    # Replace punctuation with spaces rather than deleting everything.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def hash_normalized(value: str) -> str:
    """
    Hash normalized text before putting it into the profiling database.

    We don't need to store actual names/addresses.
    SHA-1 collision probability is negligible for this profiling task.
    """
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def is_missing(value: object) -> bool:
    if value is None:
        return True

    text = str(value).strip()

    return (
        text == ""
        or text.lower() in {
            "nan",
            "none",
            "null",
            "n/a",
            "na",
        }
    )


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return float(values[lower])

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def format_number(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"

    return f"{value:,}"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


# ============================================================
# SQLite duplicate tracker
# ============================================================

class HashTracker:
    """
    Disk-backed tracker for normalized values.

    This prevents a huge Python set from consuming RAM when
    profiling millions of records.
    """

    def __init__(self, path: Path, table_name: str):
        self.path = path

        self.connection = sqlite3.connect(str(path))

        self.table_name = table_name

        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                value_hash TEXT PRIMARY KEY
            )
            """
        )

        self.connection.commit()

        self.total_nonempty = 0

    def add_many(self, values: Iterable[str]) -> None:
        hashes = []

        for value in values:
            if not value:
                continue

            self.total_nonempty += 1
            hashes.append((hash_normalized(value),))

        if not hashes:
            return

        self.connection.executemany(
            f"""
            INSERT OR IGNORE INTO {self.table_name}(value_hash)
            VALUES (?)
            """,
            hashes,
        )

        self.connection.commit()

    def unique_count(self) -> int:
        cursor = self.connection.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        )

        return int(cursor.fetchone()[0])

    def duplicate_count(self) -> int:
        return max(
            0,
            self.total_nonempty - self.unique_count(),
        )

    def close(self) -> None:
        self.connection.close()


# ============================================================
# File discovery
# ============================================================

def discover_files(data_dir: Path) -> dict[str, Path]:
    """
    Discover competition files.

    Supports both:

        data/train_source1.tsv

    and:

        data/train/train_source1.tsv

    etc.
    """

    expected_names = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]

    found: dict[str, Path] = {}

    for filename in expected_names:
        direct = data_dir / filename

        if direct.exists():
            found[filename] = direct
            continue

        matches = list(data_dir.rglob(filename))

        if matches:
            found[filename] = matches[0]

    return found


# ============================================================
# File statistics
# ============================================================

class FileStats:
    def __init__(self, filename: str):
        self.filename = filename

        self.rows = 0

        self.columns: list[str] = []

        self.missing = Counter()

        self.empty = Counter()

        self.lengths: dict[str, list[int]] = {
            column: []
            for column in TEXT_COLUMNS
        }

        self.countries = Counter()

        self.id_prefixes = Counter()

        self.entity_ids_seen = set()

        self.duplicate_entity_ids = 0

        self.name_tracker: HashTracker | None = None

        self.address_tracker: HashTracker | None = None

    def report(self) -> str:
        lines = []

        lines.append("")
        lines.append("=" * 72)
        lines.append(f"FILE: {self.filename}")
        lines.append("=" * 72)

        lines.append(f"Rows: {format_number(self.rows)}")

        lines.append("")
        lines.append("Columns:")
        for column in self.columns:
            lines.append(f"  - {column}")

        lines.append("")
        lines.append("Missing / empty values:")

        for column in self.columns:
            missing = self.missing[column]
            empty = self.empty[column]

            missing_pct = (
                missing / self.rows * 100
                if self.rows
                else 0
            )

            empty_pct = (
                empty / self.rows * 100
                if self.rows
                else 0
            )

            lines.append(
                f"  {column}: "
                f"missing={format_number(missing)} "
                f"({format_percent(missing_pct)}), "
                f"empty={format_number(empty)} "
                f"({format_percent(empty_pct)})"
            )

        lines.append("")
        lines.append("Text length statistics:")

        for column in TEXT_COLUMNS:
            values = self.lengths.get(column, [])

            if not values:
                continue

            lines.append(f"  {column}:")
            lines.append(
                f"    min={min(values)}, "
                f"p25={percentile(values, 0.25):.1f}, "
                f"median={percentile(values, 0.50):.1f}, "
                f"p75={percentile(values, 0.75):.1f}, "
                f"p95={percentile(values, 0.95):.1f}, "
                f"max={max(values)}"
            )

        if self.countries:
            lines.append("")
            lines.append(
                f"Unique countries: {len(self.countries):,}"
            )

            lines.append("Top countries by record count:")

            for country, count in self.countries.most_common(20):
                # Country values are safe to report and are not PII.
                display_country = country if country else "<EMPTY>"

                percentage = (
                    count / self.rows * 100
                    if self.rows
                    else 0
                )

                lines.append(
                    f"  {display_country}: "
                    f"{format_number(count)} "
                    f"({format_percent(percentage)})"
                )

        if self.id_prefixes:
            lines.append("")
            lines.append("Entity ID prefixes:")

            for prefix, count in self.id_prefixes.most_common():
                lines.append(
                    f"  {prefix}: {format_number(count)}"
                )

        if self.columns and "entity_id" in self.columns:
            unique_ids = len(self.entity_ids_seen)

            lines.append("")
            lines.append("Entity ID statistics:")
            lines.append(
                f"  Unique IDs: {format_number(unique_ids)}"
            )
            lines.append(
                f"  Duplicate ID occurrences: "
                f"{format_number(self.duplicate_entity_ids)}"
            )

        if self.name_tracker:
            unique_names = self.name_tracker.unique_count()
            duplicate_names = self.name_tracker.duplicate_count()

            lines.append("")
            lines.append("Normalized business-name duplication:")
            lines.append(
                f"  Non-empty normalized names: "
                f"{format_number(self.name_tracker.total_nonempty)}"
            )
            lines.append(
                f"  Unique normalized names: "
                f"{format_number(unique_names)}"
            )
            lines.append(
                f"  Duplicate occurrences: "
                f"{format_number(duplicate_names)}"
            )

        if self.address_tracker:
            unique_addresses = self.address_tracker.unique_count()
            duplicate_addresses = self.address_tracker.duplicate_count()

            lines.append("")
            lines.append("Normalized business-address duplication:")
            lines.append(
                f"  Non-empty normalized addresses: "
                f"{format_number(self.address_tracker.total_nonempty)}"
            )
            lines.append(
                f"  Unique normalized addresses: "
                f"{format_number(unique_addresses)}"
            )
            lines.append(
                f"  Duplicate occurrences: "
                f"{format_number(duplicate_addresses)}"
            )

        return "\n".join(lines)


# ============================================================
# Profiling a regular source TSV
# ============================================================

def profile_source_file(
    path: Path,
    chunk_size: int,
    tracker_dir: Path,
) -> FileStats:

    print(f"\nProfiling: {path}")
    print("This may take a while for large files...")

    stats = FileStats(path.name)

    safe_name = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        path.stem,
    )

    name_db = tracker_dir / f"{safe_name}_names.sqlite"
    address_db = tracker_dir / f"{safe_name}_addresses.sqlite"

    stats.name_tracker = HashTracker(
        name_db,
        "normalized_names",
    )

    stats.address_tracker = HashTracker(
        address_db,
        "normalized_addresses",
    )

    start_time = time.time()

    try:
        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
            encoding="utf-8",
            on_bad_lines="warn",
        )

        first_chunk = True

        for chunk_number, chunk in enumerate(reader, start=1):

            if first_chunk:
                stats.columns = list(chunk.columns)

                missing_columns = (
                    EXPECTED_COLUMNS - set(stats.columns)
                )

                if missing_columns:
                    print(
                        f"WARNING: {path.name} is missing "
                        f"expected columns: "
                        f"{sorted(missing_columns)}"
                    )

                first_chunk = False

            stats.rows += len(chunk)

            # ------------------------------------------------
            # Missingness and string lengths
            # ------------------------------------------------

            for column in stats.columns:

                if column not in chunk.columns:
                    continue

                series = chunk[column].astype(str)

                missing_mask = series.map(is_missing)

                stats.missing[column] += int(
                    missing_mask.sum()
                )

                stats.empty[column] += int(
                    (series.str.strip() == "").sum()
                )

            for column in TEXT_COLUMNS:

                if column not in chunk.columns:
                    continue

                values = (
                    chunk[column]
                    .astype(str)
                    .str.len()
                    .tolist()
                )

                # Store lengths. This is generally manageable because
                # it stores only integers, but if the dataset is huge,
                # keep a sample instead.
                if len(stats.lengths[column]) < 1_000_000:
                    remaining = (
                        1_000_000
                        - len(stats.lengths[column])
                    )

                    stats.lengths[column].extend(
                        values[:remaining]
                    )

            # ------------------------------------------------
            # Country distribution
            # ------------------------------------------------

            if "country" in chunk.columns:

                countries = (
                    chunk["country"]
                    .astype(str)
                    .map(
                        lambda x: (
                            "<EMPTY>"
                            if is_missing(x)
                            else x.strip()
                        )
                    )
                )

                stats.countries.update(
                    countries.tolist()
                )

            # ------------------------------------------------
            # Entity IDs
            # ------------------------------------------------

            if "entity_id" in chunk.columns:

                for entity_id in chunk["entity_id"].astype(str):

                    if not entity_id:
                        continue

                    prefix_match = re.match(
                        r"^(S\d+)-",
                        entity_id,
                        flags=re.IGNORECASE,
                    )

                    if prefix_match:
                        prefix = prefix_match.group(1).upper()
                    else:
                        prefix = "<OTHER>"

                    stats.id_prefixes[prefix] += 1

                    if entity_id in stats.entity_ids_seen:
                        stats.duplicate_entity_ids += 1
                    else:
                        stats.entity_ids_seen.add(entity_id)

            # ------------------------------------------------
            # Normalized duplicate tracking
            # ------------------------------------------------

            if "business_name" in chunk.columns:

                names = (
                    chunk["business_name"]
                    .astype(str)
                    .map(normalize_text)
                    .tolist()
                )

                stats.name_tracker.add_many(names)

            if "business_address" in chunk.columns:

                addresses = (
                    chunk["business_address"]
                    .astype(str)
                    .map(normalize_text)
                    .tolist()
                )

                stats.address_tracker.add_many(addresses)

            if chunk_number % 10 == 0:
                elapsed = time.time() - start_time

                print(
                    f"  processed "
                    f"{format_number(stats.rows)} rows "
                    f"({elapsed:.1f}s)"
                )

    finally:
        elapsed = time.time() - start_time

    print(
        f"Finished {path.name}: "
        f"{format_number(stats.rows)} rows "
        f"in {elapsed:.1f}s"
    )

    return stats


# ============================================================
# Ground truth profiling
# ============================================================

def profile_ground_truth(
    path: Path,
    chunk_size: int,
) -> str:

    print(f"\nProfiling ground truth: {path}")

    total_rows = 0

    zero_matches = 0
    one_match = 0
    multiple_matches = 0

    match_count_distribution = Counter()

    total_match_links = 0

    malformed_rows = 0

    source_id_counts = Counter()

    start_time = time.time()

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        encoding="utf-8",
        on_bad_lines="warn",
    )

    for chunk in reader:

        total_rows += len(chunk)

        expected = {
            "source1_entity_id",
            "matched_entity_ids",
        }

        missing = expected - set(chunk.columns)

        if missing:
            raise ValueError(
                f"Ground truth is missing columns: {sorted(missing)}"
            )

        for _, row in chunk.iterrows():

            s1_id = str(
                row["source1_entity_id"]
            ).strip()

            matched = str(
                row["matched_entity_ids"]
            ).strip()

            if s1_id:
                source_id_counts[s1_id] += 1

            if not matched:
                count = 0

            else:
                ids = [
                    x.strip()
                    for x in matched.split(",")
                    if x.strip()
                ]

                # Remove duplicates for profiling purposes.
                ids = list(dict.fromkeys(ids))

                count = len(ids)

                total_match_links += count

                # Check ID prefixes.
                for entity_id in ids:
                    if entity_id.startswith("S2-"):
                        pass
                    elif entity_id.startswith("S3-"):
                        pass
                    else:
                        malformed_rows += 1

            match_count_distribution[count] += 1

            if count == 0:
                zero_matches += 1
            elif count == 1:
                one_match += 1
            else:
                multiple_matches += 1

    duplicate_s1_rows = sum(
        count - 1
        for count in source_id_counts.values()
        if count > 1
    )

    lines = []

    lines.append("")
    lines.append("=" * 72)
    lines.append("GROUND TRUTH ANALYSIS")
    lines.append("=" * 72)

    lines.append(
        f"Ground-truth rows: {format_number(total_rows)}"
    )

    lines.append(
        f"Unique Source1 IDs: "
        f"{format_number(len(source_id_counts))}"
    )

    lines.append(
        f"Duplicate Source1 ID occurrences: "
        f"{format_number(duplicate_s1_rows)}"
    )

    lines.append("")
    lines.append("Match-count distribution per Source1:")

    for count in sorted(match_count_distribution):

        number = match_count_distribution[count]

        percentage = (
            number / total_rows * 100
            if total_rows
            else 0
        )

        if count == 0:
            label = "0 matches"
        elif count == 1:
            label = "1 match"
        else:
            label = f"{count} matches"

        lines.append(
            f"  {label}: "
            f"{format_number(number)} "
            f"({format_percent(percentage)})"
        )

    lines.append("")
    lines.append("Summary:")

    lines.append(
        f"  Zero-match Source1: "
        f"{format_number(zero_matches)} "
        f"({format_percent(zero_matches / total_rows * 100 if total_rows else 0)})"
    )

    lines.append(
        f"  Single-match Source1: "
        f"{format_number(one_match)} "
        f"({format_percent(one_match / total_rows * 100 if total_rows else 0)})"
    )

    lines.append(
        f"  Multi-match Source1: "
        f"{format_number(multiple_matches)} "
        f"({format_percent(multiple_matches / total_rows * 100 if total_rows else 0)})"
    )

    lines.append(
        f"  Total match links: "
        f"{format_number(total_match_links)}"
    )

    lines.append(
        f"  Malformed/non-S2/S3 match IDs: "
        f"{format_number(malformed_rows)}"
    )

    elapsed = time.time() - start_time

    lines.append(
        f"  Processing time: {elapsed:.1f}s"
    )

    return "\n".join(lines)


# ============================================================
# Global dataset comparison
# ============================================================

def generate_overall_summary(
    file_stats: dict[str, FileStats],
) -> str:

    lines = []

    lines.append("")
    lines.append("=" * 72)
    lines.append("OVERALL DATASET SUMMARY")
    lines.append("=" * 72)

    for filename, stats in file_stats.items():

        lines.append(
            f"{filename}: "
            f"{format_number(stats.rows)} rows"
        )

    # --------------------------------------------------------
    # Training source sizes
    # --------------------------------------------------------

    train_sources = [
        file_stats.get("train_source1.tsv"),
        file_stats.get("train_source2.tsv"),
        file_stats.get("train_source3.tsv"),
    ]

    train_sources = [
        x for x in train_sources if x is not None
    ]

    if train_sources:

        lines.append("")
        lines.append("Training source sizes:")

        total = 0

        for stats in train_sources:
            total += stats.rows

            lines.append(
                f"  {stats.filename}: "
                f"{format_number(stats.rows)}"
            )

        lines.append(
            f"  Combined: {format_number(total)}"
        )

    # --------------------------------------------------------
    # Test source sizes
    # --------------------------------------------------------

    test_sources = [
        file_stats.get("test_source1.tsv"),
        file_stats.get("test_source2.tsv"),
        file_stats.get("test_source3.tsv"),
    ]

    test_sources = [
        x for x in test_sources if x is not None
    ]

    if test_sources:

        lines.append("")
        lines.append("Test source sizes:")

        total = 0

        for stats in test_sources:
            total += stats.rows

            lines.append(
                f"  {stats.filename}: "
                f"{format_number(stats.rows)}"
            )

        lines.append(
            f"  Combined: {format_number(total)}"
        )

    return "\n".join(lines)


# ============================================================
# Recommendations based on observed statistics
# ============================================================

def generate_initial_observations(
    file_stats: dict[str, FileStats],
) -> str:

    lines = []

    lines.append("")
    lines.append("=" * 72)
    lines.append("INITIAL ENGINEERING OBSERVATIONS")
    lines.append("=" * 72)

    # Country observations.
    all_countries = set()

    for stats in file_stats.values():
        all_countries.update(
            country
            for country in stats.countries
            if country != "<EMPTY>"
        )

    lines.append(
        f"Observed country values across profiled files: "
        f"{len(all_countries):,}"
    )

    if len(all_countries) > 2:
        lines.append(
            "Country is clearly not a binary/fixed training attribute; "
            "keep country handling open-set."
        )

    # Missingness.
    for filename, stats in file_stats.items():

        if stats.rows == 0:
            continue

        name_missing = (
            stats.missing["business_name"]
            / stats.rows
            * 100
        )

        address_missing = (
            stats.missing["business_address"]
            / stats.rows
            * 100
        )

        if name_missing > 5:
            lines.append(
                f"{filename}: business_name has substantial "
                f"missingness ({name_missing:.2f}%). "
                f"Missing-name features will matter."
            )

        if address_missing > 5:
            lines.append(
                f"{filename}: business_address has substantial "
                f"missingness ({address_missing:.2f}%). "
                f"Missing-address features will matter."
            )

    # Duplicate names.
    for filename, stats in file_stats.items():

        if not stats.name_tracker:
            continue

        nonempty = stats.name_tracker.total_nonempty

        if nonempty == 0:
            continue

        duplicate_rate = (
            stats.name_tracker.duplicate_count()
            / nonempty
            * 100
        )

        if duplicate_rate > 20:
            lines.append(
                f"{filename}: normalized business names are highly "
                f"non-unique ({duplicate_rate:.2f}% duplicate occurrences). "
                f"Name-only matching will be dangerous."
            )

    # Duplicate addresses.
    for filename, stats in file_stats.items():

        if not stats.address_tracker:
            continue

        nonempty = stats.address_tracker.total_nonempty

        if nonempty == 0:
            continue

        duplicate_rate = (
            stats.address_tracker.duplicate_count()
            / nonempty
            * 100
        )

        if duplicate_rate > 20:
            lines.append(
                f"{filename}: normalized addresses are highly "
                f"non-unique ({duplicate_rate:.2f}% duplicate occurrences). "
                f"Address-only matching will be dangerous."
            )

    lines.append("")
    lines.append(
        "IMPORTANT: These are profiling observations, not final "
        "model decisions. Validate all modeling choices using "
        "Source1-grouped validation and the official F0.5 metric."
    )

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Profile the Business Entity Resolution dataset "
            "without loading the entire dataset into RAM."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing competition TSV files.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows processed per chunk.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_profile.txt"),
        help="Where to save the profile report.",
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    if not data_dir.exists():
        print(
            f"ERROR: Data directory does not exist:\n"
            f"  {data_dir}"
        )

        return 1

    if args.chunk_size <= 0:
        print("ERROR: chunk size must be positive.")
        return 1

    print("=" * 72)
    print("BUSINESS ENTITY RESOLUTION DATASET PROFILER")
    print("=" * 72)

    print(f"Data directory : {data_dir}")
    print(f"Chunk size     : {args.chunk_size:,}")
    print(f"Output report  : {args.output.resolve()}")

    print("")
    print(
        "IMPORTANT: Raw business names and addresses will NOT "
        "be printed."
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    files = discover_files(data_dir)

    if not files:
        print("")
        print(
            "ERROR: No expected competition TSV files were found."
        )
        print("")
        print("Expected filenames include:")
        for name in [
            "train_source1.tsv",
            "train_source2.tsv",
            "train_source3.tsv",
            "train_ground_truth.tsv",
            "test_source1.tsv",
            "test_source2.tsv",
            "test_source3.tsv",
        ]:
            print(f"  {name}")

        return 1

    print("")
    print("Discovered files:")

    for filename, path in files.items():
        size_mb = path.stat().st_size / (1024 * 1024)

        print(
            f"  {filename}: "
            f"{size_mb:,.1f} MB"
        )

    # --------------------------------------------------------
    # Temporary profiling directory
    # --------------------------------------------------------

    tracker_dir = data_dir / ".profile_cache"
    tracker_dir.mkdir(exist_ok=True)

    file_stats: dict[str, FileStats] = {}

    report_parts = []

    # --------------------------------------------------------
    # Profile normal source files
    # --------------------------------------------------------

    for filename, path in files.items():

        if filename == "train_ground_truth.tsv":
            continue

        stats = profile_source_file(
            path=path,
            chunk_size=args.chunk_size,
            tracker_dir=tracker_dir,
        )

        file_stats[filename] = stats

        report_parts.append(stats.report())



    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = files.get(
        "train_ground_truth.tsv"
    )

    if ground_truth:

        ground_truth_report = profile_ground_truth(
            ground_truth,
            args.chunk_size,
        )

        report_parts.append(
            ground_truth_report
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    report_parts.insert(
        0,
        generate_overall_summary(file_stats),
    )

    report_parts.append(
        generate_initial_observations(file_stats)
    )

    report = "\n".join(report_parts)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        report,
        encoding="utf-8",
    )

    # Close duplicate-tracking databases only after all
    # reports and observations have finished using them.
    for stats in file_stats.values():
        if stats.name_tracker:
            stats.name_tracker.close()
        if stats.address_tracker:
            stats.address_tracker.close()

    print("")
    print("=" * 72)
    print("PROFILE COMPLETE")
    print("=" * 72)

    print(
        f"Report saved to:\n"
        f"  {args.output.resolve()}"
    )

    print("")
    print(
        "You can now inspect the report without sharing "
        "the 2GB dataset."
    )

    # --------------------------------------------------------
    # Cleanup instructions
    # --------------------------------------------------------

    print("")
    print(
        f"Temporary duplicate-tracking files are stored in:\n"
        f"  {tracker_dir}"
    )

    print(
        "You may delete that directory after profiling."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())