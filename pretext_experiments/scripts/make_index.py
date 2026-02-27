#!/usr/bin/env python3
"""
Build a Surya-compatible index CSV from the Surya 1.0 validation dataset NetCDF (*.nc) files.

Defaults:
- Downloads/uses the full HF dataset into the same path used by tests/test_surya.py:
    data/Surya-1.0_validation_data
- Creates the index at:
    pretext_experiments/outputs/index/full_index.csv

This produces a minimal index format expected by `surya.datasets.helio.HelioNetCDFDataset`:
    - path: str        (absolute OR relative to data root)
    - present: int     (1 if present)
    - timestep: str    (parseable datetime string)

Timestamp parsing:
- Attempts to parse timestamps from filenames containing a pattern like YYYYMMDD_HHMM
  anywhere in the filename (e.g., 20140107_1500.nc). Files that don't match are skipped
  by default (with a warning count).

Usage examples:

  # Default: download full dataset to data/Surya-1.0_validation_data and build index
  python pretext_experiments/scripts/make_index.py

  # Specify output path
  python pretext_experiments/scripts/make_index.py \
    --output pretext_experiments/outputs/index/full_index.csv

  # Create relative paths (recommended if you pass sdo_data_root_path later)
  python pretext_experiments/scripts/make_index.py --relative-paths

  # Download only a subset (for debugging)
  python pretext_experiments/scripts/make_index.py \
    --allow-pattern "20140107_*.nc" --allow-pattern "20140108_*.nc"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# This module is part of the code wrote in Phase 1.
# It mirrors how test_surya.py downloads the dataset, but defaults to full download.
from pretext_experiments.pretext.data.hf_validation import ensure_validation_data, iter_netcdf_files

_TS_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{4})")


@dataclass(frozen=True)
class IndexRow:
    path: str
    present: int
    timestep: str  # ISO-like string parseable by pandas


# def _iter_netcdf_files(data_dir: Path) -> Iterable[Path]:
#     if not data_dir.exists():
#         raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
#     yield from sorted(data_dir.rglob("*.nc"))


def _parse_timestep_from_name(nc_path: Path) -> pd.Timestamp | None:
    """
    Parse timestamp from filename by searching for YYYYMMDD_HHMM.
    Returns a pandas.Timestamp or None if no match.
    """
    m = _TS_RE.search(nc_path.name)
    if not m:
        return None
    ymd = m.group("date")  # YYYYMMDD
    hm = m.group("time")   # HHMM
    ts_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]} {hm[0:2]}:{hm[2:4]}:00"
    try:
        return pd.to_datetime(ts_str, utc=False)
    except Exception:
        return None


def build_index(
    *,
    data_dir: Path,
    relative_paths: bool,
    skip_unparseable: bool,
    sort_by_time: bool,
) -> pd.DataFrame:
    """
    Create a DataFrame with columns: path,present,timestep
    """
    rows: list[IndexRow] = []
    n_total = 0
    n_skipped = 0

    for nc_file in iter_netcdf_files(data_dir):
        n_total += 1
        ts = _parse_timestep_from_name(nc_file)
        if ts is None:
            n_skipped += 1
            if skip_unparseable:
                continue
            raise ValueError(
                f"Could not parse timestamp from filename: {nc_file.name}. "
                "Expected pattern like YYYYMMDD_HHMM somewhere in the name."
            )

        if relative_paths:
            path_str = str(nc_file.relative_to(data_dir))
        else:
            path_str = str(nc_file)

        rows.append(
            IndexRow(
                path=path_str,
                present=1,
                timestep=ts.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

    if not rows:
        raise RuntimeError(
            f"No index rows produced. Found {n_total} *.nc files, skipped {n_skipped}. "
            "Check that filenames contain timestamps like YYYYMMDD_HHMM."
        )

    df = pd.DataFrame([r.__dict__ for r in rows])

    if sort_by_time:
        df["timestep"] = pd.to_datetime(df["timestep"])
        df = df.sort_values("timestep").reset_index(drop=True)
        df["timestep"] = df["timestep"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Helpful metadata for debugging (not required by Surya, but harmless)
    # Uncomment if you want:
    # df["source"] = "Surya-1.0_validation_data"

    return df


def _ensure_dataset(
    local_dir: Path,
    allow_patterns: Sequence[str] | None,
    ignore_patterns: Sequence[str] | None,
    revision: str | None,
) -> Path:
    """
    Ensure HF dataset is present locally in the same default directory used by test_surya.py.
    """
    return ensure_validation_data(
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        revision=revision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Surya HelioNetCDFDataset index CSV.")

    # Match test_surya.py storage convention by default.
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../data/Surya-1.0_validation_data",
        help=(
            "Root directory for the downloaded HF validation dataset snapshot. "
            "Default matches tests/test_surya.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../outputs/index/full_index.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help=(
            "Write file paths relative to --data-dir. "
            "Recommended if you will pass sdo_data_root_path=DATA_DIR to HelioNetCDFDataset."
        ),
    )

    # By default we skip filenames that don't match YYYYMMDD_HHMM.
    parser.add_argument(
        "--no-skip-unparseable",
        action="store_true",
        help="If set, raise an error on any filename that doesn't match YYYYMMDD_HHMM.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="If set, do not sort by timestep (default: sort).",
    )

    # Optional: allow restricting which files are downloaded from HF.
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help=(
            "HF snapshot allow_patterns entry. Can be repeated. "
            "If omitted, downloads the full dataset snapshot."
        ),
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="HF snapshot ignore_patterns entry. Can be repeated.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision (commit hash or tag).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure data exists locally (auto-download unless already present).
    data_root = _ensure_dataset(
        local_dir=data_dir,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        revision=args.revision,
    )

    df = build_index(
        data_dir=data_root,
        relative_paths=bool(args.relative_paths),
        skip_unparseable=not bool(args.no_skip_unparseable),
        sort_by_time=not bool(args.no_sort),
    )

    df.to_csv(output_path, index=False)

    # Summary
    print(f"Wrote index CSV: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Time range: {df['timestep'].iloc[0]}  ->  {df['timestep'].iloc[-1]}")
    if args.relative_paths:
        print("Note: paths are relative. Use sdo_data_root_path=<data-dir> in HelioNetCDFDataset.")
    else:
        print("Note: paths are absolute. sdo_data_root_path is not required (but still allowed).")


if __name__ == "__main__":
    main()