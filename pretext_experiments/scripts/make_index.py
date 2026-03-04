#!/usr/bin/env python3
"""
Build a Surya-compatible index CSV from locally available NetCDF (*.nc) files.

This script is a thin CLI wrapper around the indexing utilities in util.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pretext_experiments.pretext.data.utils import (
    IndexParams,
    build_index_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Surya HelioNetCDFDataset index CSV from local files."
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../../data/core-sdo/2024/12",
        help=(
            "Local root directory containing downloaded .nc files (recursively). "
            "Example: ../../data/core-sdo or ../../data/core-sdo/2024/12"
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
            "Write file paths relative to --dataset-path. "
            "Recommended if you will pass sdo_data_root_path=<dataset-path> "
            "to HelioNetCDFDataset."
        ),
    )

    parser.add_argument(
        "--no-skip-unparseable",
        action="store_true",
        help="If set, raise an error on filenames that don't match YYYYMMDD_HHMM.",
    )

    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="If set, do not sort by timestep.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)
    output_csv = Path(args.output)

    params = IndexParams(
        dataset_root=dataset_root,
        output_csv=output_csv,
        relative_paths=bool(args.relative_paths),
        skip_unparseable=not bool(args.no_skip_unparseable),
        sort_by_time=not bool(args.no_sort),
    )

    out_path = build_index_csv(params)

    print(f"Wrote index CSV: {out_path}")
    print(f"Dataset root: {dataset_root}")


if __name__ == "__main__":
    main()