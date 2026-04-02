"""
Split a Surya index CSV into train/val/test splits using time-based blocks.

This script is a CLI wrapper around the splitting utilities in util.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pretext_experiments.pretext.data.utils import (
    SplitParams,
    split_index_by_time,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-based split for Surya index CSV.")

    parser.add_argument(
        "--index-csv",
        type=str,
        default="../outputs/index/full_index.csv",
        help="Input index CSV (default: output of make_index.py).",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="../outputs/index",
        help="Output directory for train/val/test index CSVs.",
    )

    # Fraction split (default)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)

    # Day-holdout split
    parser.add_argument(
        "--test-days",
        type=int,
        default=0,
        help="If >0, use day-holdout split with this many last days as test.",
    )

    parser.add_argument(
        "--val-days",
        type=int,
        default=0,
        help="Used with --test-days. Number of days immediately before test to use as val.",
    )

    parser.add_argument(
        "--gap-hours",
        type=float,
        default=0.0,
        help="Optional temporal gap (in hours) to drop around split boundaries.",
    )

    parser.add_argument(
        "--train-name",
        type=str,
        default="train_index.csv",
        help="Output filename for train split.",
    )

    parser.add_argument(
        "--val-name",
        type=str,
        default="val_index.csv",
        help="Output filename for val split.",
    )

    parser.add_argument(
        "--test-name",
        type=str,
        default="test_index.csv",
        help="Output filename for test split.",
    )

    args = parser.parse_args()

    params = SplitParams(
        index_csv=Path(args.index_csv),
        out_dir=Path(args.out_dir),
        train_name=args.train_name,
        val_name=args.val_name,
        test_name=args.test_name,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        test_days=int(args.test_days),
        val_days=int(args.val_days),
        gap_hours=float(args.gap_hours),
    )

    paths = split_index_by_time(params)

    print("Split complete.")
    print(f"Train index: {paths.train}")
    print(f"Val index:   {paths.val}")
    print(f"Test index:  {paths.test}")


if __name__ == "__main__":
    main()