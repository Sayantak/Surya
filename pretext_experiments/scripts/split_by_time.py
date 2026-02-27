#!/usr/bin/env python3
"""
Split a Surya index CSV into train/val/test splits using time-based blocks.

Why time-based splitting?
- This is time-series solar data; random row splits can leak temporal structure.
- A contiguous (or block) split is the safest default.

Inputs:
- An index CSV produced by `make_index.py` with columns:
    path,present,timestep

Outputs (by default):
- pretext_experiments/outputs/index/train_index.csv
- pretext_experiments/outputs/index/val_index.csv
- pretext_experiments/outputs/index/test_index.csv

Default behavior:
- Uses the full index at:
    pretext_experiments/outputs/index/full_index.csv
- Performs a contiguous split (no gaps):
    train = first 80%
    val   = next 10%
    test  = last 10%

You can also:
- Split by day blocks (e.g., hold out last N days for test)
- Leave a temporal gap between splits (recommended if you want extra caution)

Usage examples:

  # Default contiguous split
  python pretext_experiments/scripts/split_by_time.py

  # Specify split fractions
  python pretext_experiments/scripts/split_by_time.py --train-frac 0.75 --val-frac 0.15 --test-frac 0.10

  # Hold out last 7 days for test and last 3 days before that for val
  python pretext_experiments/scripts/split_by_time.py --test-days 7 --val-days 3

  # Add a gap of 12 hours between train->val and val->test
  python pretext_experiments/scripts/split_by_time.py --gap-hours 12
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitPaths:
    train: Path
    val: Path
    test: Path


def _read_index(index_csv: Path) -> pd.DataFrame:
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")
    df = pd.read_csv(index_csv)
    required = {"path", "present", "timestep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Index CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestep"] = pd.to_datetime(df["timestep"], errors="raise")
    df = df.sort_values("timestep").reset_index(drop=True)

    # Keep only present==1 by default; Surya dataset expects/uses this.
    # If you want to preserve absent rows, remove this filter.
    df = df[df["present"] == 1].reset_index(drop=True)

    if df.empty:
        raise ValueError("Index CSV contains no rows with present==1.")
    return df


def _write_split(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep consistent timestep serialization
    out = df.copy()
    out["timestep"] = out["timestep"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(out_path, index=False)


def _apply_gap(df: pd.DataFrame, boundary_time: pd.Timestamp, gap: pd.Timedelta, side: str) -> pd.DataFrame:
    """
    side="left": keep rows strictly < (boundary_time - gap)
    side="right": keep rows strictly > (boundary_time + gap)
    """
    if gap <= pd.Timedelta(0):
        return df
    if side == "left":
        return df[df["timestep"] < (boundary_time - gap)]
    if side == "right":
        return df[df["timestep"] > (boundary_time + gap)]
    raise ValueError(f"Invalid side: {side}")


def contiguous_fraction_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    gap: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac <= 0 or val_frac < 0 or test_frac < 0:
        raise ValueError("Fractions must be non-negative and train_frac must be > 0.")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0 (got {total}).")

    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Ensure all rows accounted for
    n_train = max(1, min(n_train, n - 2))  # leave room for val/test
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Computed test split size <= 0. Adjust fractions.")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    # Apply gaps at boundaries (drop rows near boundaries to avoid leakage)
    if gap > pd.Timedelta(0):
        train_end = train_df["timestep"].max()
        val_start = val_df["timestep"].min()
        val_end = val_df["timestep"].max()
        test_start = test_df["timestep"].min()

        # Trim train and val around train/val boundary
        train_df = _apply_gap(train_df, val_start, gap, side="left")
        val_df = _apply_gap(val_df, train_end, gap, side="right")

        # Recompute boundary times after trimming (optional but safer)
        if not val_df.empty and not test_df.empty:
            val_end = val_df["timestep"].max()
            test_start = test_df["timestep"].min()
            # Trim val and test around val/test boundary
            val_df = _apply_gap(val_df, test_start, gap, side="left")
            test_df = _apply_gap(test_df, val_end, gap, side="right")

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "One or more splits became empty after applying gaps. "
            "Reduce --gap-hours or adjust split sizes."
        )

    return train_df, val_df, test_df


def day_holdout_split(
    df: pd.DataFrame,
    *,
    test_days: int,
    val_days: int,
    gap: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hold out the last `test_days` (by calendar day) for test, and the `val_days`
    immediately before for validation. The remaining earlier days are training.

    This is often a nice way to avoid subtle leakage and keep splits interpretable.
    """
    if test_days <= 0:
        raise ValueError("--test-days must be > 0 for day-holdout split.")
    if val_days < 0:
        raise ValueError("--val-days must be >= 0 for day-holdout split.")

    df = df.copy()
    df["day"] = df["timestep"].dt.floor("D")
    days = sorted(df["day"].unique())
    if len(days) < (test_days + max(val_days, 1) + 1):
        raise ValueError(
            f"Not enough unique days ({len(days)}) for test_days={test_days}, val_days={val_days}."
        )

    test_day_set = set(days[-test_days:])
    remaining_days = days[:-test_days]

    if val_days == 0:
        val_day_set = set()
        train_day_set = set(remaining_days)
    else:
        val_day_set = set(remaining_days[-val_days:])
        train_day_set = set(remaining_days[:-val_days])

    train_df = df[df["day"].isin(train_day_set)].copy()
    val_df = df[df["day"].isin(val_day_set)].copy() if val_days > 0 else pd.DataFrame(columns=df.columns)
    test_df = df[df["day"].isin(test_day_set)].copy()

    # Apply gaps at the boundaries of day ranges (optional)
    if gap > pd.Timedelta(0):
        # boundary between train and val (if val exists)
        if not val_df.empty:
            val_start = val_df["timestep"].min()
            train_df = _apply_gap(train_df, val_start, gap, side="left")
            val_df = _apply_gap(val_df, train_df["timestep"].max(), gap, side="right")

        # boundary between val and test
        test_start = test_df["timestep"].min()
        if not val_df.empty:
            val_df = _apply_gap(val_df, test_start, gap, side="left")
            test_df = _apply_gap(test_df, val_df["timestep"].max(), gap, side="right")
        else:
            # boundary between train and test if val doesn't exist
            train_df = _apply_gap(train_df, test_start, gap, side="left")
            test_df = _apply_gap(test_df, train_df["timestep"].max(), gap, side="right")

    # If val_days==0, we still want a val split (for training sanity).
    if val_days == 0:
        # Create a small val split from the end of train by time (5% of train, min 1 day)
        train_df = train_df.sort_values("timestep").reset_index(drop=True)
        n_train = len(train_df)
        n_val = max(1, int(round(n_train * 0.05)))
        if n_train - n_val <= 0:
            raise ValueError("Train split too small to carve out a validation split.")
        val_df = train_df.iloc[-n_val:].copy()
        train_df = train_df.iloc[:-n_val].copy()

    # Final checks
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "One or more splits are empty. "
            "Reduce --gap-hours or choose smaller --test-days/--val-days."
        )

    # Drop helper column
    train_df = train_df.drop(columns=["day"], errors="ignore")
    val_df = val_df.drop(columns=["day"], errors="ignore")
    test_df = test_df.drop(columns=["day"], errors="ignore")

    return train_df, val_df, test_df


def _summarize(name: str, df: pd.DataFrame) -> str:
    start = df["timestep"].min()
    end = df["timestep"].max()
    return f"{name}: rows={len(df):,}, range={start} -> {end}"


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

    # Fraction-based split (default)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)

    # Day-based holdout split (optional)
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

    index_csv = Path(args.index_csv)
    out_dir = Path(args.out_dir)
    out_paths = SplitPaths(
        train=out_dir / args.train_name,
        val=out_dir / args.val_name,
        test=out_dir / args.test_name,
    )

    df = _read_index(index_csv)
    gap = pd.Timedelta(hours=float(args.gap_hours))

    if args.test_days > 0:
        train_df, val_df, test_df = day_holdout_split(
            df,
            test_days=int(args.test_days),
            val_days=int(args.val_days),
            gap=gap,
        )
        split_kind = f"day-holdout (test_days={args.test_days}, val_days={args.val_days})"
    else:
        train_df, val_df, test_df = contiguous_fraction_split(
            df,
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            test_frac=float(args.test_frac),
            gap=gap,
        )
        split_kind = f"fraction (train={args.train_frac}, val={args.val_frac}, test={args.test_frac})"

    _write_split(train_df, out_paths.train)
    _write_split(val_df, out_paths.val)
    _write_split(test_df, out_paths.test)

    print(f"Split kind: {split_kind}")
    print(_summarize("TRAIN", train_df))
    print(_summarize("VAL", val_df))
    print(_summarize("TEST", test_df))
    print(f"Wrote:\n  {out_paths.train}\n  {out_paths.val}\n  {out_paths.test}")


if __name__ == "__main__":
    main()