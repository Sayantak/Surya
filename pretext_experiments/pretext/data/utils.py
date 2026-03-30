# pretext_experiments/pretext/data/util.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from pretext_experiments.pretext.data.s3_download import (
    DEFAULT_BUCKET,
    S3DownloadSpec,
    ensure_sdo_day_downloaded,
    verify_day_present,
)

# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadParams:
    """
    Parameters for downloading a subset of SuryaBench/Core-SDO from public S3.

    - date: YYYYMMDD (e.g. 20241231)
    - include_globs: optional filename globs (e.g. ["20241231_10*.nc", "20241231_11*.nc"]).
      If include_globs is non-empty, the downloader must treat it as restrictive includes.
    """
    date: str
    bucket: str = DEFAULT_BUCKET
    local_root: Path = Path("data/core-sdo")
    mirror_year_month_dirs: bool = True
    include_globs: Tuple[str, ...] = ()


def download_sdo_subset(params: DownloadParams) -> Path:
    """
    Download SDO NetCDF files for a day (optionally restricted by include patterns).

    Returns the local directory where files were downloaded (typically <local_root>/<YYYY>/<MM>/).
    """
    spec = S3DownloadSpec(
        bucket=params.bucket,
        date=params.date,
        local_root=params.local_root,
        mirror_year_month_dirs=params.mirror_year_month_dirs,
        extra_includes=tuple(params.include_globs),
    )
    out_dir = ensure_sdo_day_downloaded(spec)
    verify_day_present(out_dir, params.date)
    return out_dir


# ---------------------------------------------------------------------
# Indexing (local *.nc -> index CSV)
# ---------------------------------------------------------------------

_TS_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{4})")


@dataclass(frozen=True)
class IndexParams:
    dataset_root: Path
    output_csv: Path
    relative_paths: bool = True
    skip_unparseable: bool = True
    sort_by_time: bool = True


def _parse_timestep_from_filename(nc_path: Path) -> Optional[pd.Timestamp]:
    """
    Parse timestamp from filename containing YYYYMMDD_HHMM.

    Example: 20241231_1736.nc -> 2024-12-31 17:36:00
    """
    m = _TS_RE.search(nc_path.name)
    if not m:
        return None
    ymd = m.group("date")
    hm = m.group("time")
    ts_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]} {hm[0:2]}:{hm[2:4]}:00"
    try:
        return pd.to_datetime(ts_str, utc=False)
    except Exception:
        return None


def build_index_df(
    *,
    dataset_root: Path,
    relative_paths: bool,
    skip_unparseable: bool,
    sort_by_time: bool,
) -> pd.DataFrame:
    """
    Build DataFrame with columns: path, present, timestep
    from all *.nc files under dataset_root (recursive).
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    nc_files = sorted(dataset_root.rglob("*.nc"))
    n_total = len(nc_files)
    n_skipped = 0

    rows = []
    for p in nc_files:
        ts = _parse_timestep_from_filename(p)
        if ts is None:
            n_skipped += 1
            if skip_unparseable:
                continue
            raise ValueError(
                f"Could not parse timestamp from filename: {p.name}. "
                "Expected pattern like YYYYMMDD_HHMM somewhere in the name."
            )

        if relative_paths:
            try:
                path_str = str(p.relative_to(dataset_root))
            except ValueError:
                path_str = str(p)
        else:
            path_str = str(p)

        rows.append(
            {
                "path": path_str,
                "present": 1,
                "timestep": ts.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    if not rows:
        raise RuntimeError(
            f"No index rows produced. Found {n_total} *.nc files, skipped {n_skipped}. "
            "Check dataset_root and filename timestamp format."
        )

    df = pd.DataFrame(rows)

    if sort_by_time:
        df["timestep"] = pd.to_datetime(df["timestep"])
        df = df.sort_values("timestep").reset_index(drop=True)
        df["timestep"] = df["timestep"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def build_index_csv(params: IndexParams) -> Path:
    """
    Build index CSV and write to disk. Returns output path.
    """
    params.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = build_index_df(
        dataset_root=params.dataset_root,
        relative_paths=params.relative_paths,
        skip_unparseable=params.skip_unparseable,
        sort_by_time=params.sort_by_time,
    )
    df.to_csv(params.output_csv, index=False)
    return params.output_csv


# ---------------------------------------------------------------------
# Splitting (time-based)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPaths:
    train: Path
    val: Path
    test: Path


@dataclass(frozen=True)
class SplitParams:
    index_csv: Path
    out_dir: Path
    train_name: str = "train_index.csv"
    val_name: str = "val_index.csv"
    test_name: str = "test_index.csv"

    # fraction mode
    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10

    # day-holdout mode (if test_days > 0)
    test_days: int = 0
    val_days: int = 0

    # optional temporal gap
    gap_hours: float = 0.0


def _read_index_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Index CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"path", "present", "timestep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Index CSV missing columns {sorted(missing)}: {path}")
    df["timestep"] = pd.to_datetime(df["timestep"])
    df = df.sort_values("timestep").reset_index(drop=True)
    return df


def _write_index_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestep"] = pd.to_datetime(out["timestep"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def _apply_gap(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    Keep rows with timestep in [start, end) if bounds provided.
    """
    out = df
    if start is not None:
        out = out[out["timestep"] >= start]
    if end is not None:
        out = out[out["timestep"] < end]
    return out


def contiguous_fraction_split(
    df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    gap: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split sorted df into contiguous blocks by fraction, with an optional gap around cut points.
    """
    if train_frac <= 0 or val_frac < 0 or test_frac < 0:
        raise ValueError("Fractions must be non-negative and train_frac > 0.")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}.")

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # remainder goes to test
    n_test = n - n_train - n_val

    train_end_idx = n_train
    val_end_idx = n_train + n_val

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    if gap > pd.Timedelta(0) and len(df) > 0:
        # Remove windows around the split boundaries
        t_train_end = train_df["timestep"].max() if len(train_df) else None
        t_val_start = val_df["timestep"].min() if len(val_df) else None
        t_val_end = val_df["timestep"].max() if len(val_df) else None
        t_test_start = test_df["timestep"].min() if len(test_df) else None

        if t_train_end is not None and t_val_start is not None:
            # drop last 'gap' from train and first 'gap' from val
            train_df = train_df[train_df["timestep"] < (t_val_start - gap)]
            val_df = val_df[val_df["timestep"] >= (t_val_start + gap)]

        if t_val_end is not None and t_test_start is not None:
            # drop last 'gap' from val and first 'gap' from test
            val_df = val_df[val_df["timestep"] < (t_test_start - gap)]
            test_df = test_df[test_df["timestep"] >= (t_test_start + gap)]

    # safety: ensure sorted
    return (
        train_df.sort_values("timestep").reset_index(drop=True),
        val_df.sort_values("timestep").reset_index(drop=True),
        test_df.sort_values("timestep").reset_index(drop=True),
    )


def day_holdout_split(
    df: pd.DataFrame,
    *,
    test_days: int,
    val_days: int,
    gap: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hold out the last `test_days` days as test, and (optionally) the preceding `val_days` days as val.
    Everything earlier is train. Uses calendar day boundaries based on timestep.
    """
    if test_days <= 0:
        raise ValueError("test_days must be > 0 for day_holdout_split.")
    if val_days < 0:
        raise ValueError("val_days must be >= 0.")

    # Normalize to dates
    df = df.sort_values("timestep").reset_index(drop=True)
    df["date"] = df["timestep"].dt.floor("D")

    unique_days = df["date"].drop_duplicates().sort_values().to_list()
    if len(unique_days) < test_days + val_days + 1:
        raise ValueError(
            f"Not enough unique days ({len(unique_days)}) for test_days={test_days} and val_days={val_days}."
        )

    test_start_day = unique_days[-test_days]
    val_start_day = unique_days[-(test_days + val_days)] if val_days > 0 else None

    test_mask = df["date"] >= test_start_day
    if val_days > 0 and val_start_day is not None:
        val_mask = (df["date"] >= val_start_day) & (df["date"] < test_start_day)
    else:
        val_mask = pd.Series([False] * len(df))

    train_mask = ~(test_mask | val_mask)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Apply temporal gap at boundaries (in timestamp space)
    if gap > pd.Timedelta(0):
        # gap around val/test boundary
        if len(val_df) and len(test_df):
            boundary = test_df["timestep"].min()
            val_df = val_df[val_df["timestep"] < (boundary - gap)]
            test_df = test_df[test_df["timestep"] >= (boundary + gap)]

        # gap around train/val boundary
        if len(train_df) and len(val_df):
            boundary = val_df["timestep"].min()
            train_df = train_df[train_df["timestep"] < (boundary - gap)]
            val_df = val_df[val_df["timestep"] >= (boundary + gap)]

    # Cleanup helper column
    for d in (train_df, val_df, test_df):
        if "date" in d.columns:
            d.drop(columns=["date"], inplace=True)

    return (
        train_df.sort_values("timestep").reset_index(drop=True),
        val_df.sort_values("timestep").reset_index(drop=True),
        test_df.sort_values("timestep").reset_index(drop=True),
    )


def split_index_by_time(params: SplitParams) -> SplitPaths:
    """
    Split an index CSV by time and write train/val/test CSVs.
    """
    df = _read_index_csv(params.index_csv)
    gap = pd.Timedelta(hours=float(params.gap_hours))

    if params.test_days > 0:
        train_df, val_df, test_df = day_holdout_split(
            df,
            test_days=int(params.test_days),
            val_days=int(params.val_days),
            gap=gap,
        )
    else:
        train_df, val_df, test_df = contiguous_fraction_split(
            df,
            train_frac=float(params.train_frac),
            val_frac=float(params.val_frac),
            test_frac=float(params.test_frac),
            gap=gap,
        )

    out_dir = params.out_dir
    out_paths = SplitPaths(
        train=out_dir / params.train_name,
        val=out_dir / params.val_name,
        test=out_dir / params.test_name,
    )

    _write_index_csv(train_df, out_paths.train)
    _write_index_csv(val_df, out_paths.val)
    _write_index_csv(test_df, out_paths.test)

    return out_paths


# ---------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedData:
    dataset_root: Path
    full_index_csv: Path
    train_index_csv: Path
    val_index_csv: Path
    test_index_csv: Path


def prepare_sdo_data_for_run(
    *,
    download: DownloadParams,
    full_index_csv: Path,
    split_params: SplitParams,
    index_relative_paths: bool = True,
    index_skip_unparseable: bool = True,
    index_sort_by_time: bool = True,
) -> PreparedData:
    """
    End-to-end:
      1) download subset
      2) build full index
      3) split index by time

    Returns paths to everything.
    """
    dataset_root = download_sdo_subset(download)

    full_index_csv = build_index_csv(
        IndexParams(
            dataset_root=dataset_root,
            output_csv=full_index_csv,
            relative_paths=index_relative_paths,
            skip_unparseable=index_skip_unparseable,
            sort_by_time=index_sort_by_time,
        )
    )

    # Ensure split uses the index we just produced
    split_params = SplitParams(
        index_csv=full_index_csv,
        out_dir=split_params.out_dir,
        train_name=split_params.train_name,
        val_name=split_params.val_name,
        test_name=split_params.test_name,
        train_frac=split_params.train_frac,
        val_frac=split_params.val_frac,
        test_frac=split_params.test_frac,
        test_days=split_params.test_days,
        val_days=split_params.val_days,
        gap_hours=split_params.gap_hours,
    )

    split_paths = split_index_by_time(split_params)

    return PreparedData(
        dataset_root=dataset_root,
        full_index_csv=full_index_csv,
        train_index_csv=split_paths.train,
        val_index_csv=split_paths.val,
        test_index_csv=split_paths.test,
    )