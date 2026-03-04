#!/usr/bin/env python3
# pretext_experiments/scripts/download_sdo_s3.py

from __future__ import annotations

import argparse
from pathlib import Path

from pretext_experiments.pretext.data.utils import (
    DownloadParams,
    download_sdo_subset,
)
from pretext_experiments.pretext.data.s3_download import DEFAULT_BUCKET


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download SDO Core dataset files from the public SuryaBench S3 bucket "
            "(no AWS account required). Uses util.py helper functions."
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to download in YYYYMMDD format (e.g., 20241231).",
    )

    parser.add_argument(
        "--bucket",
        type=str,
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET}).",
    )

    parser.add_argument(
        "--local-root",
        type=str,
        default="../../data/core-sdo",
        help="Local root directory for downloaded files.",
    )

    parser.add_argument(
        "--no-mirror-year-month-dirs",
        action="store_true",
        help="If set, files are downloaded directly into --local-root (no YYYY/MM subfolders).",
    )

    parser.add_argument(
        "--hour-prefix",
        type=str,
        default="",
        help=(
            "Optional hour prefix to restrict downloads (00-23). "
            'Example: --hour-prefix 10 downloads "YYYYMMDD_10*.nc".'
        ),
    )

    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "Additional filename glob(s) for filtering. "
            'Example: --include "20241231_12*.nc". Can be used multiple times.'
        ),
    )

    args = parser.parse_args()

    date = args.date.strip()
    bucket = args.bucket.strip()
    local_root = Path(args.local_root)

    include_globs: list[str] = []

    # Handle hour restriction
    if args.hour_prefix:
        hp = args.hour_prefix.strip()
        if len(hp) != 2 or not hp.isdigit() or not (0 <= int(hp) <= 23):
            raise ValueError(f"--hour-prefix must be between 00 and 23, got: {args.hour_prefix}")
        include_globs.append(f"{date}_{hp}*.nc")

    # Additional include patterns
    if args.include:
        include_globs.extend([s.strip() for s in args.include if s.strip()])

    params = DownloadParams(
        date=date,
        bucket=bucket,
        local_root=local_root,
        mirror_year_month_dirs=not args.no_mirror_year_month_dirs,
        include_globs=tuple(include_globs),
    )

    out_dir = download_sdo_subset(params)

    print(f"Downloaded data for {date} into: {out_dir}")


if __name__ == "__main__":
    main()