from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_BUCKET = "nasa-surya-bench"


@dataclass(frozen=True)
class S3DownloadSpec:
    """
    Download spec for SuryaBench/Core-SDO data from the public S3 bucket.

    Bucket is public; we always use --no-sign-request.
    """
    bucket: str = DEFAULT_BUCKET
    # Date in YYYYMMDD
    date: str = "20241231"
    # Root where data will be stored locally.
    local_root: Path = Path("data/core-sdo")
    # If True, store under <local_root>/<YYYY>/<MM>/ like the S3 prefix.
    mirror_year_month_dirs: bool = True
    # Optional additional include globs (besides "{date}_*.nc"), e.g. ["{date}_00*.nc"].
    extra_includes: tuple[str, ...] = ()


def _validate_yyyymmdd(date: str) -> None:
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"date must be YYYYMMDD, got: {date}")
    yyyy = int(date[:4])
    mm = int(date[4:6])
    dd = int(date[6:8])
    if not (2010 <= yyyy <= 2024):
        raise ValueError(f"Year out of expected range [2010, 2024]: {yyyy}")
    if not (1 <= mm <= 12):
        raise ValueError(f"Month out of range [1, 12]: {mm}")
    if not (1 <= dd <= 31):
        raise ValueError(f"Day out of range [1, 31]: {dd}")

def _find_aws_cli() -> str:
    """
    Locate AWS CLI executable on Windows/Linux.

    On Windows, prefer aws.exe/aws.cmd (CreateProcess cannot execute a bash script).
    """
    override = os.environ.get("AWS_CLI")
    if override:
        return override

    # Windows: prefer aws.exe or aws.cmd explicitly
    if sys.platform.startswith("win"):
        for name in ("aws.exe", "aws.cmd", "aws.bat", "aws"):
            p = shutil.which(name)
            if p:
                ext = Path(p).suffix.lower()
                if ext in (".exe", ".cmd", ".bat"):
                    return p
                if ext == "" and Path(p).exists():
                    return p

        raise FileNotFoundError(
            "AWS CLI not found as aws.exe/aws.cmd on PATH. "
            "Install AWS CLI v2 or ensure `aws.exe` is available, "
            "or set AWS_CLI to the full path of aws.exe."
        )

    # Linux/macOS
    p = shutil.which("aws")
    if p:
        return p

    raise FileNotFoundError(
        "AWS CLI not found on PATH. Install it and ensure `aws` is available, "
        "or set AWS_CLI to the executable path."
    )

def _run(cmd: list[str]) -> None:
    """
    Run a command and raise a detailed error on failure.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = "Command failed.\n"
        msg += f"Command: {format_cmd_bash(cmd)}\n"
        if stdout:
            msg += f"STDOUT:\n{stdout}\n"
        if stderr:
            msg += f"STDERR:\n{stderr}\n"
        raise RuntimeError(msg)

def _run_streaming(cmd: list[str]) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    total_downloaded = 0

    for line in proc.stdout:
        line = line.strip()
        if line.lower().startswith("download:"):
            total_downloaded += 1
            print(f"Downloaded file #{total_downloaded}")

    rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}")


def _make_s3_month_prefix(date: str) -> tuple[str, str, str]:
    yyyy = date[:4]
    mm = date[4:6]
    return yyyy, mm, f"{yyyy}/{mm}/"

def build_includes_for_day(date: str, extra_includes: Sequence[str] | None = None) -> list[str]:
    """
    Build include patterns for a given date.

    - If extra_includes is provided and non-empty, we assume the caller wants
      a restricted subset (e.g., a single hour), so we ONLY use extra_includes.
    - Otherwise, default to all files for that day: '{date}_*.nc'
    """
    if extra_includes:
        extras = [s for s in extra_includes if str(s).strip()]
        if extras:
            return list(extras)
    return [f"{date}_*.nc"]


def format_cmd_bash(cmd: Sequence[str]) -> str:
    """
    Format a subprocess command as a bash-friendly string.
    """
    return " ".join(shlex.quote(c) for c in cmd)


def format_cmd_powershell(cmd: Sequence[str]) -> str:
    """
    Format a subprocess command for PowerShell.
    PowerShell quoting rules are different; this is mainly for debugging display.
    """
    def ps_quote(s: str) -> str:
        if re.search(r"[ \t\"']", s):
            return '"' + s.replace('"', '`"') + '"'
        return s

    return " ".join(ps_quote(c) for c in cmd)


def ensure_sdo_day_downloaded(spec: S3DownloadSpec) -> Path:
    """
    Download all NetCDF files for a single day from the SuryaBench S3 bucket.

    Uses:
      aws s3 sync --no-sign-request s3://{bucket}/{YYYY}/{MM}/ <local_dir>
        --exclude "*" --include "{YYYYMMDD}_*.nc" [--include "<extra>"]...

    Returns:
      Path to local directory containing downloaded files:
        - if mirror_year_month_dirs=True: <local_root>/<YYYY>/<MM>/
        - else: <local_root>/
    """
    _validate_yyyymmdd(spec.date)
    aws = _find_aws_cli()

    yyyy, mm, month_prefix = _make_s3_month_prefix(spec.date)

    if spec.mirror_year_month_dirs:
        local_dir = spec.local_root / yyyy / mm
    else:
        local_dir = spec.local_root

    local_dir.mkdir(parents=True, exist_ok=True)

    s3_src = f"s3://{spec.bucket}/{month_prefix}"
    includes = build_includes_for_day(spec.date, list(spec.extra_includes))

    cmd = [
    aws,
    "s3",
    "sync",
    "--no-sign-request",
    "--no-progress",
    s3_src,
    str(local_dir),
    "--exclude",
    "*",
    ]
    for inc in includes:
        cmd.extend(["--include", inc])

    _run_streaming(cmd)
    return local_dir


def verify_day_present(local_dir: Path, date: str) -> None:
    """
    Verify at least one file matching '{date}_*.nc' exists under local_dir.
    """
    pat = re.compile(rf"{re.escape(date)}_\d{{4}}\.nc$")
    matches = [p for p in local_dir.rglob("*.nc") if pat.search(p.name)]
    if not matches:
        raise RuntimeError(
            f"No .nc files for {date} found under: {local_dir}. "
            "Check bucket access and date/prefix."
        )


def list_local_nc_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.nc"))