from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import tarfile

logger = logging.getLogger(__name__)

SURYA_CORE_SDO_REPO_ID = "nasa-ibm-ai4science/core-sdo"
SURYA_VALIDATION_DATA_REPO_ID = "nasa-ibm-ai4science/Surya-1.0_validation_data"
SURYA_BASE_MODEL_REPO_ID = "nasa-ibm-ai4science/Surya-1.0"
SURYA_AR_SEG_REPO_ID = "nasa-ibm-ai4science/surya-bench-ar-segmentation"
SURYA_AR_SEG_MODEL_REPO_ID = "nasa-ibm-ai4science/ar_segmentation_surya"


@dataclass(frozen=True)
class SnapshotSpec:
    repo_id: str
    repo_type: str  # "dataset" or "model"
    local_dir: Path
    allow_patterns: Sequence[str] | None = None
    ignore_patterns: Sequence[str] | None = None
    revision: str | None = None
    force_download: bool = False


def _require_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e


def _snapshot_download(spec: SnapshotSpec) -> Path:
    """
    Download a Hugging Face repo snapshot into `spec.local_dir`.

    Behavior:
    - If `force_download=False`, `allow_patterns=None`, and `local_dir` already contains files,
      this function returns the existing directory immediately.
    - Otherwise, it calls HF `snapshot_download`, which is incremental and will only fetch
      what is missing for the requested allow_patterns/revision.
    """
    _require_huggingface_hub()
    from huggingface_hub import snapshot_download

    local_dir = spec.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    can_short_circuit = (
        not spec.force_download
        and spec.allow_patterns is None
        and any(local_dir.iterdir())
    )
    if can_short_circuit:
        logger.info("Using existing local_dir=%s for repo_id=%s", local_dir, spec.repo_id)
        return local_dir

    logger.info(
        "Downloading repo_id=%s (type=%s) into local_dir=%s | allow_patterns=%s",
        spec.repo_id,
        spec.repo_type,
        local_dir,
        list(spec.allow_patterns) if spec.allow_patterns else None,
    )

    snapshot_download(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
        ignore_patterns=list(spec.ignore_patterns) if spec.ignore_patterns else None,
        revision=spec.revision,
    )
    return local_dir


def ensure_validation_data(
    *,
    local_dir: str | Path = "data/Surya-1.0_validation_data",
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    revision: str | None = None,
) -> Path:
    spec = SnapshotSpec(
        repo_id=SURYA_VALIDATION_DATA_REPO_ID,
        repo_type="dataset",
        local_dir=Path(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        revision=revision,
    )
    return _snapshot_download(spec)


def ensure_core_sdo_data(
    *,
    local_dir: str | Path = "data/core-sdo",
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    revision: str | None = None,
) -> Path:
    spec = SnapshotSpec(
        repo_id=SURYA_CORE_SDO_REPO_ID,
        repo_type="dataset",
        local_dir=Path(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        revision=revision,
    )
    return _snapshot_download(spec)


def ensure_surya_base_model(
    *,
    local_dir: str | Path = "models/Surya-1.0",
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    revision: str | None = None,
) -> Path:
    spec = SnapshotSpec(
        repo_id=SURYA_BASE_MODEL_REPO_ID,
        repo_type="model",
        local_dir=Path(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        revision=revision,
    )
    return _snapshot_download(spec)

def ensure_ar_segmentation_surya_model(
    *,
    local_dir: str | Path = "pretext_experiments/data/ar_segmentation_surya",
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    revision: str | None = None,
) -> Path:
    spec = SnapshotSpec(
        repo_id=SURYA_AR_SEG_MODEL_REPO_ID,
        repo_type="model",
        local_dir=Path(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        revision=revision,
    )
    return _snapshot_download(spec)


def ar_seg_allow_patterns_for_date(date_yyyymmdd: str) -> list[str]:
    """
    Download:
    - all split CSVs
    - any .h5 files whose path contains the requested date

    The AR repo layout may vary, so keep the mask patterns broad.
    """
    if len(date_yyyymmdd) != 8 or not date_yyyymmdd.isdigit():
        raise ValueError(f"Expected YYYYMMDD, got: {date_yyyymmdd}")

    yyyy = date_yyyymmdd[:4]
    mm = date_yyyymmdd[4:6]

    return [
        "*.csv",
        f"**/{date_yyyymmdd}*.h5",
        f"**/{yyyy}/{mm}/{date_yyyymmdd}*.h5",
    ]

def ensure_ar_segmentation_data(
    *,
    local_dir: str | Path = "assets/surya-bench-ar-segmentation",
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
    revision: str | None = None,
    restrict_date: str | None = None,
) -> Path:
    """
    Ensure the SuryaBench AR segmentation dataset is present locally.

    For robustness, this downloads the full dataset snapshot: Avoids
    missing-mask issues from pattern mismatches.
    """
    if allow_patterns is not None:
        raise ValueError("Custom allow_patterns are not supported for AR segmentation data here.")

    if restrict_date is not None:
        logger.info(
            "restrict_date=%s was requested, but AR mask download is performed as a full snapshot "
            "for robustness. Date filtering will happen later in run_ar_seg.py.",
            restrict_date,
        )

    spec = SnapshotSpec(
        repo_id=SURYA_AR_SEG_REPO_ID,
        repo_type="dataset",
        local_dir=Path(local_dir),
        allow_patterns=None,
        ignore_patterns=ignore_patterns,
        revision=revision,
        force_download=False,
    )    

    out_dir = _snapshot_download(spec)

    archive_path = Path(out_dir) / "data.tar.gz"
    sample_expected = Path(out_dir) / "data"

    if archive_path.exists() and not sample_expected.exists():
        logger.info("Extracting AR masks from %s", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=out_dir)

    return out_dir


def list_netcdf_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    return sorted(root_path.rglob("*.nc"))


def iter_netcdf_files(root: str | Path) -> Iterable[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    yield from sorted(root_path.rglob("*.nc"))