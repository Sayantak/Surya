from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

from surya.utils.log import create_logger as surya_create_logger
from surya.utils.log import log as surya_wandb_log
from surya.utils.distributed import get_rank, is_main_process


def setup_surya_logger(
    run_dir: str | Path,
    *,
    name: str = "pretext_main",
) -> logging.Logger:
    """
    Create a Surya-style logger writing to:
      <run_dir>/<name>.log
    """
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    dist_rank = int(get_rank())
    logger = surya_create_logger(str(run_dir_path), dist_rank=dist_rank, name=name)
    return logger


@dataclass
class JsonlLogger:
    """
    Simple JSONL file logger for metrics.
    Writes one JSON object per line.
    """
    path: Path
    only_main_process: bool = True

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        if self.only_main_process and not is_main_process():
            return
        payload = dict(record)
        payload.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_hparams(run_dir: str | Path, hparams: dict[str, Any], filename: str = "hparams.json") -> Path:
    """
    Write run hyperparameters/config to <run_dir>/<filename>.
    """
    run_dir_path = Path(run_dir)
    out_path = run_dir_path / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False, sort_keys=True)
    return out_path


def wandb_log(
    run: Any,
    data: dict[str, Any],
    *,
    step: int | None = None,
    commit: bool | None = None,
    sync: bool | None = None,
) -> None:
    """
    Thin wrapper over surya.utils.log.log.
    If `run is None`, Surya's helper prints the dict (handy during debugging).
    """
    surya_wandb_log(run, data, step=step, commit=commit, sync=sync)