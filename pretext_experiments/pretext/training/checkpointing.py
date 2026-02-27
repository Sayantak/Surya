# pretext_experiments/pretext/training/checkpointing.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from surya.utils.distributed import is_dist_avail_and_initialized, is_main_process, save_state_singular

def resolve_checkpoint_path(path_or_dir: str | Path) -> Path:
    """Resolve a checkpoint path from either a file path or a directory.

    - If given a directory, prefers 'surya.366m.v1.pt' if present, otherwise picks the
      largest '.pt' file.
    - If given a file, returns it.
    """
    p = Path(path_or_dir)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")

    if p.is_file():
        return p

    preferred = p / "surya.366m.v1.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(p.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in directory: {p}")

    return max(candidates, key=lambda x: x.stat().st_size)


def load_baseline_ckpt(
    path_or_dir: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[Path, int]:
    """Load a baseline Surya checkpoint into `model`.

    Supports:
      - our wrapper format: {"model_state": ..., "step": ...}
      - "raw state_dict" format (payload is itself the state_dict)

    Returns:
      (resolved_ckpt_path, step). step is 0 if not present.
    """
    ckpt_path = resolve_checkpoint_path(path_or_dir)
    payload: Any = load_checkpoint(ckpt_path, map_location=map_location)

    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        step = int(payload.get("step", 0))
    else:
        state = payload
        step = 0

    model.load_state_dict(state, strict=strict)
    return ckpt_path, step


@dataclass(frozen=True)
class CheckpointPayload:
    step: int
    model_state: dict[str, Any]
    optim_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    meta: dict[str, Any]


def create_run_dir(base_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(base_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    *,
    run_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    meta: dict[str, Any] | None = None,
    filename: str | None = None,
) -> Path:
    """
    Save checkpoint to <run_dir>/checkpoints/<filename or step_*.pt>.

    If distributed is initialized, uses Surya's save_state_singular (barrier-safe).
    Otherwise falls back to torch.save.
    """
    run_dir_path = Path(run_dir)
    ckpt_dir = run_dir_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = filename if filename is not None else f"step_{int(step)}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    payload = CheckpointPayload(
        step=int(step),
        model_state=model.state_dict(),
        optim_state=optimizer.state_dict() if optimizer is not None else None,
        scaler_state=scaler.state_dict() if scaler is not None else None,
        meta=meta or {},
    )

    obj = {
        "step": payload.step,
        "model_state": payload.model_state,
        "optim_state": payload.optim_state,
        "scaler_state": payload.scaler_state,
        "meta": payload.meta,
    }

    if is_dist_avail_and_initialized():
        # save_state_singular does barrier() for us
        save_state_singular(obj, str(ckpt_path))
    else:
        if is_main_process():
            torch.save(obj, ckpt_path)

    return ckpt_path


def load_checkpoint(
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def restore_checkpoint(
    ckpt_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> int:
    payload = load_checkpoint(ckpt_path, map_location=map_location)

    model.load_state_dict(payload["model_state"], strict=strict)

    if optimizer is not None and payload.get("optim_state") is not None:
        optimizer.load_state_dict(payload["optim_state"])

    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])

    return int(payload.get("step", 0))


def get_latest_checkpoint(run_dir: str | Path) -> Path | None:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("step_*.pt"))
    if not candidates:
        return None

    def step_num(p: Path) -> int:
        # step_<N>.pt
        stem = p.stem
        parts = stem.split("_")
        if len(parts) != 2:
            return -1
        try:
            return int(parts[1])
        except Exception:
            return -1

    candidates = sorted(candidates, key=step_num)
    return candidates[-1]