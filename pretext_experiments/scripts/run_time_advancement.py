#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List

import torch
from torch.utils.data import DataLoader
import yaml

import pandas as pd
from torch.utils.data import Subset

from surya.utils.distributed import get_rank, set_global_seed
from surya.utils.data import build_scalers, custom_collate_fn
from surya.models.helio_spectformer import HelioSpectFormer
from surya.datasets.helio import HelioNetCDFDataset

from pretext_experiments.pretext.data.hf_validation import (
    ensure_surya_base_model,  # keep for model weights only
)
from pretext_experiments.pretext.data.utils import (
    DownloadParams,
    SplitParams,
    prepare_sdo_data_for_run,
)
from pretext_experiments.pretext.objectives.time_advancement import TimeAdvancementObjective
from pretext_experiments.pretext.training.checkpointing import create_run_dir
from pretext_experiments.pretext.training.logging import JsonlLogger, setup_surya_logger, write_hparams
from pretext_experiments.pretext.training.trainer import Trainer, TrainerConfig
from pretext_experiments.pretext.eval.surya_viz import visualize_batch_from_dataloader


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model_from_config(config: dict[str, Any]) -> HelioSpectFormer:
    model_cfg = config["model"]
    data_cfg = config["data"]

    model = HelioSpectFormer(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=len(data_cfg["sdo_channels"]),
        embed_dim=model_cfg["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": len(data_cfg["time_delta_input_minutes"]),
        },
        depth=model_cfg["depth"],
        n_spectral_blocks=model_cfg["n_spectral_blocks"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        drop_rate=model_cfg["drop_rate"],
        dtype=torch.bfloat16,
        window_size=model_cfg["window_size"],
        dp_rank=model_cfg["dp_rank"],
        learned_flow=model_cfg["learned_flow"],
        use_latitude_in_learned_flow=model_cfg["learned_flow"],
        init_weights=False,
        checkpoint_layers=list(range(model_cfg["depth"])),
        rpe=model_cfg["rpe"],
        ensemble=model_cfg["ensemble"],
        finetune=model_cfg["finetune"],
    )
    return model


def _load_weights_strict(model: torch.nn.Module, weights_path: Path, device: str) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"Baseline weights not found: {weights_path}")

    try:
        weights = torch.load(weights_path, map_location=torch.device(device), weights_only=True)
    except TypeError:
        weights = torch.load(weights_path, map_location=torch.device(device))

    model.load_state_dict(weights, strict=True)


def _build_train_dataloader(
    *,
    train_index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    rollout_steps: int,
) -> DataLoader:
    data_cfg = config["data"]

    ds = HelioNetCDFDataset(
        index_path=str(train_index_csv),
        time_delta_input_minutes=data_cfg["time_delta_input_minutes"],
        time_delta_target_minutes=data_cfg["time_delta_target_minutes"],
        n_input_timestamps=len(data_cfg["time_delta_input_minutes"]),
        rollout_steps=rollout_steps,
        channels=data_cfg["sdo_channels"],
        drop_hmi_probability=data_cfg["drop_hmi_probability"],
        num_mask_aia_channels=data_cfg["num_mask_aia_channels"],
        use_latitude_in_learned_flow=data_cfg["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="train",
        pooling=data_cfg.get("pooling", None),
        random_vert_flip=data_cfg.get("random_vert_flip", False),
        sdo_data_root_path=str(dataset_root),
    )
    n = len(ds)
    if n == 0:
        raise RuntimeError(
            "HelioNetCDFDataset has length 0. This usually means the downloaded subset "
            "doesn't contain enough temporal context for time_delta_input_minutes / targets / rollout_steps. "
            "Try downloading a larger contiguous time window (e.g., include earlier hours)."
        )

    return DataLoader(
        ds,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 2,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

def _build_dataset(
    *,
    index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    rollout_steps: int,
    phase: str = "valid",
) -> HelioNetCDFDataset:
    data_cfg = config["data"]

    ds = HelioNetCDFDataset(
        index_path=str(index_csv),
        time_delta_input_minutes=data_cfg["time_delta_input_minutes"],
        time_delta_target_minutes=data_cfg["time_delta_target_minutes"],
        n_input_timestamps=len(data_cfg["time_delta_input_minutes"]),
        rollout_steps=rollout_steps,
        channels=data_cfg["sdo_channels"],
        drop_hmi_probability=data_cfg["drop_hmi_probability"],
        num_mask_aia_channels=data_cfg["num_mask_aia_channels"],
        use_latitude_in_learned_flow=data_cfg["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase=phase,
        pooling=data_cfg.get("pooling", None),
        random_vert_flip=False,
        sdo_data_root_path=str(dataset_root),
    )

    if len(ds) == 0:
        raise RuntimeError(f"Visualization dataset has length 0 for index: {index_csv}")

    return ds

def _build_eval_dataloader(
    *,
    index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    num_workers: int,
    pin_memory: bool,
    rollout_steps: int,
    phase: str = "valid",
) -> DataLoader:
    ds = _build_dataset(
        index_csv=index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        rollout_steps=rollout_steps,
        phase=phase,
    )

    return DataLoader(
        ds,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 2,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

def _build_split_eval_dataloader(
    *,
    full_index_csv: Path,
    allowed_index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    num_workers: int,
    pin_memory: bool,
    rollout_steps: int,
    phase: str = "valid",
) -> DataLoader:
    """
    Build an evaluation dataloader using full_index.csv for temporal context,
    but only score samples whose reference timestep belongs to allowed_index_csv
    (e.g. val_index.csv or test_index.csv).
    """
    ds = _build_dataset(
        index_csv=full_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        rollout_steps=rollout_steps,
        phase=phase,
    )

    allowed_df = pd.read_csv(allowed_index_csv)
    if "timestep" not in allowed_df.columns:
        raise ValueError(f"Missing 'timestep' column in {allowed_index_csv}")
    allowed_times = set(pd.to_datetime(allowed_df["timestep"]))

    full_df = pd.read_csv(full_index_csv)
    full_times = pd.to_datetime(full_df["timestep"])

    if not hasattr(ds, "valid_indices"):
        raise AttributeError(
            "HelioNetCDFDataset does not expose 'valid_indices'. "
            "You need to inspect helio.py and adapt this helper to the dataset's internal field name."
        )

    # ds.valid_indices are the candidate row indices in full_index.csv that survived context filtering
    keep_positions = []
    for subset_pos, ts in enumerate(ds.valid_indices):
        ts = pd.to_datetime(ts)
        if ts in allowed_times:
            keep_positions.append(subset_pos)

    if len(keep_positions) == 0:
        raise RuntimeError(
            f"No valid evaluation samples remain after restricting {allowed_index_csv} "
            f"against context from {full_index_csv}."
        )

    subset = Subset(ds, keep_positions)

    return DataLoader(
        subset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 2,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

def _coerce_includes(date: str, hour_prefix: str, includes: List[str] | None) -> List[str]:
    """
    Build the restrictive include list used for download when --prepare-data is set.

    Priority rules:
    - If --include is provided, use those (as-is).
    - Else if --hour-prefix provided, include f"{date}_{hour_prefix}*.nc".
    - Else: no includes (download full day).
    """
    if includes:
        cleaned = [s.strip() for s in includes if s.strip()]
        return cleaned

    if hour_prefix:
        hp = hour_prefix.strip()
        if len(hp) != 2 or not hp.isdigit() or not (0 <= int(hp) <= 23):
            raise ValueError(f"--hour-prefix must be an hour 00-23, got: {hour_prefix}")
        return [f"{date}_{hp}*.nc"]

    return []

def _count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _set_trainable_params(model: torch.nn.Module, mode: str, logger: Any) -> None:
    """
    mode:
      - "all": train everything
      - "head": freeze most weights and unfreeze a small, safe subset
    """
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        trainable, total = _count_trainable_params(model)
        logger.info("Trainable params: %d / %d (mode=all)", trainable, total)
        return

    if mode != "head":
        raise ValueError(f"Unknown trainable mode: {mode}")

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Heuristic unfreeze (robust across many ViT-like models):
    # 1) Unfreeze all LayerNorm weights/biases (tiny memory, makes loss differentiable)
    # 2) Unfreeze the last transformer block if present
    unfrozen_names: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            for pn, p in module.named_parameters(recurse=False):
                p.requires_grad = True
                unfrozen_names.append(f"{name}.{pn}")

    # Try to unfreeze the last block / last layer collection if the model has something like blocks/layers
    # Common patterns: model.blocks (nn.ModuleList) or model.layers (nn.ModuleList)
    if hasattr(model, "blocks"):
        blocks = getattr(model, "blocks")
        try:
            last = blocks[-1]
            for name, p in last.named_parameters():
                p.requires_grad = True
                unfrozen_names.append(f"blocks[-1].{name}")
        except Exception:
            pass

    if hasattr(model, "layers"):
        layers = getattr(model, "layers")
        try:
            last = layers[-1]
            for name, p in last.named_parameters():
                p.requires_grad = True
                unfrozen_names.append(f"layers[-1].{name}")
        except Exception:
            pass

    trainable, total = _count_trainable_params(model)
    logger.info("Trainable params: %d / %d (mode=head)", trainable, total)
    if trainable == 0:
        raise RuntimeError(
            "trainable=head resulted in 0 trainable parameters. "
            "HelioSpectFormer does not expose recognizable head/decoder params, and the fallback "
            "LayerNorm/last-block unfreeze did not apply. Please inspect model.named_parameters() "
            "and choose a better unfreeze rule."
        )

    # Optional: log a few unfrozen names for sanity
    preview = unfrozen_names[:20]
    logger.info("Unfrozen param examples (up to 20): %s", preview)

def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    objective,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_data, batch_metadata in dataloader:
            batch_data = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch_data.items()
            }

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    loss, _ = objective.compute_loss(model, batch_data)
            else:
                loss, _ = objective.compute_loss(model, batch_data)

            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Evaluation dataloader produced zero batches.")

    return total_loss / total_batches

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue training Surya with time-advancement objective (Phase 1 baseline)."
    )

    # Dataset-path (used for training; also used as download root when --prepare-data)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="pretext_experiments/data/core-sdo",
        help=(
            "Local root directory containing downloaded .nc files (recursively). "
            "Example: pretext_experiments/data/core-sdo or pretext_experiments/data/core-sdo/2024/12"
        ),
    )

    # Optional: do download+index+split inside this script
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help=(
            "If set, this script will (1) download a subset from S3, (2) build a full index CSV, "
            "(3) split train/val/test by time, and then run training."
        ),
    )
    parser.add_argument(
        "--download-date",
        type=str,
        default="20241231",
        help="Date to download when --prepare-data is set (YYYYMMDD).",
    )
    parser.add_argument(
        "--hour-prefix",
        type=str,
        default="",
        help=(
            "Optional hour restriction (00-23) when --prepare-data is set and --include is not provided. "
            'Example: --hour-prefix 10 downloads "YYYYMMDD_10*.nc".'
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "Additional include glob(s) used during download when --prepare-data is set. "
            'Example: --include "20241231_10*.nc" (can be provided multiple times).'
        ),
    )

    parser.add_argument(
        "--full-index-csv",
        type=str,
        default="pretext_experiments/outputs/index/full_index.csv",
        help="Where to write the full index CSV when --prepare-data is set.",
    )
    parser.add_argument(
        "--split-out-dir",
        type=str,
        default="pretext_experiments/outputs/index",
        help="Where to write train/val/test index CSVs when --prepare-data is set.",
    )
    parser.add_argument(
        "--split-gap-hours",
        type=float,
        default=0.0,
        help="Optional temporal gap (hours) to remove around split boundaries when --prepare-data is set.",
    )
    parser.add_argument(
        "--split-train-frac",
        type=float,
        default=0.80,
        help="Train fraction for contiguous fraction split (only used when --prepare-data and --split-test-days=0).",
    )
    parser.add_argument(
        "--split-val-frac",
        type=float,
        default=0.10,
        help="Val fraction for contiguous fraction split (only used when --prepare-data and --split-test-days=0).",
    )
    parser.add_argument(
        "--split-test-frac",
        type=float,
        default=0.10,
        help="Test fraction for contiguous fraction split (only used when --prepare-data and --split-test-days=0).",
    )
    parser.add_argument(
        "--split-test-days",
        type=int,
        default=0,
        help="If > 0, use day-holdout split with last N days as test (only used when --prepare-data).",
    )
    parser.add_argument(
        "--split-val-days",
        type=int,
        default=0,
        help="If > 0, use day-holdout split with preceding N days as val (only used when --prepare-data).",
    )

    parser.add_argument("--model-dir", type=str, default="pretext_experiments/data/Surya-1.0")

    # Default train index path (used when NOT preparing data)
    parser.add_argument(
        "--train-index-csv",
        type=str,
        default="pretext_experiments/outputs/index/train_index.csv",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a forward-only smoke test (no backward/optimizer) to validate the pipeline on small GPUs.",
    )

    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer. Use sgd for low-memory local debugging."
    )

    parser.add_argument(
        "--trainable",
        type=str,
        default="all",
        choices=["all", "head"],
        help="Trainable params. Use head for low-memory local debugging."
    )

    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=200)

    parser.add_argument("--runs-dir", type=str, default="pretext_experiments/outputs/runs")
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="Dataset rollout_steps (1 matches Surya test baseline).",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run post-training visualization on a few samples."
    )

    parser.add_argument(
        "--viz-index-csv",
        type=str,
        default="",
        help=(
            "Optional index CSV to use for visualization. "
            "If empty, uses val_index.csv when --prepare-data is set, otherwise train_index.csv."
        ),
    )

    parser.add_argument(
        "--viz-batches",
        type=int,
        default=8,
        help="Number of batches to visualize."
    )

    parser.add_argument(
        "--viz-save-path",
        type=str,
        default="",
        help="Optional output image path. If empty, saves into the run directory."
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate the loaded checkpoint."
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    base_seed = int(args.seed)
    rank = int(get_rank())
    set_global_seed(base_seed + rank)

    run_name = args.run_name.strip() or f"time_adv_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = create_run_dir(args.runs_dir, run_name)

    logger = setup_surya_logger(run_dir, name="pretext_main")
    metrics_logger = JsonlLogger(Path(run_dir) / "logs" / "metrics.jsonl")

    # ------------------------------------------------------------
    # Optional: download/index/split
    # ------------------------------------------------------------
    dataset_root = Path(args.dataset_path)

    if args.prepare_data:
        includes = _coerce_includes(args.download_date, args.hour_prefix, args.include)

        logger.info("Preparing data (download -> index -> split)")
        logger.info("Download date: %s", args.download_date)
        if includes:
            logger.info("Download include globs: %s", includes)
        else:
            logger.info("Download include globs: <none> (full day)")

        prepared = prepare_sdo_data_for_run(
            download=DownloadParams(
                date=args.download_date,
                local_root=dataset_root,
                mirror_year_month_dirs=True,
                include_globs=tuple(includes),
            ),
            full_index_csv=Path(args.full_index_csv),
            split_params=SplitParams(
                index_csv=Path(args.full_index_csv),  # will be overwritten internally, but must be a Path
                out_dir=Path(args.split_out_dir),
                train_name="train_index.csv",
                val_name="val_index.csv",
                test_name="test_index.csv",
                train_frac=float(args.split_train_frac),
                val_frac=float(args.split_val_frac),
                test_frac=float(args.split_test_frac),
                test_days=int(args.split_test_days),
                val_days=int(args.split_val_days),
                gap_hours=float(args.split_gap_hours),
            ),
            index_relative_paths=True,
            index_skip_unparseable=True,
            index_sort_by_time=True,
        )

        # IMPORTANT: dataset_root returned is the actual leaf dir used by downloader
        # (typically <dataset_root>/<YYYY>/<MM>), and the index paths are relative to that.
        dataset_root = prepared.dataset_root
        train_index_csv = prepared.train_index_csv
        val_index_csv = prepared.val_index_csv
        test_index_csv = prepared.test_index_csv

        logger.info("Prepared dataset root: %s", dataset_root)
        logger.info("Prepared train index CSV: %s", train_index_csv)
        logger.info("Prepared val index CSV: %s", val_index_csv)
        logger.info("Prepared test index CSV: %s", test_index_csv)

    else:
        # Original behavior: require dataset-path and train-index-csv to exist
        if not dataset_root.exists():
            raise FileNotFoundError(f"--dataset-path does not exist: {dataset_root}")

        train_index_csv = Path(args.train_index_csv)
        if not train_index_csv.exists():
            raise FileNotFoundError(
                f"Train index CSV not found: {train_index_csv}. Run make_index.py + split_by_time.py first."
            )
        val_index_csv = Path(args.split_out_dir) / "val_index.csv"
        test_index_csv = Path(args.split_out_dir) / "test_index.csv"

        logger.info("Dataset root: %s", dataset_root)
        logger.info("Train index CSV: %s", train_index_csv)
        logger.info("Prepared val index CSV: %s", val_index_csv)
        logger.info("Prepared test index CSV: %s", test_index_csv)

    # ------------------------------------------------------------
    # Hparams logging
    # ------------------------------------------------------------
    hparams = {
        "run_name": run_name,
        "objective": "time_advancement",
        "seed": int(args.seed),
        "effective_seed": int(base_seed + rank),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "dataset_path": str(dataset_root),
        "train_index_csv": str(train_index_csv),
        "rollout_steps": int(args.rollout_steps),
        "prepare_data": bool(args.prepare_data),
        "smoke_test": bool(args.smoke_test),
    }
    if args.prepare_data:
        hparams.update(
            {
                "download_date": args.download_date,
                "download_includes": _coerce_includes(args.download_date, args.hour_prefix, args.include),
                "full_index_csv": str(Path(args.full_index_csv)),
                "split_out_dir": str(Path(args.split_out_dir)),
                "split_gap_hours": float(args.split_gap_hours),
                "split_mode": "day_holdout" if int(args.split_test_days) > 0 else "fraction",
                "split_train_frac": float(args.split_train_frac),
                "split_val_frac": float(args.split_val_frac),
                "split_test_frac": float(args.split_test_frac),
                "split_test_days": int(args.split_test_days),
                "split_val_days": int(args.split_val_days),
            }
        )

    write_hparams(run_dir, hparams, filename="hparams.json")

    # ------------------------------------------------------------
    # Load Surya model + scalers
    # ------------------------------------------------------------
    model_root = Path(ensure_surya_base_model(local_dir=args.model_dir))
    logger.info("Model directory: %s", model_root)

    config = _load_yaml(model_root / "config.yaml")
    scalers_info = _load_yaml(model_root / "scalers.yaml")
    scalers = build_scalers(info=scalers_info)

    # Build model + load pretrained weights
    model = _build_model_from_config(config)

    weights_path = model_root / "surya.366m.v1.pt"
    _load_weights_strict(model, weights_path, device="cpu")
    logger.info("Loaded baseline weights: %s", weights_path)

    # _set_trainable_params(model, args.trainable, logger)

    # ------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------
    dl = _build_train_dataloader(
        train_index_csv=train_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        rollout_steps=int(args.rollout_steps),
    )
    
    val_dl = _build_split_eval_dataloader(
        full_index_csv=Path(args.full_index_csv),
        allowed_index_csv=val_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        rollout_steps=int(args.rollout_steps),
        phase="valid",
    )

    test_dl = _build_split_eval_dataloader(
        full_index_csv=Path(args.full_index_csv),
        allowed_index_csv=test_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        rollout_steps=int(args.rollout_steps),
        phase="test",
    )

    viz_dl = None
    if args.visualize:
        if args.viz_index_csv.strip():
            viz_index_csv = Path(args.viz_index_csv)
        else:
            if args.prepare_data:
                viz_index_csv = Path(args.full_index_csv)
            else:
                viz_index_csv = train_index_csv

        if not viz_index_csv.exists():
            raise FileNotFoundError(f"Visualization index CSV not found: {viz_index_csv}")

        viz_phase = "valid" if args.prepare_data else "train"

        viz_dl = _build_eval_dataloader(
            index_csv=viz_index_csv,
            dataset_root=dataset_root,
            config=config,
            scalers=scalers,
            num_workers=0,  # safer and lighter for viz
            pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
            rollout_steps=int(args.rollout_steps),
            phase=viz_phase,
        )

    # ------------------------------------------------------------
    # Eval-only mode
    # ------------------------------------------------------------
    if args.eval_only:
        logger.info("Running in eval-only mode (no training).")

        eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model.to(eval_device)

        objective = TimeAdvancementObjective(reduce="mean")

        val_loss = evaluate_loss(model, val_dl, objective, eval_device)
        test_loss = evaluate_loss(model, test_dl, objective, eval_device)

        logger.info("Validation loss: %.6f", val_loss)
        logger.info("Test loss: %.6f", test_loss)

        if args.visualize and viz_dl is not None:
            viz_save_path = (
                Path(args.viz_save_path)
                if args.viz_save_path.strip()
                else Path(run_dir) / "prediction_samples.png"
            )

            batch_loss = visualize_batch_from_dataloader(
                model=model,
                dataloader=viz_dl,
                device=eval_device,
                rollout=int(args.rollout_steps),
                save_path=str(viz_save_path),
            )

            logger.info("Visualization saved to: %s", viz_save_path)
            logger.info("Visualization batch loss: %.6f", batch_loss)

        print("Eval-only run complete.")
        return

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    trainer_cfg = TrainerConfig(
        device=args.device,
        use_amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        keep_last_ckpt=True,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_mode=True,
        smoke_test=args.smoke_test,
        optim=args.optim,
    )

    trainer = Trainer(run_dir=run_dir, config=trainer_cfg, logger=logger, metrics_logger=metrics_logger)

    objective = TimeAdvancementObjective(reduce="mean")
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optim}")

    final_ckpt = trainer.fit_n_steps(
        model=model,
        dataloader=dl,
        objective=objective,
        n_steps=args.n_steps,
        optimizer=optimizer,
        start_step=0,
        meta={
            "baseline_weights": str(weights_path),
            "train_index_csv": str(train_index_csv),
        },
    )

    logger.info("Finished. Final checkpoint: %s", final_ckpt)

    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(eval_device)

    val_loss = evaluate_loss(model, val_dl, objective, eval_device)
    test_loss = evaluate_loss(model, test_dl, objective, eval_device)

    logger.info("Validation loss: %.6f", val_loss)
    logger.info("Test loss: %.6f", test_loss)

    if args.visualize and viz_dl is not None:
        viz_save_path = (
            Path(args.viz_save_path)
            if args.viz_save_path.strip()
            else Path(run_dir) / "prediction_samples.png"
        )

        viz_device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

        model.eval()
        batch_loss = visualize_batch_from_dataloader(
            model=model,
            dataloader=viz_dl,
            device=viz_device,
            rollout=int(args.rollout_steps),
            save_path=str(viz_save_path),
        )

        logger.info("Visualization saved to: %s", viz_save_path)
        logger.info("Visualization batch loss: %.6f", batch_loss)
    
    print(f"Run complete.\nRun dir: {run_dir}\nFinal checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()