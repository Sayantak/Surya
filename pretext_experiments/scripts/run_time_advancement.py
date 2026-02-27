#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from surya.utils.distributed import get_rank, set_global_seed
from surya.utils.data import build_scalers, custom_collate_fn
from surya.models.helio_spectformer import HelioSpectFormer
from surya.datasets.helio import HelioNetCDFDataset

from pretext_experiments.pretext.data.hf_validation import (
    ensure_surya_base_model,
    ensure_validation_data,
)
from pretext_experiments.pretext.objectives.time_advancement import TimeAdvancementObjective
from pretext_experiments.pretext.training.checkpointing import create_run_dir
from pretext_experiments.pretext.training.logging import JsonlLogger, setup_surya_logger, write_hparams
from pretext_experiments.pretext.training.trainer import Trainer, TrainerConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model_from_config(config: dict[str, Any]) -> HelioSpectFormer:
    # This matches tests/test_surya.py model construction.
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

    # torch.load(..., weights_only=True) exists in newer torch; keep compatibility.
    try:
        weights = torch.load(weights_path, map_location=torch.device(device), weights_only=True)
    except TypeError:
        weights = torch.load(weights_path, map_location=torch.device(device))

    if not isinstance(weights, dict):
        raise TypeError(f"Expected weights to be a state_dict dict, got: {type(weights)}")

    model.load_state_dict(weights, strict=True)


def _build_train_dataloader(
    *,
    train_index_csv: Path,
    data_root: Path,
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
        sdo_data_root_path=str(data_root),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue training Surya with time-advancement objective (Phase 1 baseline).")

    parser.add_argument("--data-dir", type=str, default="../../data/Surya-1.0_validation_data")
    parser.add_argument("--model-dir", type=str, default="../../data/Surya-1.0")

    parser.add_argument(
        "--train-index-csv",
        type=str,
        default="../outputs/index/train_index.csv",
    )

    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)  # Surya baseline is batch size 1 per GPU in paper/tests
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=200)

    parser.add_argument("--runs-dir", type=str, default="../outputs/runs")
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--data-allow-pattern", action="append", default=None)
    parser.add_argument("--data-ignore-pattern", action="append", default=None)
    parser.add_argument("--data-revision", type=str, default=None)

    parser.add_argument("--rollout-steps", type=int, default=1, help="Dataset rollout_steps (1 matches Surya test baseline).")

    args = parser.parse_args()

    # Reproducible seeding: base seed + rank offset for DDP workers
    base_seed = int(args.seed)
    rank = int(get_rank())
    set_global_seed(base_seed + rank)

    run_name = args.run_name.strip() or f"time_adv_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = create_run_dir(args.runs_dir, run_name)

    logger = setup_surya_logger(run_dir, name="pretext_main")
    metrics_logger = JsonlLogger(Path(run_dir) / "logs" / "metrics.jsonl")

    hparams = {
        "run_name": run_name,
        "objective": "time_advancement",
        "seed": int(args.seed),
        "effective_seed": int(base_seed + rank),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip_norm": float(args.grad_clip_norm),
        "device": args.device,
        "use_amp": not args.no_amp,
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "train_index_csv": args.train_index_csv,
        "log_every": int(args.log_every),
        "ckpt_every": int(args.ckpt_every),
        "data_allow_patterns": args.data_allow_pattern,
        "data_ignore_patterns": args.data_ignore_pattern,
        "data_revision": args.data_revision,
        "rollout_steps": int(args.rollout_steps),
    }
    _ = write_hparams(run_dir, hparams, filename="hparams.json")

    # Ensure data/model are present locally
    data_root = Path(
        ensure_validation_data(
            local_dir=args.data_dir,
            allow_patterns=args.data_allow_pattern,
            ignore_patterns=args.data_ignore_pattern,
            revision=args.data_revision,
        )
    )
    logger.info("Validation data directory: %s", data_root)

    model_root = Path(ensure_surya_base_model(local_dir=args.model_dir))
    logger.info("Model directory: %s", model_root)

    # Load Surya config + scalers (as in tests/test_surya.py)
    config = _load_yaml(model_root / "config.yaml")
    scalers_info = _load_yaml(model_root / "scalers.yaml")
    scalers = build_scalers(info=scalers_info)

    # Build model + load pretrained weights
    model = _build_model_from_config(config)

    weights_path = model_root / "surya.366m.v1.pt"
    _load_weights_strict(model, weights_path, device="cpu")
    logger.info("Loaded baseline weights: %s", weights_path)

    train_index_csv = Path(args.train_index_csv)
    if not train_index_csv.exists():
        raise FileNotFoundError(
            f"Train index CSV not found: {train_index_csv}. Run make_index.py + split_by_time.py first."
        )

    dl = _build_train_dataloader(
        train_index_csv=train_index_csv,
        data_root=data_root,
        config=config,
        scalers=scalers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        rollout_steps=int(args.rollout_steps),
    )

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
    )
    trainer = Trainer(run_dir=run_dir, config=trainer_cfg, logger=logger, metrics_logger=metrics_logger)

    objective = TimeAdvancementObjective(reduce="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"Run complete.\nRun dir: {run_dir}\nFinal checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()