#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from surya.datasets.helio import HelioNetCDFDataset
from surya.models.helio_spectformer import HelioSpectFormer
from surya.utils.data import build_scalers, custom_collate_fn
from surya.utils.distributed import get_rank, set_global_seed

from pretext_experiments.pretext.data.dataset_wrappers import RandomBandMaskingDataset
from pretext_experiments.pretext.data.utils import (
    DownloadParams,
    SplitParams,
    prepare_sdo_data_for_run,
)
from pretext_experiments.pretext.objectives.band_to_band import RandomBandMaskingObjective
from pretext_experiments.pretext.training.checkpointing import create_run_dir
from pretext_experiments.pretext.training.logging import (
    JsonlLogger,
    setup_surya_logger,
    write_hparams,
)
from pretext_experiments.pretext.training.trainer import Trainer, TrainerConfig
from pretext_experiments.pretext.eval.band_to_band_viz import (
    visualize_model_predictions,
)


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


def _coerce_includes(date: str, hour_prefix: str, includes: List[str] | None) -> List[str]:
    if includes:
        cleaned = [s.strip() for s in includes if s.strip()]
        return cleaned

    if hour_prefix:
        hp = hour_prefix.strip()
        if len(hp) != 2 or not hp.isdigit() or not (0 <= int(hp) <= 23):
            raise ValueError(f"--hour-prefix must be an hour 00-23, got: {hour_prefix}")
        return [f"{date}_{hp}*.nc"]

    return []


def _resolve_model_root(model_dir: Path) -> Path:
    if (model_dir / "config.yaml").exists() and (model_dir / "scalers.yaml").exists():
        return model_dir

    try:
        from pretext_experiments.pretext.data.model_download import ensure_surya_base_model
        return Path(ensure_surya_base_model(local_dir=str(model_dir)))
    except Exception:
        pass

    try:
        from pretext_experiments.pretext.data.hf_validation import ensure_surya_base_model
        return Path(ensure_surya_base_model(local_dir=str(model_dir)))
    except Exception as e:
        raise RuntimeError(
            "Could not resolve Surya base model directory. "
            "Either point --model-dir to a local model folder containing config.yaml, scalers.yaml, "
            "and checkpoint weights, or expose ensure_surya_base_model from your current codebase."
        ) from e


def _resolve_weights_path(model_root: Path, explicit: str) -> Path:
    if explicit.strip():
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--weights-path does not exist: {p}")
        return p

    candidates = sorted(model_root.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt weight files found in model directory: {model_root}. "
            "Pass --weights-path explicitly."
        )

    preferred = [p for p in candidates if "epoch" in p.name.lower()]
    if preferred:
        return preferred[0]

    return candidates[0]


def _build_base_dataset(
    *,
    index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    phase: str,
) -> HelioNetCDFDataset:
    data_cfg = config["data"]

    ds = HelioNetCDFDataset(
        index_path=str(index_csv),
        time_delta_input_minutes=data_cfg["time_delta_input_minutes"],
        time_delta_target_minutes=0,
        n_input_timestamps=len(data_cfg["time_delta_input_minutes"]),
        rollout_steps=0,
        channels=data_cfg["sdo_channels"],
        drop_hmi_probability=0.0 if phase != "train" else data_cfg["drop_hmi_probability"],
        num_mask_aia_channels=0,
        use_latitude_in_learned_flow=data_cfg["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase=phase,
        pooling=data_cfg.get("pooling", None),
        random_vert_flip=bool(data_cfg.get("random_vert_flip", False)) if phase == "train" else False,
        sdo_data_root_path=str(dataset_root),
    )

    if len(ds) == 0:
        raise RuntimeError(
            f"Dataset has length 0 for index {index_csv}. "
            "This usually means your downloaded subset does not contain enough temporal context."
        )

    return ds


def _wrap_random_mask_dataset(
    base_ds: HelioNetCDFDataset,
    *,
    min_masked_channels: int,
    max_masked_channels: int | None,
    mask_all_timesteps: bool,
    seed: int,
    include_hmi_as_target: bool,
) -> RandomBandMaskingDataset:
    return RandomBandMaskingDataset(
        base_dataset=base_ds,
        min_masked_channels=min_masked_channels,
        max_masked_channels=max_masked_channels,
        mask_all_timesteps=mask_all_timesteps,
        seed=seed,
        include_hmi_as_target=include_hmi_as_target,
    )


def _build_train_dataloader(
    *,
    train_index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    min_masked_channels: int,
    max_masked_channels: int | None,
    mask_all_timesteps: bool,
    seed: int,
    include_hmi_as_target: bool,
) -> DataLoader:
    base_ds = _build_base_dataset(
        index_csv=train_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        phase="train",
    )
    ds = _wrap_random_mask_dataset(
        base_ds,
        min_masked_channels=min_masked_channels,
        max_masked_channels=max_masked_channels,
        mask_all_timesteps=mask_all_timesteps,
        seed=seed,
        include_hmi_as_target=include_hmi_as_target,
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


def _build_eval_dataset(
    *,
    full_index_csv: Path,
    allowed_index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    phase: str,
    min_masked_channels: int,
    max_masked_channels: int | None,
    mask_all_timesteps: bool,
    seed: int,
    include_hmi_as_target: bool,
):
    base_ds = _build_base_dataset(
        index_csv=full_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        phase=phase,
    )

    allowed_df = pd.read_csv(allowed_index_csv)
    if "timestep" not in allowed_df.columns:
        raise ValueError(f"Missing 'timestep' column in {allowed_index_csv}")
    allowed_times = set(pd.to_datetime(allowed_df["timestep"]))

    if not hasattr(base_ds, "valid_indices"):
        raise AttributeError("HelioNetCDFDataset does not expose valid_indices.")

    keep_positions = []
    for subset_pos, ts in enumerate(base_ds.valid_indices):
        ts = pd.to_datetime(ts)
        if ts in allowed_times:
            keep_positions.append(subset_pos)

    if len(keep_positions) == 0:
        raise RuntimeError(
            f"No valid evaluation samples remain after restricting {allowed_index_csv} "
            f"against context from {full_index_csv}."
        )

    wrapped_ds = _wrap_random_mask_dataset(
        base_ds,
        min_masked_channels=min_masked_channels,
        max_masked_channels=max_masked_channels,
        mask_all_timesteps=mask_all_timesteps,
        seed=seed,
        include_hmi_as_target=include_hmi_as_target,
    )

    subset = Subset(wrapped_ds, keep_positions)
    return subset, wrapped_ds


def _build_eval_dataloader(
    *,
    full_index_csv: Path,
    allowed_index_csv: Path,
    dataset_root: Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    num_workers: int,
    pin_memory: bool,
    phase: str,
    min_masked_channels: int,
    max_masked_channels: int | None,
    mask_all_timesteps: bool,
    seed: int,
    include_hmi_as_target: bool,
):
    subset, wrapped_ds = _build_eval_dataset(
        full_index_csv=full_index_csv,
        allowed_index_csv=allowed_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        phase=phase,
        min_masked_channels=min_masked_channels,
        max_masked_channels=max_masked_channels,
        mask_all_timesteps=mask_all_timesteps,
        seed=seed,
        include_hmi_as_target=include_hmi_as_target,
    )

    dl = DataLoader(
        subset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 2,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )
    return dl, wrapped_ds


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue training Surya with band-to-band pretraining."
    )

    parser.add_argument("--dataset-path", type=str, default="pretext_experiments/data/core-sdo")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--download-date", type=str, default="20241231")
    parser.add_argument("--hour-prefix", type=str, default="")
    parser.add_argument("--include", action="append", default=None)

    parser.add_argument("--full-index-csv", type=str, default="pretext_experiments/outputs/index/full_index.csv")
    parser.add_argument("--split-out-dir", type=str, default="pretext_experiments/outputs/index")
    parser.add_argument("--split-gap-hours", type=float, default=0.0)
    parser.add_argument("--split-train-frac", type=float, default=0.80)
    parser.add_argument("--split-val-frac", type=float, default=0.10)
    parser.add_argument("--split-test-frac", type=float, default=0.10)
    parser.add_argument("--split-test-days", type=int, default=0)
    parser.add_argument("--split-val-days", type=int, default=0)

    parser.add_argument("--train-index-csv", type=str, default="pretext_experiments/outputs/index/train_index.csv")

    parser.add_argument("--model-dir", type=str, default="pretext_experiments/data/Surya-1.0")
    parser.add_argument("--weights-path", type=str, default="")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=200)

    parser.add_argument("--runs-dir", type=str, default="pretext_experiments/outputs/runs")
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--min-masked-channels", type=int, default=1)
    parser.add_argument("--max-masked-channels", type=int, default=3)
    parser.add_argument(
        "--exclude-hmi-from-targets",
        action="store_true",
        help="If set, HMI channels are never selected as masked targets.",
    )
    parser.add_argument(
        "--mask-only-latest-timestep",
        action="store_true",
        help="If set, only the latest input timestep is masked. Default behavior masks all timesteps.",
    )

    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-batches", type=int, default=8)
    parser.add_argument("--viz-save-path", type=str, default="")

    args = parser.parse_args()

    base_seed = int(args.seed)
    rank = int(get_rank())
    set_global_seed(base_seed + rank)

    run_name = args.run_name.strip() or f"band_to_band_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = create_run_dir(args.runs_dir, run_name)

    logger = setup_surya_logger(run_dir, name="pretext_main")
    metrics_logger = JsonlLogger(Path(run_dir) / "logs" / "metrics.jsonl")

    dataset_root = Path(args.dataset_path)

    if args.prepare_data:
        includes = _coerce_includes(args.download_date, args.hour_prefix, args.include)

        prepared = prepare_sdo_data_for_run(
            download=DownloadParams(
                date=args.download_date,
                local_root=dataset_root,
                mirror_year_month_dirs=True,
                include_globs=tuple(includes),
            ),
            full_index_csv=Path(args.full_index_csv),
            split_params=SplitParams(
                index_csv=Path(args.full_index_csv),
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

        dataset_root = prepared.dataset_root
        train_index_csv = prepared.train_index_csv
        val_index_csv = prepared.val_index_csv
        test_index_csv = prepared.test_index_csv
    else:
        if not dataset_root.exists():
            raise FileNotFoundError(f"--dataset-path does not exist: {dataset_root}")

        train_index_csv = Path(args.train_index_csv)
        val_index_csv = Path(args.split_out_dir) / "val_index.csv"
        test_index_csv = Path(args.split_out_dir) / "test_index.csv"

        if not train_index_csv.exists():
            raise FileNotFoundError(f"Train index CSV not found: {train_index_csv}")
        if not val_index_csv.exists():
            raise FileNotFoundError(f"Val index CSV not found: {val_index_csv}")
        if not test_index_csv.exists():
            raise FileNotFoundError(f"Test index CSV not found: {test_index_csv}")

    model_root = _resolve_model_root(Path(args.model_dir))
    logger.info("Model directory: %s", model_root)

    config = _load_yaml(model_root / "config.yaml")
    scalers_info = _load_yaml(model_root / "scalers.yaml")
    scalers = build_scalers(scalers_info)

    data_cfg = config["data"]
    all_channels = list(data_cfg["sdo_channels"])

    hparams = {
        "run_name": run_name,
        "objective": "random_band_masking",
        "seed": int(args.seed),
        "effective_seed": int(base_seed + rank),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "dataset_path": str(dataset_root),
        "train_index_csv": str(train_index_csv),
        "prepare_data": bool(args.prepare_data),
        "smoke_test": bool(args.smoke_test),
        "min_masked_channels": int(args.min_masked_channels),
        "max_masked_channels": int(args.max_masked_channels),
        "mask_all_timesteps": not bool(args.mask_only_latest_timestep),
        "include_hmi_as_target": not bool(args.exclude_hmi_from_targets),
    }
    write_hparams(run_dir, hparams, filename="hparams.json")

    model = _build_model_from_config(config)
    weights_path = _resolve_weights_path(model_root, args.weights_path)
    _load_weights_strict(model, weights_path, device="cpu")
    logger.info("Loaded baseline weights: %s", weights_path)

    objective = RandomBandMaskingObjective(reduce="mean")

    val_dl, val_ds = _build_eval_dataloader(
        full_index_csv=Path(args.full_index_csv),
        allowed_index_csv=val_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        phase="valid",
        min_masked_channels=args.min_masked_channels,
        max_masked_channels=args.max_masked_channels,
        mask_all_timesteps=not args.mask_only_latest_timestep,
        seed=args.seed + 1000,
        include_hmi_as_target=not args.exclude_hmi_from_targets,
    )

    test_dl, test_ds = _build_eval_dataloader(
        full_index_csv=Path(args.full_index_csv),
        allowed_index_csv=test_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        phase="test",
        min_masked_channels=args.min_masked_channels,
        max_masked_channels=args.max_masked_channels,
        mask_all_timesteps=not args.mask_only_latest_timestep,
        seed=args.seed + 2000,
        include_hmi_as_target=not args.exclude_hmi_from_targets,
    )

    if args.eval_only:
        eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model.to(eval_device)

        val_loss = evaluate_loss(model, val_dl, objective, eval_device)
        test_loss = evaluate_loss(model, test_dl, objective, eval_device)

        logger.info("Eval-only | Validation loss: %.6f", val_loss)
        logger.info("Eval-only | Test loss: %.6f", test_loss)

        if args.visualize:
            viz_save_path = (
                Path(args.viz_save_path)
                if args.viz_save_path.strip()
                else Path(run_dir) / "band_to_band_predictions.png"
            )
            batch_loss = visualize_model_predictions(
                model,
                val_ds,
                all_channels=all_channels,
                input_channels=input_channels,
                target_channels=target_channels,
                device=args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu",
                n_batches=args.viz_batches,
                save_path=str(viz_save_path),
            )
            logger.info("Visualization batch loss: %.6f", batch_loss)
            logger.info("Visualization saved to: %s", viz_save_path)

        print("Eval-only complete.")
        return

    train_dl = _build_train_dataloader(
        train_index_csv=train_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        min_masked_channels=args.min_masked_channels,
        max_masked_channels=args.max_masked_channels,
        mask_all_timesteps=not args.mask_only_latest_timestep,
        seed=args.seed,
        include_hmi_as_target=not args.exclude_hmi_from_targets,
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
        smoke_test=args.smoke_test,
        optim=args.optim,
    )

    trainer = Trainer(
        run_dir=run_dir,
        config=trainer_cfg,
        logger=logger,
        metrics_logger=metrics_logger,
    )

    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
        )

    final_ckpt = trainer.fit_n_steps(
        model=model,
        dataloader=train_dl,
        objective=objective,
        n_steps=args.n_steps,
        optimizer=optimizer,
        start_step=0,
        meta={
            "baseline_weights": str(weights_path),
            "train_index_csv": str(train_index_csv),
            "min_masked_channels": int(args.min_masked_channels),
            "max_masked_channels": int(args.max_masked_channels),
            "include_hmi_as_target": not bool(args.exclude_hmi_from_targets),
        },
    )

    logger.info("Finished. Final checkpoint: %s", final_ckpt)

    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(eval_device)

    val_loss = evaluate_loss(model, val_dl, objective, eval_device)
    test_loss = evaluate_loss(model, test_dl, objective, eval_device)

    logger.info("Validation loss: %.6f", val_loss)
    logger.info("Test loss: %.6f", test_loss)

    if args.visualize:
        viz_save_path = (
            Path(args.viz_save_path)
            if args.viz_save_path.strip()
            else Path(run_dir) / "band_to_band_predictions.png"
        )

        batch_loss = visualize_model_predictions(
            model,
            val_ds,
            all_channels=all_channels,
            input_channels=input_channels,
            target_channels=target_channels,
            device=args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu",
            n_batches=args.viz_batches,
            save_path=str(viz_save_path),
        )

        logger.info("Visualization batch loss: %.6f", batch_loss)
        logger.info("Visualization saved to: %s", viz_save_path)

    print("Run complete.")


if __name__ == "__main__":
    main()