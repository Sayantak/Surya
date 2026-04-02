#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from surya.utils.data import build_scalers, custom_collate_fn
from surya.utils.distributed import get_rank, set_global_seed
from torch.utils.data import DataLoader

from pretext_experiments.pretext.data.ar_seg import (
    ARSegDataset,
    build_or_filter_sdo_index,
    make_same_day_ar_splits,
)
from pretext_experiments.pretext.data.hf_validation import (
    ensure_ar_segmentation_data,
    ensure_ar_segmentation_surya_model,
    ensure_surya_base_model,
)
from pretext_experiments.pretext.data.utils import DownloadParams, download_sdo_subset
from pretext_experiments.pretext.eval.ar_seg_viz import visualize_ar_predictions_from_dataloader
from pretext_experiments.pretext.objectives.ar_seg import ARSegObjective, evaluate_ar_seg, select_best_threshold
from pretext_experiments.pretext.pipelines.common import create_optimizer, load_yaml, resolve_include_globs
from pretext_experiments.pretext.training.checkpointing import create_run_dir, load_checkpoint
from pretext_experiments.pretext.training.logging import JsonlLogger, setup_surya_logger, write_hparams
from pretext_experiments.pretext.training.trainer import Trainer, TrainerConfig


def _import_helio_spectformer_2d():
    repo_root = Path(__file__).resolve().parents[2]
    candidate_dirs = [
        repo_root,
        repo_root / "downstream_examples",
        repo_root / "downstream_examples" / "ar_segmentation",
        repo_root / "active_region_segmentation",
    ]
    for d in candidate_dirs:
        if d.exists() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    last_err = None
    for mod_name in ["models", "downstream_examples.models", "downstream_examples.ar_segmentation.models", "active_region_segmentation.models"]:
        try:
            module = __import__(mod_name, fromlist=["HelioSpectformer2D"])
            return module.HelioSpectformer2D
        except Exception as e:
            last_err = e
    raise ImportError("Could not import HelioSpectformer2D from downstream models.py.") from last_err


HelioSpectformer2D = _import_helio_spectformer_2d()


def _extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")


def _find_single_checkpoint_file(model_dir: Path) -> Path:
    ckpt_files = sorted(model_dir.rglob("*.pt")) + sorted(model_dir.rglob("*.pth"))
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No .pt or .pth file found under: {model_dir}")
    if len(ckpt_files) > 1:
        raise RuntimeError(f"Found multiple checkpoint files under {model_dir}: {ckpt_files}. Pass --init-ckpt explicitly.")
    return ckpt_files[0]


def _apply_peft_lora(model: torch.nn.Module, *, r: int, lora_alpha: int, target_modules: list[str], lora_dropout: float, bias: str) -> torch.nn.Module:
    return get_peft_model(
        model,
        LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias=bias),
    )


def _load_partial_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, *, map_location: str | torch.device = "cpu", logger=None) -> tuple[int, list[str], list[str]]:
    payload = load_checkpoint(ckpt_path, map_location=map_location)
    state = _extract_model_state(payload)
    model_state = model.state_dict()
    filtered: dict[str, Any] = {}
    skipped: list[str] = []
    for k, v in state.items():
        if k not in model_state or model_state[k].shape != v.shape:
            skipped.append(k)
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    step = int(payload.get("step", 0)) if isinstance(payload, dict) else 0
    if logger is not None:
        logger.info("Loaded partial checkpoint from %s | matched=%d skipped=%d missing=%d unexpected=%d", str(ckpt_path), len(filtered), len(skipped), len(missing), len(unexpected))
    return step, skipped, list(missing)


def _load_authors_baseline_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, *, map_location: str | torch.device = "cpu", logger=None) -> None:
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        model_state = checkpoint
    if any(key.startswith("module.") for key in model_state.keys()):
        model_state = {key.replace("module.", ""): value for key, value in model_state.items()}
    model.load_state_dict(model_state, strict=True)
    if logger is not None:
        logger.info("Loaded authors' baseline checkpoint strictly from %s", str(ckpt_path))


def _freeze_backbone_except_head(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("unembed"):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    return trainable


def _freeze_for_lora_finetuning(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        if "lora_" in name or name.startswith("unembed"):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    return trainable


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_ar_model_from_surya_config(base_config: dict[str, Any], *, ft_unembedding_type: str) -> torch.nn.Module:
    model_cfg = base_config["model"]
    data_cfg = base_config["data"]
    cfg = json.loads(json.dumps(base_config))
    cfg["model"]["ft_unembedding_type"] = ft_unembedding_type
    cfg["model"]["ft_out_chans"] = 1
    cfg["model"]["finetune"] = True
    return HelioSpectformer2D(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=len(data_cfg["sdo_channels"]),
        embed_dim=model_cfg["embed_dim"],
        time_embedding={"type": "linear", "time_dim": 1},
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
        finetune=True,
        config=cfg,
    )


def _build_authors_baseline_model_from_surya_config(base_config: dict[str, Any], *, ft_unembedding_type: str) -> torch.nn.Module:
    model = _build_ar_model_from_surya_config(base_config, ft_unembedding_type=ft_unembedding_type)
    return _apply_peft_lora(model, r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"], lora_dropout=0.1, bias="none")


def _build_ar_dataset(*, split_csvs: list[str], sdo_index_csv: str, sdo_data_root: str, mask_root: str, config: dict[str, Any], scalers: dict[str, Any], phase: str, n_input_timestamps: int) -> ARSegDataset:
    if n_input_timestamps != 1:
        raise ValueError(f"AR segmentation is configured for a single same-time input, but got n_input_timestamps={n_input_timestamps}")
    return ARSegDataset(
        sdo_data_root_path=sdo_data_root,
        index_path=sdo_index_csv,
        time_delta_input_minutes=[0],
        time_delta_target_minutes=0,
        n_input_timestamps=n_input_timestamps,
        rollout_steps=0,
        scalers=scalers,
        channels=config["data"]["sdo_channels"],
        phase=phase,
        ds_ar_index_paths=split_csvs,
        mask_root=mask_root,
        num_mask_aia_channels=0,
        use_latitude_in_learned_flow=config["data"].get("use_latitude_in_learned_flow", False),
        pooling=config["data"].get("pooling", None),
        random_vert_flip=False,
    )


def _build_dataloader(dataset, *, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool, drop_last: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, prefetch_factor=None if num_workers == 0 else 2, collate_fn=custom_collate_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AR segmentation downstream training/eval for Surya checkpoints.")
    parser.add_argument("--sdo-data-root", type=str, default="pretext_experiments/data/ar-seg-sdo")
    parser.add_argument("--sdo-index-csv", type=str, default="")
    parser.add_argument("--prepare-sdo-data", action="store_true")
    parser.add_argument("--download-date", type=str, default="")
    parser.add_argument("--hour-prefix", type=str, default="")
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--ar-data-root", type=str, default="pretext_experiments/data/surya-bench-ar-segmentation")
    parser.add_argument("--prepare-ar-data", action="store_true")
    parser.add_argument("--restrict-date", type=str, default="")
    parser.add_argument("--generate-ar-split", action="store_true")
    parser.add_argument("--ar-train-frac", type=float, default=0.8)
    parser.add_argument("--ar-val-frac", type=float, default=0.1)
    parser.add_argument("--ar-test-frac", type=float, default=0.1)
    parser.add_argument("--train-csv", action="append", default=None)
    parser.add_argument("--val-csv", action="append", default=None)
    parser.add_argument("--test-csv", action="append", default=None)
    parser.add_argument("--model-dir", type=str, default="pretext_experiments/data/Surya-1.0")
    parser.add_argument("--init-ckpt", type=str, default="")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--baseline-model-dir", type=str, default="pretext_experiments/data/ar_segmentation_surya")
    parser.add_argument("--finetune-mode", type=str, default="lora", choices=["full", "head_only", "lora"])
    parser.add_argument("--ft-unembedding-type", type=str, default="linear", choices=["linear", "perceiver"])
    parser.add_argument("--n-input-timestamps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-target-modules", nargs="*", default=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"])
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-pos-weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false")
    parser.set_defaults(use_pos_weight=True)
    parser.add_argument("--eval-threshold", type=float, default=-1.0)
    parser.add_argument("--threshold-sweep", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--debug-loss-once", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-batches", type=int, default=1)
    parser.add_argument("--viz-save-dir", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--runs-dir", type=str, default="pretext_experiments/outputs/runs")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    base_seed = int(args.seed)
    rank = int(get_rank())
    set_global_seed(base_seed + rank)

    run_name = args.run_name.strip() or f"ar_seg_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = create_run_dir(args.runs_dir, run_name)
    logger = setup_surya_logger(run_dir, name="pretext_main")
    metrics_logger = JsonlLogger(Path(run_dir) / "logs" / "metrics.jsonl")

    restrict_date = args.restrict_date.strip() or None
    download_date = args.download_date.strip() or None
    effective_date = restrict_date or download_date

    if args.prepare_sdo_data:
        if effective_date is None:
            raise ValueError("--prepare-sdo-data requires either --download-date or --restrict-date.")
        out_dir = download_sdo_subset(DownloadParams(date=effective_date, local_root=Path(args.sdo_data_root), mirror_year_month_dirs=True, include_globs=tuple(resolve_include_globs(effective_date, args.hour_prefix, args.include))))
        logger.info("Downloaded SDO data for %s into %s", effective_date, out_dir)

    ar_data_root = Path(args.ar_data_root)
    if args.prepare_ar_data:
        ensure_ar_segmentation_data(local_dir=ar_data_root, restrict_date=restrict_date)
        logger.info("Prepared AR segmentation labels under %s", ar_data_root)

    default_train = ar_data_root / "train.csv"
    default_val = ar_data_root / "validation.csv"
    default_test = ar_data_root / "test.csv"
    train_csvs = args.train_csv if args.train_csv else [str(default_train)]
    val_csvs = args.val_csv if args.val_csv else [str(default_val)]
    test_csvs = args.test_csv if args.test_csv else [str(default_test)]

    tmp_dir = Path(run_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if effective_date:
        effective_sdo_index_csv = build_or_filter_sdo_index(
            sdo_data_root=args.sdo_data_root,
            existing_index_csv=args.sdo_index_csv or None,
            effective_date=effective_date,
            tmp_dir=tmp_dir,
        )

        effective_train_csvs, effective_val_csvs, effective_test_csvs = make_same_day_ar_splits(
            csv_paths=train_csvs + val_csvs + test_csvs,
            date_yyyymmdd=effective_date,
            out_dir=tmp_dir / "same_day_ar_splits",
            train_frac=args.ar_train_frac,
            val_frac=args.ar_val_frac,
            test_frac=args.ar_test_frac,
        )
    else:
        effective_sdo_index_csv = args.sdo_index_csv
        effective_train_csvs, effective_val_csvs, effective_test_csvs = (
            train_csvs,
            val_csvs,
            test_csvs,
        )

    if args.generate_ar_split:
        if effective_date is None:
            raise ValueError("--generate-ar-split requires --restrict-date or --download-date.")
        effective_train_csvs, effective_val_csvs, effective_test_csvs = make_same_day_ar_splits(csv_paths=train_csvs + val_csvs + test_csvs, date_yyyymmdd=effective_date, out_dir=tmp_dir / "generated_ar_splits", train_frac=args.ar_train_frac, val_frac=args.ar_val_frac, test_frac=args.ar_test_frac)

    model_root = Path(ensure_surya_base_model(local_dir=args.model_dir))
    config = load_yaml(model_root / "config.yaml")
    scalers = build_scalers(load_yaml(model_root / "scalers.yaml"))

    if args.baseline:
        baseline_root = Path(ensure_ar_segmentation_surya_model(local_dir=args.baseline_model_dir))
        resolved_init_ckpt = Path(args.init_ckpt) if args.init_ckpt.strip() else _find_single_checkpoint_file(baseline_root)
    else:
        resolved_init_ckpt = Path(args.init_ckpt) if args.init_ckpt.strip() else _find_single_checkpoint_file(model_root)

    hparams = {
        "run_name": run_name,
        "seed": int(args.seed),
        "effective_seed": int(base_seed + rank),
        "batch_size": int(args.batch_size),
        "n_steps": int(args.n_steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "optim": args.optim,
        "n_input_timestamps": int(args.n_input_timestamps),
        "finetune_mode": args.finetune_mode,
        "ft_unembedding_type": args.ft_unembedding_type,
        "init_ckpt": str(resolved_init_ckpt),
        "baseline": bool(args.baseline),
        "sdo_data_root": args.sdo_data_root,
        "sdo_index_csv": str(effective_sdo_index_csv),
        "ar_data_root": str(ar_data_root),
        "download_date": download_date,
        "restrict_date": restrict_date,
        "train_csv": effective_train_csvs,
        "val_csv": effective_val_csvs,
        "test_csv": effective_test_csvs,
        "bce_weight": args.bce_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "use_pos_weight": args.use_pos_weight,
        "eval_threshold": args.eval_threshold,
        "threshold_sweep": [float(x) for x in args.threshold_sweep],
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_target_modules": list(args.lora_target_modules),
        "lora_dropout": float(args.lora_dropout),
        "lora_bias": str(args.lora_bias),
    }
    write_hparams(run_dir, hparams, filename="hparams.json")

    if args.baseline:
        model = _build_authors_baseline_model_from_surya_config(config, ft_unembedding_type=args.ft_unembedding_type)
        _load_authors_baseline_checkpoint(model, resolved_init_ckpt, map_location="cpu", logger=logger)
    else:
        model = _build_ar_model_from_surya_config(config, ft_unembedding_type=args.ft_unembedding_type)
        _load_partial_checkpoint(model, resolved_init_ckpt, map_location="cpu", logger=logger)
        if args.finetune_mode == "lora":
            model = _apply_peft_lora(model, r=int(args.lora_r), lora_alpha=int(args.lora_alpha), target_modules=list(args.lora_target_modules), lora_dropout=float(args.lora_dropout), bias=str(args.lora_bias))

    if not args.eval_only:
        if args.finetune_mode == "head_only":
            logger.info("Using head_only mode. Trainable params: %d.", _freeze_backbone_except_head(model))
        elif args.finetune_mode == "lora":
            logger.info("Using LoRA finetuning mode. Trainable params: %d.", _freeze_for_lora_finetuning(model))
        elif args.finetune_mode == "full":
            for p in model.parameters():
                p.requires_grad = True
            logger.info("Using full finetuning mode. Trainable params: %d", _count_trainable_params(model))

    pin_memory = args.device == "cuda" and torch.cuda.is_available()
    train_ds = _build_ar_dataset(split_csvs=effective_train_csvs, sdo_index_csv=str(effective_sdo_index_csv), sdo_data_root=args.sdo_data_root, mask_root=str(ar_data_root), config=config, scalers=scalers, phase="train", n_input_timestamps=args.n_input_timestamps)
    val_ds = _build_ar_dataset(split_csvs=effective_val_csvs, sdo_index_csv=str(effective_sdo_index_csv), sdo_data_root=args.sdo_data_root, mask_root=str(ar_data_root), config=config, scalers=scalers, phase="valid", n_input_timestamps=args.n_input_timestamps)
    test_ds = _build_ar_dataset(split_csvs=effective_test_csvs, sdo_index_csv=str(effective_sdo_index_csv), sdo_data_root=args.sdo_data_root, mask_root=str(ar_data_root), config=config, scalers=scalers, phase="test", n_input_timestamps=args.n_input_timestamps)
    train_dl = _build_dataloader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    val_dl = _build_dataloader(val_ds, batch_size=1, num_workers=args.num_workers, pin_memory=pin_memory, shuffle=False, drop_last=False)
    test_dl = _build_dataloader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=pin_memory, shuffle=False, drop_last=False)

    objective = ARSegObjective(bce_weight=args.bce_weight, dice_loss_weight=args.dice_loss_weight, use_pos_weight=args.use_pos_weight, debug_once=args.debug_loss_once)
    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(eval_device)

    if args.eval_only:
        if args.eval_threshold >= 0.0:
            chosen_threshold = float(args.eval_threshold)
            val_metrics = evaluate_ar_seg(model, val_dl, eval_device, threshold=chosen_threshold, use_pos_weight=args.use_pos_weight)
        else:
            chosen_threshold, val_metrics = select_best_threshold(model, val_dl, eval_device, thresholds=[float(x) for x in args.threshold_sweep], use_pos_weight=args.use_pos_weight)
        test_metrics = evaluate_ar_seg(model, test_dl, eval_device, threshold=chosen_threshold, use_pos_weight=args.use_pos_weight)
        logger.info("Eval-only | Validation | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f", val_metrics["loss"], val_metrics["bce"], val_metrics["soft_dice"], val_metrics["dice"], val_metrics["iou"], chosen_threshold)
        logger.info("Eval-only | Test | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f", test_metrics["loss"], test_metrics["bce"], test_metrics["soft_dice"], test_metrics["dice"], test_metrics["iou"], chosen_threshold)
        if args.visualize:
            viz_save_dir = Path(args.viz_save_dir) if args.viz_save_dir.strip() else Path(run_dir) / "figures"
            visualize_ar_predictions_from_dataloader(model=model, dataloader=test_dl, device=eval_device, save_dir=viz_save_dir, scalers=scalers, channels=config["data"]["sdo_channels"], threshold=chosen_threshold, max_batches=int(args.viz_batches))
        print("Eval-only complete.")
        return

    trainer = Trainer(run_dir=run_dir, config=TrainerConfig(device=args.device, use_amp=not args.no_amp, grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None, log_every=args.log_every, ckpt_every=args.ckpt_every, keep_last_ckpt=True, lr=args.lr, weight_decay=args.weight_decay, train_mode=True, smoke_test=args.smoke_test, optim=args.optim), logger=logger, metrics_logger=metrics_logger)
    optimizer = create_optimizer(model, optim=args.optim, lr=args.lr, weight_decay=args.weight_decay)
    final_ckpt = trainer.fit_n_steps(model=model, dataloader=train_dl, objective=objective, n_steps=args.n_steps, optimizer=optimizer, start_step=0, meta={"init_ckpt": str(resolved_init_ckpt), "finetune_mode": args.finetune_mode, "baseline": bool(args.baseline)})
    logger.info("Finished training. Final checkpoint: %s", final_ckpt)
    if args.eval_threshold >= 0.0:
        chosen_threshold = float(args.eval_threshold)
    else:
        chosen_threshold, _ = select_best_threshold(model, val_dl, eval_device, thresholds=[float(x) for x in args.threshold_sweep], use_pos_weight=args.use_pos_weight)
    val_metrics = evaluate_ar_seg(model, val_dl, eval_device, threshold=chosen_threshold, use_pos_weight=args.use_pos_weight)
    test_metrics = evaluate_ar_seg(model, test_dl, eval_device, threshold=chosen_threshold, use_pos_weight=args.use_pos_weight)
    logger.info("Validation | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f", val_metrics["loss"], val_metrics["bce"], val_metrics["soft_dice"], val_metrics["dice"], val_metrics["iou"], chosen_threshold)
    logger.info("Test | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f", test_metrics["loss"], test_metrics["bce"], test_metrics["soft_dice"], test_metrics["dice"], test_metrics["iou"], chosen_threshold)
    if args.visualize:
        viz_save_dir = Path(args.viz_save_dir) if args.viz_save_dir.strip() else Path(run_dir) / "figures"
        visualize_ar_predictions_from_dataloader(model=model, dataloader=test_dl, device=eval_device, save_dir=viz_save_dir, scalers=scalers, channels=config["data"]["sdo_channels"], threshold=chosen_threshold, max_batches=int(args.viz_batches))
        logger.info("Saved AR visualizations to: %s", viz_save_dir)
    print("Run complete.")


if __name__ == "__main__":
    main()
