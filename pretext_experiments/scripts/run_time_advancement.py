#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from surya.utils.data import build_scalers
from surya.utils.distributed import get_rank, set_global_seed

from pretext_experiments.pretext.eval.time_adv_viz import visualize_batch_from_dataloader
from pretext_experiments.pretext.objectives.time_advancement import TimeAdvancementObjective
from pretext_experiments.pretext.pipelines.common import (
    build_dataloader,
    build_helio_dataset,
    build_split_eval_dataloader,
    build_surya_model_from_config,
    create_optimizer,
    evaluate_objective_loss,
    load_weights_strict,
    load_yaml,
    prepare_standard_sdo_run_data,
    resolve_include_globs,
    resolve_model_root,
)
from pretext_experiments.pretext.training.checkpointing import create_run_dir
from pretext_experiments.pretext.training.logging import JsonlLogger, setup_surya_logger, write_hparams
from pretext_experiments.pretext.training.trainer import Trainer, TrainerConfig


def _count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _set_trainable_params(model: torch.nn.Module, mode: str, logger: Any) -> None:
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        trainable, total = _count_trainable_params(model)
        logger.info("Trainable params: %d / %d (mode=all)", trainable, total)
        return

    if mode != "head":
        raise ValueError(f"Unknown trainable mode: {mode}")

    for p in model.parameters():
        p.requires_grad = False

    unfrozen_names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            for pn, p in module.named_parameters(recurse=False):
                p.requires_grad = True
                unfrozen_names.append(f"{name}.{pn}")

    for attr_name in ("blocks", "layers"):
        if hasattr(model, attr_name):
            try:
                last = getattr(model, attr_name)[-1]
                for name, p in last.named_parameters():
                    p.requires_grad = True
                    unfrozen_names.append(f"{attr_name}[-1].{name}")
            except Exception:
                pass

    trainable, total = _count_trainable_params(model)
    logger.info("Trainable params: %d / %d (mode=head)", trainable, total)
    if trainable == 0:
        raise RuntimeError("trainable=head resulted in 0 trainable parameters.")
    logger.info("Unfrozen param examples (up to 20): %s", unfrozen_names[:20])


def _build_train_dataloader(*, train_index_csv: Path, dataset_root: Path, config: dict[str, Any], scalers: dict[str, Any], batch_size: int, num_workers: int, pin_memory: bool, rollout_steps: int):
    ds = build_helio_dataset(
        index_csv=train_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        phase="train",
        rollout_steps=rollout_steps,
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
    )
    return build_dataloader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)


def _build_eval_dataloader(*, index_csv: Path, dataset_root: Path, config: dict[str, Any], scalers: dict[str, Any], num_workers: int, pin_memory: bool, rollout_steps: int, phase: str):
    ds = build_helio_dataset(
        index_csv=index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        phase=phase,
        rollout_steps=rollout_steps,
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        random_vert_flip=False,
    )
    return build_dataloader(ds, batch_size=1, num_workers=num_workers, pin_memory=pin_memory, shuffle=False, drop_last=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue training Surya with time-advancement objective (Phase 1 baseline).")
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
    parser.add_argument("--model-dir", type=str, default="pretext_experiments/data/Surya-1.0")
    parser.add_argument("--train-index-csv", type=str, default="pretext_experiments/outputs/index/train_index.csv")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--trainable", type=str, default="all", choices=["all", "head"])
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
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-index-csv", type=str, default="")
    parser.add_argument("--viz-batches", type=int, default=8)
    parser.add_argument("--viz-save-path", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    base_seed = int(args.seed)
    rank = int(get_rank())
    set_global_seed(base_seed + rank)

    run_name = args.run_name.strip() or f"time_adv_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = create_run_dir(args.runs_dir, run_name)
    logger = setup_surya_logger(run_dir, name="pretext_main")
    metrics_logger = JsonlLogger(Path(run_dir) / "logs" / "metrics.jsonl")

    run_data = prepare_standard_sdo_run_data(
        dataset_path=args.dataset_path,
        prepare_data=bool(args.prepare_data),
        download_date=args.download_date,
        hour_prefix=args.hour_prefix,
        includes=args.include,
        full_index_csv=args.full_index_csv,
        split_out_dir=args.split_out_dir,
        split_gap_hours=args.split_gap_hours,
        split_train_frac=args.split_train_frac,
        split_val_frac=args.split_val_frac,
        split_test_frac=args.split_test_frac,
        split_test_days=args.split_test_days,
        split_val_days=args.split_val_days,
        train_index_csv=args.train_index_csv,
    )

    hparams = {
        "run_name": run_name,
        "objective": "time_advancement",
        "seed": int(args.seed),
        "effective_seed": int(base_seed + rank),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "dataset_path": str(run_data.dataset_root),
        "train_index_csv": str(run_data.train_index_csv),
        "rollout_steps": int(args.rollout_steps),
        "prepare_data": bool(args.prepare_data),
        "smoke_test": bool(args.smoke_test),
        "download_includes": resolve_include_globs(args.download_date, args.hour_prefix, args.include),
    }
    write_hparams(run_dir, hparams, filename="hparams.json")

    model_root = resolve_model_root(args.model_dir)
    config = load_yaml(model_root / "config.yaml")
    scalers = build_scalers(info=load_yaml(model_root / "scalers.yaml"))

    model = build_surya_model_from_config(config)
    weights_path = model_root / "surya.366m.v1.pt"
    load_weights_strict(model, weights_path, device="cpu")
    _set_trainable_params(model, args.trainable, logger)

    pin_memory = args.device == "cuda" and torch.cuda.is_available()
    train_dl = _build_train_dataloader(train_index_csv=run_data.train_index_csv, dataset_root=run_data.dataset_root, config=config, scalers=scalers, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory, rollout_steps=int(args.rollout_steps))
    if args.prepare_data:
        val_dl = build_split_eval_dataloader(
            full_index_csv=run_data.full_index_csv,
            allowed_index_csv=run_data.val_index_csv,
            dataset_root=run_data.dataset_root,
            config=config,
            scalers=scalers,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            rollout_steps=int(args.rollout_steps),
            phase="valid",
            dataset_builder_fn=lambda **kwargs: build_helio_dataset(
                **kwargs,
                time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
                random_vert_flip=False,
            ),
        )

        test_dl = build_split_eval_dataloader(
            full_index_csv=run_data.full_index_csv,
            allowed_index_csv=run_data.test_index_csv,
            dataset_root=run_data.dataset_root,
            config=config,
            scalers=scalers,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            rollout_steps=int(args.rollout_steps),
            phase="test",
            dataset_builder_fn=lambda **kwargs: build_helio_dataset(
                **kwargs,
                time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
                random_vert_flip=False,
            ),
    )
    else:
        val_dl = _build_eval_dataloader(index_csv=run_data.val_index_csv, dataset_root=run_data.dataset_root, config=config, scalers=scalers, num_workers=args.num_workers, pin_memory=pin_memory, rollout_steps=int(args.rollout_steps), phase="valid")
        test_dl = _build_eval_dataloader(index_csv=run_data.test_index_csv, dataset_root=run_data.dataset_root, config=config, scalers=scalers, num_workers=args.num_workers, pin_memory=pin_memory, rollout_steps=int(args.rollout_steps), phase="test")

    viz_dl = None
    if args.visualize:
        viz_index_csv = Path(args.viz_index_csv) if args.viz_index_csv.strip() else (run_data.full_index_csv if args.prepare_data else run_data.train_index_csv)
        viz_phase = "valid" if args.prepare_data else "train"
        viz_dl = _build_eval_dataloader(index_csv=viz_index_csv, dataset_root=run_data.dataset_root, config=config, scalers=scalers, num_workers=0, pin_memory=pin_memory, rollout_steps=int(args.rollout_steps), phase=viz_phase)

    objective = TimeAdvancementObjective(reduce="mean")
    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(eval_device)

    if args.eval_only:
        logger.info("Running in eval-only mode (no training).")
        logger.info("Validation loss: %.6f", evaluate_objective_loss(model, val_dl, objective, eval_device))
        logger.info("Test loss: %.6f", evaluate_objective_loss(model, test_dl, objective, eval_device))
        if args.visualize and viz_dl is not None:
            viz_save_path = Path(args.viz_save_path) if args.viz_save_path.strip() else Path(run_dir) / "prediction_samples.png"
            batch_loss = visualize_batch_from_dataloader(model=model, dataloader=viz_dl, device=eval_device, rollout=int(args.rollout_steps), save_path=str(viz_save_path))
            logger.info("Visualization saved to: %s", viz_save_path)
            logger.info("Visualization batch loss: %.6f", batch_loss)
        print("Eval-only run complete.")
        return

    trainer = Trainer(
        run_dir=run_dir,
        config=TrainerConfig(
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
        ),
        logger=logger,
        metrics_logger=metrics_logger,
    )
    optimizer = create_optimizer(model, optim=args.optim, lr=args.lr, weight_decay=args.weight_decay)
    final_ckpt = trainer.fit_n_steps(model=model, dataloader=train_dl, objective=objective, n_steps=args.n_steps, optimizer=optimizer, start_step=0, meta={"baseline_weights": str(weights_path), "train_index_csv": str(run_data.train_index_csv)})
    logger.info("Finished. Final checkpoint: %s", final_ckpt)
    logger.info("Validation loss: %.6f", evaluate_objective_loss(model, val_dl, objective, eval_device))
    logger.info("Test loss: %.6f", evaluate_objective_loss(model, test_dl, objective, eval_device))
    if args.visualize and viz_dl is not None:
        viz_save_path = Path(args.viz_save_path) if args.viz_save_path.strip() else Path(run_dir) / "prediction_samples.png"
        batch_loss = visualize_batch_from_dataloader(model=model, dataloader=viz_dl, device=eval_device, rollout=int(args.rollout_steps), save_path=str(viz_save_path))
        logger.info("Visualization saved to: %s", viz_save_path)
        logger.info("Visualization batch loss: %.6f", batch_loss)
    print("Run complete.")


if __name__ == "__main__":
    main()
