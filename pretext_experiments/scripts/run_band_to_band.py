from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from surya.utils.data import build_scalers
from surya.utils.distributed import get_rank, set_global_seed

from pretext_experiments.pretext.data.dataset_wrappers import RandomBandMaskingDataset
from pretext_experiments.pretext.eval.band_to_band_viz import visualize_model_predictions
from pretext_experiments.pretext.objectives.band_to_band import RandomBandMaskingObjective
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
    resolve_weights_path,
)
from pretext_experiments.pretext.training.checkpointing import create_run_dir
from pretext_experiments.pretext.training.logging import JsonlLogger, setup_surya_logger, write_hparams
from pretext_experiments.pretext.training.trainer import Trainer, TrainerConfig


def _wrap_random_mask_dataset(base_ds, *, min_masked_channels: int, max_masked_channels: int | None, mask_all_timesteps: bool, seed: int, include_hmi_as_target: bool) -> RandomBandMaskingDataset:
    return RandomBandMaskingDataset(
        base_dataset=base_ds,
        min_masked_channels=min_masked_channels,
        max_masked_channels=max_masked_channels,
        mask_all_timesteps=mask_all_timesteps,
        seed=seed,
        include_hmi_as_target=include_hmi_as_target,
    )


def _build_base_dataset(*, index_csv: Path, dataset_root: Path, config: dict[str, Any], scalers: dict[str, Any], phase: str, **kwargs):
    return build_helio_dataset(
        index_csv=index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        phase=phase,
        rollout_steps=0,
        time_delta_target_minutes=0,
        drop_hmi_probability=0.0 if phase != "train" else config["data"]["drop_hmi_probability"],
        num_mask_aia_channels=0,
        random_vert_flip=bool(config["data"].get("random_vert_flip", False)) if phase == "train" else False,
    )


def _build_train_dataloader(*, train_index_csv: Path, dataset_root: Path, config: dict[str, Any], scalers: dict[str, Any], batch_size: int, num_workers: int, pin_memory: bool, min_masked_channels: int, max_masked_channels: int | None, mask_all_timesteps: bool, seed: int, include_hmi_as_target: bool):
    base_ds = _build_base_dataset(index_csv=train_index_csv, dataset_root=dataset_root, config=config, scalers=scalers, phase="train")
    ds = _wrap_random_mask_dataset(base_ds, min_masked_channels=min_masked_channels, max_masked_channels=max_masked_channels, mask_all_timesteps=mask_all_timesteps, seed=seed, include_hmi_as_target=include_hmi_as_target)
    return build_dataloader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue training Surya with band-to-band pretraining.")
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
    parser.add_argument("--ckpt-every", type=int, default=400)
    parser.add_argument("--runs-dir", type=str, default="pretext_experiments/outputs/runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--min-masked-channels", type=int, default=1)
    parser.add_argument("--max-masked-channels", type=int, default=3)
    parser.add_argument("--exclude-hmi-from-targets", action="store_true")
    parser.add_argument("--mask-only-latest-timestep", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-batches", type=int, default=8)
    parser.add_argument("--viz-save-path", type=str, default="")
    parser.add_argument("--viz-input-channels", type=int, default=3)
    args = parser.parse_args()

    base_seed = int(args.seed)
    rank = int(get_rank())
    set_global_seed(base_seed + rank)

    run_name = args.run_name.strip() or f"band_to_band_{time.strftime('%Y%m%d_%H%M%S')}"
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

    model_root = resolve_model_root(args.model_dir)
    config = load_yaml(model_root / "config.yaml")
    scalers = build_scalers(load_yaml(model_root / "scalers.yaml"))
    all_channels = list(config["data"]["sdo_channels"])

    hparams = {
        "run_name": run_name,
        "objective": "random_band_masking",
        "seed": int(args.seed),
        "effective_seed": int(base_seed + rank),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "dataset_path": str(run_data.dataset_root),
        "train_index_csv": str(run_data.train_index_csv),
        "prepare_data": bool(args.prepare_data),
        "smoke_test": bool(args.smoke_test),
        "min_masked_channels": int(args.min_masked_channels),
        "max_masked_channels": int(args.max_masked_channels),
        "mask_all_timesteps": not bool(args.mask_only_latest_timestep),
        "include_hmi_as_target": not bool(args.exclude_hmi_from_targets),
        "download_includes": resolve_include_globs(args.download_date, args.hour_prefix, args.include),
    }
    write_hparams(run_dir, hparams, filename="hparams.json")

    model = build_surya_model_from_config(config)
    weights_path = resolve_weights_path(model_root, args.weights_path)
    load_weights_strict(model, weights_path, device="cpu")
    objective = RandomBandMaskingObjective(reduce="mean")

    pin_memory = args.device == "cuda" and torch.cuda.is_available()
    val_dl = build_split_eval_dataloader(
        full_index_csv=run_data.full_index_csv,
        allowed_index_csv=run_data.val_index_csv,
        dataset_root=run_data.dataset_root,
        config=config,
        scalers=scalers,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        rollout_steps=0,
        phase="valid",
        dataset_builder_fn=lambda **kwargs: _wrap_random_mask_dataset(
            _build_base_dataset(**kwargs),
            min_masked_channels=args.min_masked_channels,
            max_masked_channels=args.max_masked_channels,
            mask_all_timesteps=not args.mask_only_latest_timestep,
            seed=args.seed + 1000,
            include_hmi_as_target=not args.exclude_hmi_from_targets,
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
        rollout_steps=0,
        phase="test",
        dataset_builder_fn=lambda **kwargs: _wrap_random_mask_dataset(
            _build_base_dataset(**kwargs),
            min_masked_channels=args.min_masked_channels,
            max_masked_channels=args.max_masked_channels,
            mask_all_timesteps=not args.mask_only_latest_timestep,
            seed=args.seed + 2000,
            include_hmi_as_target=not args.exclude_hmi_from_targets,
        ),
    )

    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(eval_device)

    if args.eval_only:
        logger.info("Eval-only | Validation loss: %.6f", evaluate_objective_loss(model, val_dl, objective, eval_device))
        logger.info("Eval-only | Test loss: %.6f", evaluate_objective_loss(model, test_dl, objective, eval_device))
        if args.visualize:
            viz_save_path = Path(args.viz_save_path) if args.viz_save_path.strip() else Path(run_dir) / "band_to_band_predictions.png"

            viz_ds = _wrap_random_mask_dataset(
                _build_base_dataset(
                    index_csv=run_data.val_index_csv,
                    dataset_root=run_data.dataset_root,
                    config=config,
                    scalers=scalers,
                    phase="valid",
                ),
                min_masked_channels=args.min_masked_channels,
                max_masked_channels=args.max_masked_channels,
                mask_all_timesteps=not args.mask_only_latest_timestep,
                seed=args.seed + 1000,
                include_hmi_as_target=not args.exclude_hmi_from_targets,
            )

            batch_loss = visualize_model_predictions(
                model,
                viz_ds,
                all_channels=all_channels,
                device=args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu",
                n_batches=args.viz_batches,
                max_input_channels_to_show=args.viz_input_channels,
                save_path=str(viz_save_path),
            )

            logger.info("Visualization batch loss: %.6f", batch_loss)
            logger.info("Visualization saved to: %s", viz_save_path)
        print("Eval-only complete.")
        return

    train_dl = _build_train_dataloader(train_index_csv=run_data.train_index_csv, dataset_root=run_data.dataset_root, config=config, scalers=scalers, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory, min_masked_channels=args.min_masked_channels, max_masked_channels=args.max_masked_channels, mask_all_timesteps=not args.mask_only_latest_timestep, seed=args.seed, include_hmi_as_target=not args.exclude_hmi_from_targets)
    trainer = Trainer(run_dir=run_dir, config=TrainerConfig(device=args.device, use_amp=not args.no_amp, grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None, log_every=args.log_every, ckpt_every=args.ckpt_every, keep_last_ckpt=True, lr=args.lr, weight_decay=args.weight_decay, train_mode=True, smoke_test=args.smoke_test, optim=args.optim), logger=logger, metrics_logger=metrics_logger)
    optimizer = create_optimizer(model, optim=args.optim, lr=args.lr, weight_decay=args.weight_decay)
    final_ckpt = trainer.fit_n_steps(model=model, dataloader=train_dl, objective=objective, n_steps=args.n_steps, optimizer=optimizer, start_step=0, meta={"baseline_weights": str(weights_path), "train_index_csv": str(run_data.train_index_csv), "min_masked_channels": int(args.min_masked_channels), "max_masked_channels": int(args.max_masked_channels), "include_hmi_as_target": not bool(args.exclude_hmi_from_targets)})
    logger.info("Finished. Final checkpoint: %s", final_ckpt)
    logger.info("Validation loss: %.6f", evaluate_objective_loss(model, val_dl, objective, eval_device))
    logger.info("Test loss: %.6f", evaluate_objective_loss(model, test_dl, objective, eval_device))
    if args.visualize:
        viz_save_path = Path(args.viz_save_path) if args.viz_save_path.strip() else Path(run_dir) / "band_to_band_predictions.png"

        viz_ds = _wrap_random_mask_dataset(
            _build_base_dataset(
                index_csv=run_data.val_index_csv,
                dataset_root=run_data.dataset_root,
                config=config,
                scalers=scalers,
                phase="valid",
            ),
            min_masked_channels=args.min_masked_channels,
            max_masked_channels=args.max_masked_channels,
            mask_all_timesteps=not args.mask_only_latest_timestep,
            seed=args.seed + 1000,
            include_hmi_as_target=not args.exclude_hmi_from_targets,
        )

        batch_loss = visualize_model_predictions(
            model,
            viz_ds,
            all_channels=all_channels,
            device=args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu",
            n_batches=args.viz_batches,
            max_input_channels_to_show=args.viz_input_channels,
            save_path=str(viz_save_path),
        )
        logger.info("Visualization batch loss: %.6f", batch_loss)
        logger.info("Visualization saved to: %s", viz_save_path)
    print("Run complete.")


if __name__ == "__main__":
    main()
