# pretext_experiments/pretext/training/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader

from surya.utils.distributed import is_main_process

from pretext_experiments.pretext.training.checkpointing import save_checkpoint
from pretext_experiments.pretext.training.logging import JsonlLogger


class Objective(Protocol):
    def compute_loss(self, model: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        ...


@dataclass
class TrainerConfig:
    device: str = "cuda"
    use_amp: bool = True
    grad_clip_norm: float | None = 1.0

    log_every: int = 10
    ckpt_every: int = 200
    keep_last_ckpt: bool = True

    lr: float = 1e-4
    weight_decay: float = 0.0
    train_mode: bool = True

    # If True, run forward-only (no backward/optimizer) to smoke-test the pipeline on small GPUs.
    smoke_test: bool = False
    optim: str = "adamw"  


class Trainer:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        config: TrainerConfig,
        logger: Any,  # logging.Logger
        metrics_logger: JsonlLogger | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = config
        self.logger = logger
        self.metrics_logger = metrics_logger

        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available; falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.cfg.device)

        self.scaler: torch.cuda.amp.GradScaler | None = None
        # In smoke-test mode, we never backprop, so scaler is unnecessary.
        if (not self.cfg.smoke_test) and self.cfg.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

    def _move_batch_to_device(self, batch: Any) -> Any:
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [self._move_batch_to_device(v) for v in batch]
            return type(batch)(moved)
        return batch

    def fit_n_steps(
        self,
        *,
        model: torch.nn.Module,
        dataloader: DataLoader,
        objective: Objective,
        n_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
        start_step: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")

        model = model.to(self.device)

        # In smoke-test mode, always eval to reduce memory and avoid training-time randomness.
        if self.cfg.smoke_test:
            model.eval()
        else:
            if self.cfg.train_mode:
                model.train()

        # Optimizer is unused in smoke_test, but we keep creation optional to preserve API.
        if optimizer is None and (not self.cfg.smoke_test):
            if self.cfg.optim == "adamw":
                optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
            elif self.cfg.optim == "sgd":
                optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                momentum=0.9,
            )

        step = int(start_step)
        target_step = step + int(n_steps)

        data_iter = iter(dataloader)
        last_ckpt_path: Path | None = None

        if is_main_process():
            self.logger.info(
                "Starting training: start_step=%d, n_steps=%d, target_step=%d, device=%s, amp=%s, smoke_test=%s",
                step,
                n_steps,
                target_step,
                str(self.device),
                str(self.scaler is not None),
                str(self.cfg.smoke_test),
            )

        while step < target_step:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = self._move_batch_to_device(batch)

            # ------------------------------------------------------------
            # Smoke test: forward-only, no backward, no optimizer step
            # ------------------------------------------------------------
            if self.cfg.smoke_test:
                with torch.no_grad():
                    # autocast is optional but can reduce memory on cuda
                    if self.device.type == "cuda" and self.cfg.use_amp:
                        # Keep backwards compatibility with older PyTorch versions
                        try:
                            autocast_ctx = torch.amp.autocast("cuda")
                        except AttributeError:
                            autocast_ctx = torch.amp.autocast("cuda")
                        with autocast_ctx:
                            loss, metrics = objective.compute_loss(model, batch)
                    else:
                        loss, metrics = objective.compute_loss(model, batch)

                if loss.dim() != 0:
                    raise ValueError(f"Loss must be scalar tensor; got shape={tuple(loss.shape)}")

                step += 1

                if (step % self.cfg.log_every == 0 or step == target_step) and is_main_process():
                    loss_val = float(loss.detach().cpu().item())
                    self.logger.info("step=%d loss=%.6f %s", step, loss_val, self._fmt_metrics(metrics))
                    if self.metrics_logger is not None:
                        self.metrics_logger.log({"step": step, "split": "train", "loss": loss_val, **(metrics or {})})

                # Optional: still checkpoint at cadence so the run dir looks identical.
                if (step % self.cfg.ckpt_every == 0 or step == target_step):
                    ckpt_meta = dict(meta or {})
                    ckpt_meta.update(
                        {
                            "objective": objective.__class__.__name__,
                            "trainer_config": self.cfg.__dict__,
                            "note": "smoke_test=True (forward-only; no optimization performed)",
                        }
                    )

                    last_ckpt_path = save_checkpoint(
                        run_dir=self.run_dir,
                        step=step,
                        model=model,
                        optimizer=None,
                        scaler=None,
                        meta=ckpt_meta,
                        filename=f"step_{step}.pt",
                    )

                    if is_main_process():
                        self.logger.info("Saved checkpoint: %s", last_ckpt_path)

                    if self.cfg.keep_last_ckpt:
                        _ = save_checkpoint(
                            run_dir=self.run_dir,
                            step=step,
                            model=model,
                            optimizer=None,
                            scaler=None,
                            meta=ckpt_meta,
                            filename="last.pt",
                        )

                continue

            # ------------------------------------------------------------
            # Normal training: backward + optimizer
            # ------------------------------------------------------------
            assert optimizer is not None, "optimizer must be provided or created when not in smoke_test mode"

            optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, metrics = objective.compute_loss(model, batch)
                if loss.dim() != 0:
                    raise ValueError(f"Loss must be scalar tensor; got shape={tuple(loss.shape)}")

                self.scaler.scale(loss).backward()

                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss, metrics = objective.compute_loss(model, batch)
                if loss.dim() != 0:
                    raise ValueError(f"Loss must be scalar tensor; got shape={tuple(loss.shape)}")

                loss.backward()

                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)

                optimizer.step()

            step += 1

            if (step % self.cfg.log_every == 0 or step == target_step) and is_main_process():
                loss_val = float(loss.detach().cpu().item())
                self.logger.info("step=%d loss=%.6f %s", step, loss_val, self._fmt_metrics(metrics))
                if self.metrics_logger is not None:
                    self.metrics_logger.log({"step": step, "split": "train", "loss": loss_val, **(metrics or {})})

            if (step % self.cfg.ckpt_every == 0 or step == target_step):
                ckpt_meta = dict(meta or {})
                ckpt_meta.update({"objective": objective.__class__.__name__, "trainer_config": self.cfg.__dict__})

                last_ckpt_path = save_checkpoint(
                    run_dir=self.run_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scaler=self.scaler,
                    meta=ckpt_meta,
                    filename=f"step_{step}.pt",
                )

                if is_main_process():
                    self.logger.info("Saved checkpoint: %s", last_ckpt_path)

                if self.cfg.keep_last_ckpt:
                    _ = save_checkpoint(
                        run_dir=self.run_dir,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scaler=self.scaler,
                        meta=ckpt_meta,
                        filename="last.pt",
                    )

        if last_ckpt_path is None:
            # Should only happen if ckpt_every > n_steps and n_steps==0 (but we guard n_steps>0).
            # Keep for robustness.
            last_ckpt_path = save_checkpoint(
                run_dir=self.run_dir,
                step=step,
                model=model,
                optimizer=None if self.cfg.smoke_test else optimizer,
                scaler=None if self.cfg.smoke_test else self.scaler,
                meta=meta or {},
                filename=f"step_{step}.pt",
            )

        if is_main_process():
            self.logger.info("Training done: final_step=%d, final_ckpt=%s", step, last_ckpt_path)

        return last_ckpt_path

    @staticmethod
    def _fmt_metrics(metrics: dict[str, Any] | None) -> str:
        if not metrics:
            return ""
        parts = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}")
            else:
                parts.append(f"{k}={v}")
        return " ".join(parts)