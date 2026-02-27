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
        if self.cfg.use_amp and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

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
        if self.cfg.train_mode:
            model.train()

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )

        step = int(start_step)
        target_step = step + int(n_steps)

        data_iter = iter(dataloader)
        last_ckpt_path: Path | None = None

        if is_main_process():
            self.logger.info(
                "Starting training: start_step=%d, n_steps=%d, target_step=%d, device=%s, amp=%s",
                step,
                n_steps,
                target_step,
                str(self.device),
                str(self.scaler is not None),
            )

        while step < target_step:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = self._move_batch_to_device(batch)

            optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
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
            last_ckpt_path = save_checkpoint(
                run_dir=self.run_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                scaler=self.scaler,
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