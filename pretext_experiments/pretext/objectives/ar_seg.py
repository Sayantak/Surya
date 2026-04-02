from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def ensure_target_shape(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 5:
        target = target[:, 0:1, 0, ...]
    elif target.ndim == 4:
        if target.shape[1] != 1:
            target = target[:, 0:1, ...]
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    else:
        raise ValueError(f"Unsupported target shape: {tuple(target.shape)}")

    if target.shape != pred_logits.shape:
        raise ValueError(
            f"Target shape {tuple(target.shape)} does not match logits shape {tuple(pred_logits.shape)}"
        )
    return target.float()


def dice_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return dice_from_probs(torch.sigmoid(logits), target, eps=eps)


def dice_at_threshold(logits: torch.Tensor, target: torch.Tensor, threshold: float, eps: float = 1e-6) -> torch.Tensor:
    probs = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def iou_at_threshold(logits: torch.Tensor, target: torch.Tensor, threshold: float, eps: float = 1e-6) -> torch.Tensor:
    probs = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims) - inter
    return ((inter + eps) / (union + eps)).mean()


@dataclass
class ARSegObjective:
    bce_weight: float = 1.0
    dice_loss_weight: float = 1.0
    use_pos_weight: bool = True
    debug_once: bool = False
    _debug_printed: bool = field(default=False, init=False, repr=False)

    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_data = batch[0] if isinstance(batch, (tuple, list)) and len(batch) == 2 else batch
        logits = model({"ts": batch_data["ts"], "time_delta_input": batch_data["time_delta_input"]})
        target = ensure_target_shape(logits, batch_data["forecast"])

        if self.use_pos_weight:
            num_pos = target.sum()
            num_neg = target.numel() - num_pos
            pos_weight = (num_neg / num_pos.clamp_min(1.0)).clamp(max=100.0)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            pos_weight_value = float(pos_weight.detach().cpu().item())
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target)
            pos_weight_value = 1.0

        soft_dice = dice_from_logits(logits, target)
        dice_loss = 1.0 - soft_dice
        loss = self.bce_weight * bce + self.dice_loss_weight * dice_loss

        hard_dice_05 = dice_at_threshold(logits, target, threshold=0.5)
        hard_iou_05 = iou_at_threshold(logits, target, threshold=0.5)

        if self.debug_once and not self._debug_printed:
            print("\n[ar_seg debug]")
            print(f"logits shape:     {tuple(logits.shape)}")
            print(f"target shape:     {tuple(target.shape)}")
            print(f"pos_weight:       {pos_weight_value:.6f}")
            print(f"bce:              {bce.item():.6f}")
            print(f"soft_dice:        {soft_dice.item():.6f}")
            print(f"hard_dice@0.5:    {hard_dice_05.item():.6f}")
            print(f"hard_iou@0.5:     {hard_iou_05.item():.6f}")
            print("[/ar_seg debug]\n")
            self._debug_printed = True

        metrics = {
            "loss": float(loss.detach().cpu().item()),
            "bce": float(bce.detach().cpu().item()),
            "soft_dice": float(soft_dice.detach().cpu().item()),
            "hard_dice@0.5": float(hard_dice_05.detach().cpu().item()),
            "hard_iou@0.5": float(hard_iou_05.detach().cpu().item()),
            "pos_weight": float(pos_weight_value),
        }
        return loss, metrics


@torch.no_grad()
def evaluate_ar_seg(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
    use_pos_weight: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_soft_dice = 0.0
    n_batches = 0
    total_intersection = 0.0
    total_union = 0.0
    total_pred = 0.0
    total_gt = 0.0

    for batch_data, _batch_metadata in dataloader:
        batch_data = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_data.items()}
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model({"ts": batch_data["ts"], "time_delta_input": batch_data["time_delta_input"]})
            target = ensure_target_shape(logits, batch_data["forecast"])
            if use_pos_weight:
                num_pos = target.sum()
                num_neg = target.numel() - num_pos
                pos_weight = (num_neg / num_pos.clamp_min(1.0)).clamp(max=100.0)
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            else:
                bce = F.binary_cross_entropy_with_logits(logits, target)
            soft_dice = dice_from_logits(logits, target)
            preds = (torch.sigmoid(logits) >= threshold).float()
            tgt = target.float()
            inter = (preds * tgt).sum().item()
            union = (preds + tgt - preds * tgt).sum().item()
            pred_sum = preds.sum().item()
            gt_sum = tgt.sum().item()

        total_loss += float(bce.detach().cpu().item())
        total_bce += float(bce.detach().cpu().item())
        total_soft_dice += float(soft_dice.detach().cpu().item())
        total_intersection += inter
        total_union += union
        total_pred += pred_sum
        total_gt += gt_sum
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Evaluation dataloader produced zero batches.")

    eps = 1e-6
    dataset_dice = (2.0 * total_intersection + eps) / (total_pred + total_gt + eps)
    dataset_iou = (total_intersection + eps) / (total_union + eps)
    return {
        "loss": total_loss / n_batches,
        "bce": total_bce / n_batches,
        "soft_dice": total_soft_dice / n_batches,
        "dice": float(dataset_dice),
        "iou": float(dataset_iou),
        "threshold": float(threshold),
    }


@torch.no_grad()
def select_best_threshold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    thresholds: list[float],
    use_pos_weight: bool = True,
) -> tuple[float, dict[str, float]]:
    best_threshold = thresholds[0]
    best_metrics: dict[str, float] | None = None
    best_dice = -1.0
    for t in thresholds:
        metrics = evaluate_ar_seg(model, dataloader, device, threshold=float(t), use_pos_weight=use_pos_weight)
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_threshold = float(t)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics
