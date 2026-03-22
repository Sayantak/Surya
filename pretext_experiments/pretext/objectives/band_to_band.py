from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RandomBandMaskingObjective:
    """
    Random channel reconstruction objective using a fixed-size target mask.

    Expected batch_data keys:
      - "ts":                  [B, C, T, H, W]
      - "time_delta_input":    [B, T] or [T]
      - "forecast":            [B, C, L, H, W] or [B, C, H, W]
      - "target_channel_mask": [B, C] with 1 for masked target channels
    """

    reduce: str = "mean"
    debug_once: bool = False
    _debug_printed: bool = field(default=False, init=False, repr=False)

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch_data = batch[0]
        else:
            batch_data = batch

        if not isinstance(batch_data, dict):
            raise TypeError(f"Expected batch_data as dict, got: {type(batch_data)}")

        ts = batch_data["ts"]
        time_delta_input = batch_data["time_delta_input"]
        forecast = batch_data["forecast"]
        target_channel_mask = batch_data["target_channel_mask"]

        curr_batch = {
            "ts": ts,
            "time_delta_input": time_delta_input,
        }
        forecast_hat = model(curr_batch)

        if forecast.ndim == 5:
            target_full = forecast[:, :, 0, ...]
        elif forecast.ndim == 4:
            target_full = forecast
        else:
            raise ValueError(
                f"Unsupported forecast shape {tuple(forecast.shape)}. "
                "Expected [B, C, L, H, W] or [B, C, H, W]."
            )

        target_channel_mask = target_channel_mask.to(
            device=forecast_hat.device,
            dtype=forecast_hat.dtype,
        )  # [B, C]

        while target_channel_mask.ndim < forecast_hat.ndim:
            target_channel_mask = target_channel_mask.unsqueeze(-1)

        target_channel_mask = target_channel_mask.expand_as(forecast_hat)

        sq_err = (forecast_hat - target_full) ** 2
        masked_sq_err = sq_err * target_channel_mask

        num_masked_elements = target_channel_mask.sum().clamp_min(1.0)
        loss = masked_sq_err.sum() / num_masked_elements

        avg_masked_channels = (
            batch_data["target_channel_mask"].sum(dim=1).float().mean().item()
        )

        plain_mse = F.mse_loss(forecast_hat, target_full)

        if self.debug_once and not self._debug_printed:
            print("\n[band_to_band debug]")
            print(f"ts shape:               {tuple(ts.shape)}")
            print(f"forecast shape:         {tuple(forecast.shape)}")
            print(f"forecast_hat shape:     {tuple(forecast_hat.shape)}")
            print(f"target_full shape:      {tuple(target_full.shape)}")
            print(f"target_mask shape:      {tuple(target_channel_mask.shape)}")
            print(f"masked elements:        {float(num_masked_elements.item()):.1f}")
            print(f"avg masked channels:    {avg_masked_channels:.3f}")

            print(
                "forecast_hat stats:     "
                f"mean={forecast_hat.mean().item():.6f}, "
                f"std={forecast_hat.std().item():.6f}, "
                f"min={forecast_hat.min().item():.6f}, "
                f"max={forecast_hat.max().item():.6f}"
            )
            print(
                "target_full stats:      "
                f"mean={target_full.mean().item():.6f}, "
                f"std={target_full.std().item():.6f}, "
                f"min={target_full.min().item():.6f}, "
                f"max={target_full.max().item():.6f}"
            )
            print(
                "sq_err stats:           "
                f"mean={sq_err.mean().item():.6f}, "
                f"std={sq_err.std().item():.6f}, "
                f"min={sq_err.min().item():.6f}, "
                f"max={sq_err.max().item():.6f}"
            )
            print(f"plain_mse:              {plain_mse.item():.6f}")
            print(f"masked_mse:             {loss.item():.6f}")
            print("[/band_to_band debug]\n")

            self._debug_printed = True

        metrics = {
            "mse": float(loss.detach().cpu().item()),
            "plain_mse": float(plain_mse.detach().cpu().item()),
            "avg_masked_channels": float(avg_masked_channels),
        }
        return loss, metrics