from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TimeAdvancementObjective:
    """
    Time-advancement (future forecasting) objective aligned with Surya's tests/test_surya.py.

    Expected batch_data keys (from surya.datasets.helio.HelioNetCDFDataset):
      - "ts": Tensor                 [B, C, T, H, W]  (after collate)
      - "time_delta_input": Tensor   [B, T] or [T]
      - "forecast": Tensor           [B, C, L, H, W]  (L = lead/rollout axis)

    Model call convention (as used in tests/test_surya.py):
      forecast_hat = model({"ts": ts, "time_delta_input": time_delta_input})

    Target convention:
      - If forecast has a lead dimension (forecast.ndim >= 3), take step=0 by default:
          target = forecast[:, :, 0, ...]
      - Otherwise use forecast as-is.

    Loss: MSE(forecast_hat, target)
    """

    reduce: str = "mean"  # "mean" or "sum"

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Support both dataset returns:
        #  - dict batch_data
        #  - (batch_data, batch_metadata) tuple (as in Surya tests with custom_collate_fn)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch_data = batch[0]
        else:
            batch_data = batch

        if not isinstance(batch_data, dict):
            raise TypeError(f"Expected batch_data as dict, got: {type(batch_data)}")

        ts = batch_data["ts"]
        time_delta_input = batch_data["time_delta_input"]
        forecast = batch_data["forecast"]

        curr_batch = {
            "ts": ts,
            "time_delta_input": time_delta_input,
        }
        forecast_hat = model(curr_batch)

        target = forecast
        if isinstance(forecast, torch.Tensor) and forecast.ndim >= 3:
            # forecast expected as [B, C, L, H, W] (or [B, C, L, ...])
            target = forecast[:, :, 0, ...]

        loss = F.mse_loss(forecast_hat, target, reduction=self.reduce)

        metrics = {
            "mse": float(loss.detach().cpu().item()),
        }
        return loss, metrics