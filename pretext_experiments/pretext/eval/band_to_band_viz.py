from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sunpy.visualization.colormaps as sunpy_cm


@dataclass
class BandImage:
    channel: str
    data: np.ndarray
    timestamp: str
    kind: str  # "input", "target", "pred"

def _get_channel_cmap(channel_name: str):
    ch = channel_name.lower()

    cmap_lookup = {
        "aia94": sunpy_cm.cmlist["sdoaia94"],
        "aia131": sunpy_cm.cmlist["sdoaia131"],
        "aia171": sunpy_cm.cmlist["sdoaia171"],
        "aia193": sunpy_cm.cmlist["sdoaia193"],
        "aia211": sunpy_cm.cmlist["sdoaia211"],
        "aia304": sunpy_cm.cmlist["sdoaia304"],
        "aia335": sunpy_cm.cmlist["sdoaia335"],
        "aia1600": sunpy_cm.cmlist["sdoaia1600"],
        "hmi_m": sunpy_cm.cmlist["hmimag"],
        "hmi_bx": sunpy_cm.cmlist["hmimag"],
        "hmi_by": sunpy_cm.cmlist["hmimag"],
        "hmi_bz": sunpy_cm.cmlist["hmimag"],
        "hmi_v": "gray",
    }

    return cmap_lookup.get(ch, "gray")

def _infer_input_and_target_channels(
    batch_data: dict,
    batch_metadata: dict,
    *,
    all_channels: list[str],
) -> tuple[list[str], list[str]]:
    """
    Prefer explicit metadata if present.
    Otherwise infer target channels from target_channel_mask and
    define input channels as the complement.
    """
    if "input_channels" in batch_metadata and "target_channels" in batch_metadata:
        sample_input_channels = batch_metadata["input_channels"][0]
        sample_target_channels = batch_metadata["target_channels"][0]
        return list(sample_input_channels), list(sample_target_channels)

    if "target_channel_mask" not in batch_data:
        raise KeyError(
            "Neither metadata input/target channel names nor target_channel_mask are available."
        )

    target_mask = batch_data["target_channel_mask"]
    target_mask = _to_tensor(target_mask)

    # Expect [B, C] after collect_predictions batching
    if target_mask.ndim == 1:
        target_mask = target_mask.unsqueeze(0)
    elif target_mask.ndim != 2:
        raise ValueError(
            f"Unsupported target_channel_mask shape: {tuple(target_mask.shape)}"
        )

    target_indices = torch.nonzero(target_mask[0] > 0.5, as_tuple=False).flatten().tolist()
    input_indices = [i for i in range(len(all_channels)) if i not in target_indices]

    sample_target_channels = [all_channels[i] for i in target_indices]
    sample_input_channels = [all_channels[i] for i in input_indices]

    return sample_input_channels, sample_target_channels

def _to_tensor(x):
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)

def resolve_channel_indices(all_channels: list[str], chosen_channels: list[str]) -> list[int]:
    lookup = {ch: idx for idx, ch in enumerate(all_channels)}
    missing = [ch for ch in chosen_channels if ch not in lookup]
    if missing:
        raise ValueError(f"Channels not found in all_channels: {missing}")
    return [lookup[ch] for ch in chosen_channels]


def _select_first_channel(tensor: torch.Tensor, channel_idx: int) -> np.ndarray:
    """
    Supports:
      - [B, C, H, W]
      - [C, H, W]
    Returns a single [H, W] numpy array.
    """
    if tensor.ndim == 4:
        arr = tensor[0, channel_idx].detach().float().cpu().numpy()
    elif tensor.ndim == 3:
        arr = tensor[channel_idx].detach().float().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape for channel selection: {tuple(tensor.shape)}")
    return arr


def batch_step(
    model,
    batch_data: dict,
    batch_metadata: dict,
    device: str | int,
    *,
    all_channels: list[str],
    max_input_channels_to_show: int = 3,
) -> Tuple[float, List[BandImage]]:
    curr_batch = {
        key: _to_tensor(batch_data[key]).to(device)
        for key in ["ts", "time_delta_input"]
    }

    forecast_hat = model(curr_batch)

    forecast = _to_tensor(batch_data["forecast"]).to(device)
    if forecast.ndim == 5:
        curr_target = forecast[:, :, 0, ...]
    elif forecast.ndim == 4:
        curr_target = forecast
    else:
        raise ValueError(f"Unsupported forecast shape: {tuple(forecast.shape)}")

    sample_input_channels, sample_target_channels = _infer_input_and_target_channels(
        batch_data,
        batch_metadata,
        all_channels=all_channels,
    )

    if len(sample_target_channels) == 0:
        raise RuntimeError("No target channels available for visualization.")
    if len(sample_input_channels) == 0:
        raise RuntimeError("No input channels available for visualization.")

    # Show one target channel for GT/pred
    target_channel_name = sample_target_channels[0]
    target_idx = resolve_channel_indices(all_channels, [target_channel_name])[0]

    # Show a few visible input channels for context
    shown_input_channels = sample_input_channels[:max_input_channels_to_show]
    shown_input_indices = resolve_channel_indices(all_channels, shown_input_channels)

    pred = forecast_hat[:, target_idx, ...]
    target = curr_target[:, target_idx, ...]
    loss = F.mse_loss(pred, target).item()

    ts_in = np.datetime_as_string(batch_metadata["timestamps_input"][0][-1], unit="s")
    ts_out = np.datetime_as_string(batch_metadata["timestamps_targets"][0][0], unit="s")

    data_returned: List[BandImage] = []

    for channel_name, channel_idx in zip(shown_input_channels, shown_input_indices):
        data_returned.append(
            BandImage(
                channel=channel_name,
                data=_select_first_channel(batch_data["ts"][:, :, -1, ...], channel_idx),
                timestamp=ts_in,
                kind="input",
            )
        )

    data_returned.append(
        BandImage(
            channel=all_channels[target_idx],
            data=_select_first_channel(curr_target, target_idx),
            timestamp=ts_out,
            kind="target",
        )
    )

    data_returned.append(
        BandImage(
            channel=all_channels[target_idx],
            data=_select_first_channel(forecast_hat, target_idx),
            timestamp=ts_out,
            kind="pred",
        )
    )

    return loss, data_returned


def collect_predictions(
    model,
    dataset,
    *,
    device="cuda",
    n_batches=8,
    all_channels: list[str],
    max_input_channels_to_show: int = 3,
):
    model.eval()

    losses = []
    plot_data = []

    for i in range(min(n_batches, len(dataset))):
        batch_data, batch_metadata = dataset[i]

        # Convert a single sample into batch form
        if isinstance(batch_data, dict):
            converted = {}
            for k, v in batch_data.items():
                if torch.is_tensor(v):
                    converted[k] = v.unsqueeze(0)
                elif isinstance(v, np.ndarray):
                    converted[k] = torch.as_tensor(v).unsqueeze(0)
                else:
                    converted[k] = v
            batch_data = converted

        if isinstance(batch_metadata, dict):
            batch_metadata = {
                k: [v] if not isinstance(v, list) else v
                for k, v in batch_metadata.items()
            }

        with torch.no_grad():
            if device != "cpu" and torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    batch_loss, data_returned = batch_step(
                        model,
                        batch_data,
                        batch_metadata,
                        device,
                        all_channels=all_channels,
                        max_input_channels_to_show=max_input_channels_to_show,
                    )
            else:
                batch_loss, data_returned = batch_step(
                    model,
                    batch_data,
                    batch_metadata,
                    device,
                    all_channels=all_channels,
                    max_input_channels_to_show=max_input_channels_to_show,
                )

        losses.append(batch_loss)
        plot_data.append(data_returned)

    return losses, plot_data


def plot_predictions(plot_data, losses, save_path="band_to_band_predictions.png"):
    """
    Per sample:
      [input_1] [input_2] ... [input_k] [GT target] [Pred target]
    """
    if len(plot_data) == 0:
        raise ValueError("plot_data is empty")

    n_rows = len(plot_data)
    n_cols = max(len(row) for row in plot_data)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for r, (row_data, loss_val) in enumerate(zip(plot_data, losses)):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")

            if c >= len(row_data):
                continue

            item = row_data[c]
            ax.imshow(item.data, cmap=_get_channel_cmap(item.channel))

            if item.kind == "input":
                ax.set_title(f"Input: {item.channel}\n{item.timestamp}", fontsize=10)
            elif item.kind == "target":
                ax.set_title(f"GT Target: {item.channel}\n{item.timestamp}", fontsize=10)
            elif item.kind == "pred":
                ax.set_title(f"Pred: {item.channel}\n{item.timestamp}", fontsize=10)

        axes[r, 0].set_ylabel(f"Sample {r + 1}\nLoss={loss_val:.4f}", fontsize=10)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_model_predictions(
    model,
    dataset,
    *,
    all_channels: list[str],
    device="cuda",
    n_batches=8,
    max_input_channels_to_show: int = 3,
    save_path="band_to_band_predictions.png",
):
    losses, plot_data = collect_predictions(
        model,
        dataset,
        device=device,
        n_batches=n_batches,
        all_channels=all_channels,
        max_input_channels_to_show=max_input_channels_to_show,
    )

    plot_predictions(plot_data, losses, save_path=save_path)
    return float(np.mean(losses))