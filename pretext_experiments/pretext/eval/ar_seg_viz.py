from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as sunpy_cm
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader


def inverse_transform_single_channel(
    data: np.ndarray,
    mean: float,
    std: float,
    sl_scale_factor: float,
    epsilon: float,
) -> np.ndarray:
    data = data * (std + epsilon) + mean
    data = np.sign(data) * np.expm1(np.abs(data))
    data = data / sl_scale_factor
    data = np.sign(data) * np.log1p(np.abs(data))
    return data


def tensor_to_numpy_channel(
    tensor: torch.Tensor,
    *,
    channel_name: str,
    scalers: dict[str, Any] | None,
) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if scalers is None or channel_name not in scalers:
        return arr

    scaler = scalers[channel_name]
    return inverse_transform_single_channel(
        arr,
        mean=float(scaler.mean),
        std=float(scaler.std),
        sl_scale_factor=float(scaler.sl_scale_factor),
        epsilon=float(scaler.epsilon),
    )


def format_ar_metadata(metadata: dict[str, Any]) -> str:
    """
    Expected metadata keys from HelioNetCDFDataset often include:
      - timestamps_input
      - timestamps_targets
    """
    parts: list[str] = []

    if "timestamps_input" in metadata:
        vals = []
        for arr in metadata["timestamps_input"]:
            vals.extend(np.datetime_as_string(arr, unit="m"))
        if vals:
            parts.append(f"Input: {' '.join(vals)}")

    if "timestamps_targets" in metadata:
        vals = []
        for arr in metadata["timestamps_targets"]:
            vals.extend(np.datetime_as_string(arr, unit="m"))
        if vals:
            parts.append(f"Target: {' '.join(vals)}")

    return " | ".join(parts)


def pick_visual_channels(all_channels: list[str]) -> list[str]:
    """
    Match the paper-style figure: prefer AIA channels only.
    Falls back gracefully if channel names differ.
    """
    preferred = ["aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600"]

    lowered = {c.lower(): c for c in all_channels}
    selected = [lowered[c] for c in preferred if c in lowered]

    if selected:
        return selected

    non_hmi = [c for c in all_channels if "hmi" not in c.lower()]
    return non_hmi[:8]


def _plot_ar_triplet(
    *,
    inputs: list[np.ndarray],
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    channel_names: list[str],
    save_path: str | Path,
    title_txt: str = "",
) -> None:
    color_channel_mapping = {
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
        "hmi_v": plt.get_cmap("bwr"),
    }

    n_cols = len(inputs)
    fig = plt.figure(figsize=(4.0 * n_cols, 12), dpi=120)
    gs = GridSpec(3, n_cols, figure=fig, wspace=0.02, hspace=0.02)

    if title_txt:
        fig.suptitle(title_txt, fontsize=18, y=0.98)

    for i, (img, ch_name) in enumerate(zip(inputs, channel_names)):
        ax = fig.add_subplot(gs[0, i])
        cmap = color_channel_mapping.get(ch_name.lower(), "gray")
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Band {ch_name}", fontsize=13)
        ax.axis("off")

    for i in range(n_cols):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        if i == 0:
            ax.set_ylabel("Prediction", fontsize=14)
        ax.axis("off")

    for i in range(n_cols):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        if i == 0:
            ax.set_ylabel("Ground Truth", fontsize=14)
        ax.axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


@torch.no_grad()
def visualize_ar_predictions_from_dataloader(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str | Path,
    scalers: dict[str, Any],
    channels: list[str],
    threshold: float,
    max_batches: int = 1,
) -> None:
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    show_channels = pick_visual_channels(channels)

    shown = 0
    for batch_idx, (batch_data, batch_metadata) in enumerate(dataloader):
        if shown >= max_batches:
            break

        batch_data = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch_data.items()
        }

        logits = model(
            {
                "ts": batch_data["ts"],
                "time_delta_input": batch_data["time_delta_input"],
            }
        )
        probs = torch.sigmoid(logits)
        pred_mask = (probs >= threshold).float()

        target = batch_data["forecast"]
        if target.ndim == 5:
            target = target[:, 0:1, 0, ...]
        elif target.ndim == 4:
            if target.shape[1] != 1:
                target = target[:, 0:1, ...]
        elif target.ndim == 3:
            target = target.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported target shape: {tuple(target.shape)}")

        ts = batch_data["ts"][0]  # [C, T, H, W]
        latest_idx = ts.shape[1] - 1 if ts.ndim == 4 else 0

        input_images = []
        for ch_name in show_channels:
            ch_idx = channels.index(ch_name)
            img = tensor_to_numpy_channel(
                ts[ch_idx, latest_idx, :, :],
                channel_name=ch_name,
                scalers=scalers,
            )
            input_images.append(img)

        pred_np = pred_mask[0, 0].detach().float().cpu().numpy()
        gt_np = target[0, 0].detach().float().cpu().numpy()

        title_txt = format_ar_metadata(batch_metadata)

        out_path = save_dir / f"ar_seg_sample_{batch_idx:03d}.png"
        _plot_ar_triplet(
            inputs=input_images,
            pred_mask=pred_np,
            gt_mask=gt_np,
            channel_names=show_channels,
            save_path=out_path,
            title_txt=title_txt,
        )
        shown += 1