from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from surya.datasets.helio import inverse_transform_single_channel
from surya.utils.data import custom_collate_fn

from pretext_experiments.pretext.data.dataset_wrappers import resolve_channel_indices

logger = logging.getLogger(__name__)


@dataclass
class BandImage:
    channel: str
    data: np.ndarray
    timestamp: str
    kind: str  # input / target / pred


def _select_first_channel(
    x: torch.Tensor,
    channel_idx: int,
) -> np.ndarray:
    return x[0, channel_idx].detach().to(dtype=torch.float32).cpu().numpy()


def batch_step(
    model,
    batch_data: dict,
    batch_metadata: dict,
    device: str | int,
    *,
    all_channels: list[str],
    input_channels: list[str],
    target_channels: list[str],
) -> Tuple[float, List[BandImage]]:
    input_idx = resolve_channel_indices(all_channels, [input_channels[0]])[0]
    target_idx = resolve_channel_indices(all_channels, [target_channels[0]])[0]

    curr_batch = {
        key: batch_data[key].to(device)
        for key in ["ts", "time_delta_input"]
    }

    forecast_hat = model(curr_batch)
    curr_target = batch_data["forecast"][:, :, 0, ...].to(device)

    pred = forecast_hat[:, target_idx, ...]
    target = curr_target[:, target_idx, ...]
    loss = F.mse_loss(pred, target).item()

    ts_in = np.datetime_as_string(batch_metadata["timestamps_input"][0][-1], unit="s")
    ts_out = np.datetime_as_string(batch_metadata["timestamps_targets"][0][0], unit="s")

    data_returned: List[BandImage] = [
        BandImage(
            channel=all_channels[input_idx],
            data=_select_first_channel(batch_data["ts"][:, :, -1, ...], input_idx),
            timestamp=ts_in,
            kind="input",
        ),
        BandImage(
            channel=all_channels[target_idx],
            data=_select_first_channel(curr_target, target_idx),
            timestamp=ts_out,
            kind="target",
        ),
        BandImage(
            channel=all_channels[target_idx],
            data=_select_first_channel(forecast_hat, target_idx),
            timestamp=ts_out,
            kind="pred",
        ),
    ]

    return loss, data_returned


def collect_predictions(
    model,
    dataset,
    *,
    device="cuda",
    n_batches=8,
    all_channels: list[str],
    input_channels: list[str],
    target_channels: list[str],
):
    dl = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    model.to(device)
    model.eval()

    losses = []
    plot_data = []

    for batch_idx, (batch_data, batch_metadata) in enumerate(dl):
        if batch_idx >= n_batches:
            break

        with torch.no_grad():
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_loss, data_returned = batch_step(
                        model,
                        batch_data,
                        batch_metadata,
                        device,
                        all_channels=all_channels,
                        input_channels=input_channels,
                        target_channels=target_channels,
                    )
            else:
                batch_loss, data_returned = batch_step(
                    model,
                    batch_data,
                    batch_metadata,
                    device,
                    all_channels=all_channels,
                    input_channels=input_channels,
                    target_channels=target_channels,
                )

        losses.append(batch_loss)
        plot_data.append(data_returned)

    return losses, plot_data


def _inverse_transform_channel(
    img: np.ndarray,
    *,
    dataset,
    channel_name: str,
) -> np.ndarray:
    means, stds, epsilons, sl_scale_factors = dataset.transformation_inputs()
    all_channels = list(getattr(dataset, "channels"))
    c_idx = resolve_channel_indices(all_channels, [channel_name])[0]
    return inverse_transform_single_channel(
        img,
        mean=means[c_idx],
        std=stds[c_idx],
        epsilon=epsilons[c_idx],
        sl_scale_factor=sl_scale_factors[c_idx],
    )


def plot_predictions(
    plot_data,
    dataset,
    *,
    save_path="band_to_band_predictions.png",
):
    n_rows = len(plot_data)
    fig, ax = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        ax = np.expand_dims(ax, axis=0)

    for row_idx, row in enumerate(plot_data):
        for col_idx, img in enumerate(row):
            arr = _inverse_transform_channel(img.data, dataset=dataset, channel_name=img.channel)

            ax[row_idx, col_idx].imshow(arr, origin="lower")
            ax[row_idx, col_idx].axis("off")
            ax[row_idx, col_idx].set_title(
                f"{img.kind.capitalize()} | {img.channel} | {img.timestamp}"
            )

    fig.suptitle("Band-to-Band Predictions", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return save_path


def visualize_model_predictions(
    model,
    dataset,
    *,
    all_channels: list[str],
    input_channels: list[str],
    target_channels: list[str],
    device="cuda",
    n_batches=8,
    save_path="band_to_band_predictions.png",
):
    losses, plot_data = collect_predictions(
        model,
        dataset,
        device=device,
        n_batches=n_batches,
        all_channels=all_channels,
        input_channels=input_channels,
        target_channels=target_channels,
    )

    if len(plot_data) == 0:
        raise RuntimeError("No batches collected for visualization.")

    plot_predictions(plot_data, dataset, save_path=save_path)
    return float(np.mean(losses))