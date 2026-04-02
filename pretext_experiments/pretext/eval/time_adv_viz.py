import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as sunpy_cm

from torch.utils.data import DataLoader

from surya.datasets.helio import inverse_transform_single_channel
from surya.utils.data import custom_collate_fn

logger = logging.getLogger(__name__)

SDO_CHANNELS = [
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "aia1600",
    "hmi_m",
    "hmi_bx",
    "hmi_by",
    "hmi_bz",
    "hmi_v",
]


@dataclass
class SDOImage:
    channel: str
    data: np.ndarray
    timestamp: str
    type: str


def batch_step(
    model,
    batch_data: dict,
    batch_metadata: dict,
    device: str | int,
    rollout: int = 0,
) -> Tuple[float, List[SDOImage]]:
    """
    Replicates Surya test batch_step behaviour for visualization.

    Returns
    -------
    loss
    list[SDOImage] containing:
        [Input t, Input t+delta, GT t+2delta, Pred t+2delta]
    """

    loss = 0.0
    n_samples_x_steps = 0
    data_returned: List[SDOImage] = []

    # record the two input frames
    for t_idx in range(2):
        timestamp = np.datetime_as_string(
            batch_metadata["timestamps_input"][0][t_idx], unit="s"
        )

        data_returned.append(
            SDOImage(
                channel="aia94",
                data=batch_data["ts"][0, SDO_CHANNELS.index("aia94"), t_idx].numpy(),
                timestamp=timestamp,
                type="input",
            )
        )

    # run one forward pass
    curr_batch = {
        key: batch_data[key].to(device)
        for key in ["ts", "time_delta_input"]
    }

    forecast_hat = model(curr_batch)

    curr_target = batch_data["forecast"][:, :, 0, ...].to(device)
    curr_batch_loss = F.mse_loss(forecast_hat, curr_target)

    loss += curr_batch_loss.item()
    n_samples_x_steps += curr_batch["ts"].shape[0]

    timestamp = np.datetime_as_string(
        batch_metadata["timestamps_targets"][0][0], unit="s"
    )

    # Ground truth
    data_returned.append(
        SDOImage(
            channel="aia94",
            data=curr_target.to(dtype=torch.float32)
            .cpu()[0, SDO_CHANNELS.index("aia94")]
            .numpy(),
            timestamp=timestamp,
            type="gt",
        )
    )

    # Prediction
    data_returned.append(
        SDOImage(
            channel="aia94",
            data=forecast_hat.to(dtype=torch.float32)
            .cpu()[0, SDO_CHANNELS.index("aia94")]
            .numpy(),
            timestamp=timestamp,
            type="pred",
        )
    )

    loss = loss / n_samples_x_steps

    return loss, data_returned

def visualize_batch_from_dataloader(
    model,
    dataloader,
    device="cuda",
    rollout=1,
    save_path="surya_predictions.png",
):
    """
    Visualize a single batch from an existing dataloader.
    Avoids rebuilding dataset/dataloader and avoids a second full inference loop.
    """
    model.to(device)
    model.eval()

    batch_data, batch_metadata = next(iter(dataloader))

    with torch.no_grad():
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss, plot_data = batch_step(
                    model,
                    batch_data,
                    batch_metadata,
                    device,
                    rollout=rollout,
                )
        else:
            batch_loss, plot_data = batch_step(
                model,
                batch_data,
                batch_metadata,
                device,
                rollout=rollout,
            )

    # Reuse existing plotting logic by wrapping single sample in a list
    plot_predictions(
        [plot_data],
        dataloader.dataset,
        save_path=save_path,
        n_rows=1,
    )

    return batch_loss

def collect_predictions(
    model,
    dataset,
    device="cuda",
    rollout=1,
    n_batches=8,
):
    """
    Run inference and collect prediction samples.
    """

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

    logger.info("Running inference for visualization")

    for batch_idx, (batch_data, batch_metadata) in enumerate(dl):

        if batch_idx >= n_batches:
            break

        with torch.no_grad():

            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_loss, data_returned = batch_step(
                        model, batch_data, batch_metadata, device, rollout
                    )
            else:
                batch_loss, data_returned = batch_step(
                    model, batch_data, batch_metadata, device, rollout
                )

        losses.append(batch_loss)
        plot_data.append(data_returned)

    logger.info("Collected %d batches for visualization", len(plot_data))

    return losses, plot_data


def inverse_transform_images(plot_data, dataset):

    means, stds, epsilons, sl_scale_factors = dataset.transformation_inputs()

    c_idx = SDO_CHANNELS.index("aia94")

    vmin = float("-inf")
    vmax = float("inf")

    for data_returned in plot_data:

        for sdo_image in data_returned:

            sdo_image.data = inverse_transform_single_channel(
                sdo_image.data,
                mean=means[c_idx],
                std=stds[c_idx],
                epsilon=epsilons[c_idx],
                sl_scale_factor=sl_scale_factors[c_idx],
            )

            vmin = max(vmin, sdo_image.data.min())
            vmax = min(vmax, np.quantile(sdo_image.data, 0.99))

    return vmin, vmax


def plot_predictions(
    plot_data,
    dataset,
    save_path="surya_predictions.png",
    n_rows=8,
):
    """
    Plot input / prediction / ground truth grid
    similar to Surya validation test.
    """

    logger.info("Preparing visualization")

    plot_data = sorted(
        plot_data,
        key=lambda data_returned: data_returned[0].timestamp,
    )

    vmin, vmax = inverse_transform_images(plot_data, dataset)

    plt_kwargs = {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": sunpy_cm.cmlist["sdoaia94"],
        "origin": "lower",
    }

    fig, ax = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))

    if n_rows == 1:
        ax = np.expand_dims(ax, axis=0)

    for j in range(n_rows):
        for i in range(4):
            ax[j, i].axis("off")
            ax[j, i].imshow(
                plot_data[j][i].data,
                **plt_kwargs,
            )

            if i == 0:
                title = f"Input - {plot_data[j][i].timestamp}"
            elif i == 1:
                title = f"Input - {plot_data[j][i].timestamp}"
            elif i == 2:
                title = f"GT - {plot_data[j][i].timestamp}"
            else:
                title = f"Pred - {plot_data[j][i].timestamp}"

            ax[j, i].set_title(title)

    fig.suptitle("Surya Predictions - AIA94", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")

    logger.info("Saved visualization to %s", save_path)

    return save_path


def visualize_model_predictions(
    model,
    dataset,
    device="cuda",
    rollout=1,
    n_batches=8,
    save_path="surya_predictions.png",
):
    """
    High-level helper used from experiment scripts.

    Runs inference + visualization.
    """

    losses, plot_data = collect_predictions(
        model,
        dataset,
        device=device,
        rollout=rollout,
        n_batches=n_batches,
    )

    plot_predictions(
        plot_data,
        dataset,
        save_path=save_path,
        n_rows=min(n_batches, len(plot_data)),
    )

    return np.mean(losses)