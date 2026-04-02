from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from surya.datasets.helio import HelioNetCDFDataset
from surya.models.helio_spectformer import HelioSpectFormer
from surya.utils.data import custom_collate_fn

from pretext_experiments.pretext.data.hf_validation import ensure_surya_base_model
from pretext_experiments.pretext.data.utils import DownloadParams, PreparedData, SplitParams, prepare_sdo_data_for_run


@dataclass(frozen=True)
class StandardRunData:
    dataset_root: Path
    train_index_csv: Path
    val_index_csv: Path
    test_index_csv: Path
    full_index_csv: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_include_globs(
    date: str,
    hour_prefix: str,
    includes: Sequence[str] | None,
) -> list[str]:
    if includes:
        cleaned = [s.strip() for s in includes if str(s).strip()]
        if cleaned:
            return cleaned

    if hour_prefix:
        hp = hour_prefix.strip()
        if len(hp) != 2 or not hp.isdigit() or not (0 <= int(hp) <= 23):
            raise ValueError(f"--hour-prefix must be an hour 00-23, got: {hour_prefix}")
        return [f"{date}_{hp}*.nc"]

    return []


def prepare_standard_sdo_run_data(
    *,
    dataset_path: str | Path,
    prepare_data: bool,
    download_date: str,
    hour_prefix: str,
    includes: Sequence[str] | None,
    full_index_csv: str | Path,
    split_out_dir: str | Path,
    split_gap_hours: float,
    split_train_frac: float,
    split_val_frac: float,
    split_test_frac: float,
    split_test_days: int,
    split_val_days: int,
    train_index_csv: str | Path,
) -> StandardRunData:
    dataset_root = Path(dataset_path)
    full_index_csv = Path(full_index_csv)
    split_out_dir = Path(split_out_dir)

    if prepare_data:
        prepared: PreparedData = prepare_sdo_data_for_run(
            download=DownloadParams(
                date=download_date,
                local_root=dataset_root,
                mirror_year_month_dirs=True,
                include_globs=tuple(resolve_include_globs(download_date, hour_prefix, includes)),
            ),
            full_index_csv=full_index_csv,
            split_params=SplitParams(
                index_csv=full_index_csv,
                out_dir=split_out_dir,
                train_name="train_index.csv",
                val_name="val_index.csv",
                test_name="test_index.csv",
                train_frac=float(split_train_frac),
                val_frac=float(split_val_frac),
                test_frac=float(split_test_frac),
                test_days=int(split_test_days),
                val_days=int(split_val_days),
                gap_hours=float(split_gap_hours),
            ),
            index_relative_paths=True,
            index_skip_unparseable=True,
            index_sort_by_time=True,
        )
        return StandardRunData(
            dataset_root=prepared.dataset_root,
            train_index_csv=prepared.train_index_csv,
            val_index_csv=prepared.val_index_csv,
            test_index_csv=prepared.test_index_csv,
            full_index_csv=prepared.full_index_csv,
        )

    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset-path does not exist: {dataset_root}")

    train_index = Path(train_index_csv)
    val_index = split_out_dir / "val_index.csv"
    test_index = split_out_dir / "test_index.csv"

    for p, label in ((train_index, "Train"), (val_index, "Val"), (test_index, "Test")):
        if not p.exists():
            raise FileNotFoundError(f"{label} index CSV not found: {p}")

    return StandardRunData(
        dataset_root=dataset_root,
        train_index_csv=train_index,
        val_index_csv=val_index,
        test_index_csv=test_index,
        full_index_csv=full_index_csv,
    )


def build_surya_model_from_config(config: dict[str, Any]) -> HelioSpectFormer:
    model_cfg = config["model"]
    data_cfg = config["data"]
    return HelioSpectFormer(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=len(data_cfg["sdo_channels"]),
        embed_dim=model_cfg["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": len(data_cfg["time_delta_input_minutes"]),
        },
        depth=model_cfg["depth"],
        n_spectral_blocks=model_cfg["n_spectral_blocks"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        drop_rate=model_cfg["drop_rate"],
        dtype=torch.bfloat16,
        window_size=model_cfg["window_size"],
        dp_rank=model_cfg["dp_rank"],
        learned_flow=model_cfg["learned_flow"],
        use_latitude_in_learned_flow=model_cfg["learned_flow"],
        init_weights=False,
        checkpoint_layers=list(range(model_cfg["depth"])),
        rpe=model_cfg["rpe"],
        ensemble=model_cfg["ensemble"],
        finetune=model_cfg["finetune"],
    )


def resolve_model_root(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    if (model_dir / "config.yaml").exists() and (model_dir / "scalers.yaml").exists():
        return model_dir
    return Path(ensure_surya_base_model(local_dir=str(model_dir)))


def resolve_weights_path(model_root: str | Path, explicit: str = "") -> Path:
    model_root = Path(model_root)
    if explicit.strip():
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {p}")
        return p

    preferred = model_root / "surya.366m.v1.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(model_root.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt weight files found in model directory: {model_root}. Pass --weights-path explicitly."
        )
    return candidates[0]


def load_weights_strict(model: torch.nn.Module, weights_path: str | Path, device: str = "cpu") -> None:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Baseline weights not found: {weights_path}")

    try:
        payload = torch.load(weights_path, map_location=torch.device(device), weights_only=True)
    except TypeError:
        payload = torch.load(weights_path, map_location=torch.device(device))

    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]

    model.load_state_dict(payload, strict=True)


def _dataset_common_kwargs(
    *,
    config: dict[str, Any],
    scalers: dict[str, Any],
    phase: str,
    dataset_root: str | Path,
    index_csv: str | Path,
    rollout_steps: int,
    time_delta_target_minutes: int,
    drop_hmi_probability: float | None = None,
    num_mask_aia_channels: int | None = None,
    random_vert_flip: bool | None = None,
) -> dict[str, Any]:
    data_cfg = config["data"]
    return {
        "index_path": str(index_csv),
        "time_delta_input_minutes": data_cfg["time_delta_input_minutes"],
        "time_delta_target_minutes": time_delta_target_minutes,
        "n_input_timestamps": len(data_cfg["time_delta_input_minutes"]),
        "rollout_steps": rollout_steps,
        "channels": data_cfg["sdo_channels"],
        "drop_hmi_probability": (
            data_cfg["drop_hmi_probability"] if drop_hmi_probability is None else float(drop_hmi_probability)
        ),
        "num_mask_aia_channels": (
            data_cfg["num_mask_aia_channels"] if num_mask_aia_channels is None else int(num_mask_aia_channels)
        ),
        "use_latitude_in_learned_flow": data_cfg["use_latitude_in_learned_flow"],
        "scalers": scalers,
        "phase": phase,
        "pooling": data_cfg.get("pooling", None),
        "random_vert_flip": (
            bool(data_cfg.get("random_vert_flip", False)) if random_vert_flip is None else bool(random_vert_flip)
        ),
        "sdo_data_root_path": str(dataset_root),
    }


def build_helio_dataset(
    *,
    index_csv: str | Path,
    dataset_root: str | Path,
    config: dict[str, Any],
    scalers: dict[str, Any],
    phase: str,
    rollout_steps: int,
    time_delta_target_minutes: int,
    drop_hmi_probability: float | None = None,
    num_mask_aia_channels: int | None = None,
    random_vert_flip: bool | None = None,
) -> HelioNetCDFDataset:
    ds = HelioNetCDFDataset(
        **_dataset_common_kwargs(
            config=config,
            scalers=scalers,
            phase=phase,
            dataset_root=dataset_root,
            index_csv=index_csv,
            rollout_steps=rollout_steps,
            time_delta_target_minutes=time_delta_target_minutes,
            drop_hmi_probability=drop_hmi_probability,
            num_mask_aia_channels=num_mask_aia_channels,
            random_vert_flip=random_vert_flip,
        )
    )
    if len(ds) == 0:
        raise RuntimeError(
            f"Dataset has length 0 for index {index_csv}. This usually means the subset does not contain enough temporal context."
        )
    return ds


def restrict_dataset_to_allowed_times(
    dataset: Dataset,
    allowed_index_csv: str | Path,
) -> Subset:
    allowed_df = pd.read_csv(allowed_index_csv)
    if "timestep" not in allowed_df.columns:
        raise ValueError(f"Missing 'timestep' column in {allowed_index_csv}")
    allowed_times = set(pd.to_datetime(allowed_df["timestep"]))

    valid_indices = getattr(dataset, "valid_indices", None)
    if valid_indices is None:
        raise AttributeError("Dataset does not expose valid_indices.")

    keep_positions: list[int] = []
    for subset_pos, ts in enumerate(valid_indices):
        if pd.to_datetime(ts) in allowed_times:
            keep_positions.append(subset_pos)

    if not keep_positions:
        raise RuntimeError(
            f"No valid evaluation samples remain after restricting with {allowed_index_csv}."
        )
    return Subset(dataset, keep_positions)


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=None if num_workers == 0 else 2,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
    )

from torch.utils.data import Subset

def build_split_eval_dataloader(
    *,
    full_index_csv: Path,
    allowed_index_csv: Path,
    dataset_root: Path,
    config: dict,
    scalers: dict,
    num_workers: int,
    pin_memory: bool,
    rollout_steps: int,
    phase: str,
    dataset_builder_fn,
):
    """
    Build eval dataloader using full_index for context,
    but only evaluating samples in allowed_index_csv.
    """

    # 1. Build full dataset (for temporal context)
    ds = dataset_builder_fn(
        index_csv=full_index_csv,
        dataset_root=dataset_root,
        config=config,
        scalers=scalers,
        phase=phase,
        rollout_steps=rollout_steps,
    )

    # 2. Load allowed timestamps
    allowed_df = pd.read_csv(allowed_index_csv)
    if "timestep" not in allowed_df.columns:
        raise ValueError(f"Missing 'timestep' column in {allowed_index_csv}")

    allowed_times = set(pd.to_datetime(allowed_df["timestep"]))

    # 3. Filter dataset positions
    if not hasattr(ds, "valid_indices"):
        raise AttributeError("Dataset must expose 'valid_indices'")

    keep_positions = [
        i for i, ts in enumerate(ds.valid_indices)
        if pd.to_datetime(ts) in allowed_times
    ]

    if len(keep_positions) == 0:
        raise RuntimeError(
            f"No valid samples remain after filtering {allowed_index_csv}"
        )

    subset = Subset(ds, keep_positions)

    # 4. Build dataloader
    return build_dataloader(
        subset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )


def evaluate_objective_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    objective: Any,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_data, _batch_metadata in dataloader:
            batch_data = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch_data.items()
            }
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    loss, _ = objective.compute_loss(model, batch_data)
            else:
                loss, _ = objective.compute_loss(model, batch_data)
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Evaluation dataloader produced zero batches.")
    return total_loss / total_batches


def create_optimizer(
    model: torch.nn.Module,
    *,
    optim: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {optim}")
