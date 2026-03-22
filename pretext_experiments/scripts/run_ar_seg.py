#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model

from surya.datasets.helio import HelioNetCDFDataset
from surya.utils.data import build_scalers, custom_collate_fn
from surya.utils.distributed import get_rank, set_global_seed

from pretext_experiments.pretext.data.hf_validation import (
    ensure_ar_segmentation_data,
    ensure_surya_base_model,
    ensure_ar_segmentation_surya_model,
)
from pretext_experiments.pretext.data.utils import (
    DownloadParams,
    IndexParams,
    build_index_csv,
    download_sdo_subset,
)
from pretext_experiments.pretext.training.checkpointing import (
    create_run_dir,
    load_checkpoint,
    save_checkpoint,
)
from pretext_experiments.pretext.training.trainer import Trainer, TrainerConfig

from pretext_experiments.pretext.eval.ar_seg_viz import (
    visualize_ar_predictions_from_dataloader,
)

# --------------------------------------------------------------------------------------
# Optional logging helpers
# --------------------------------------------------------------------------------------
try:
    from pretext_experiments.pretext.training.logging import (  # type: ignore
        JsonlLogger,
        setup_surya_logger,
        write_hparams,
    )
except Exception:
    class JsonlLogger:
        def __init__(self, path: str | Path):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)

        def log(self, payload: dict[str, Any]) -> None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

    def setup_surya_logger(run_dir: str | Path, name: str = "pretext_main"):
        run_dir = Path(run_dir)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        fmt = logging.Formatter(
            "[%(asctime)s %(name)s]: %(levelname)s %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(run_dir / "logs" / "train.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
        return logger

    def write_hparams(run_dir: str | Path, hparams: dict[str, Any], filename: str = "hparams.json") -> None:
        run_dir = Path(run_dir)
        with (run_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2)


# --------------------------------------------------------------------------------------
# Robust import of downstream AR model
# --------------------------------------------------------------------------------------
def _import_helio_spectformer_2d():
    repo_root = Path(__file__).resolve().parents[2]
    candidate_dirs = [
        repo_root,
        repo_root / "downstream_examples",
        repo_root / "downstream_examples" / "ar_segmentation",
        repo_root / "active_region_segmentation",
    ]

    for d in candidate_dirs:
        if d.exists() and str(d) not in sys.path:
            sys.path.insert(0, str(d))

    last_err = None
    for mod_name in [
        "models",
        "downstream_examples.models",
        "downstream_examples.ar_segmentation.models",
        "active_region_segmentation.models",
    ]:
        try:
            module = __import__(mod_name, fromlist=["HelioSpectformer2D"])
            return module.HelioSpectformer2D
        except Exception as e:
            last_err = e

    raise ImportError(
        "Could not import HelioSpectformer2D from your downstream models.py. "
        "Adjust _import_helio_spectformer_2d() to match your repo layout."
    ) from last_err


HelioSpectformer2D = _import_helio_spectformer_2d()


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce_mask_to_chw(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    mask = mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # [1, H, W]
    elif mask.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported AR mask shape: {tuple(mask.shape)}")

    if mask.max().item() > 1.0:
        mask = mask / 255.0

    return mask


def _extract_model_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")

def _find_single_checkpoint_file(model_dir: Path) -> Path:
    ckpt_files = sorted(model_dir.rglob("*.pt")) + sorted(model_dir.rglob("*.pth"))
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No .pt or .pth file found under: {model_dir}")
    if len(ckpt_files) > 1:
        raise RuntimeError(
            f"Found multiple checkpoint files under {model_dir}: {ckpt_files}. "
            "Pass --init-ckpt explicitly to disambiguate."
        )
    return ckpt_files[0]

# def _apply_authors_peft_lora(model: torch.nn.Module) -> torch.nn.Module:
#     """
#     Mirror downstream_examples/ar_segmentation/finetune.py and infer.py.
#     """
#     lora_config = {
#         "r": 32,
#         "lora_alpha": 64,
#         "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
#         "lora_dropout": 0.1,
#         "bias": "none",
#     }

#     peft_config = LoraConfig(
#         r=lora_config["r"],
#         lora_alpha=lora_config["lora_alpha"],
#         target_modules=lora_config["target_modules"],
#         lora_dropout=lora_config["lora_dropout"],
#         bias=lora_config["bias"],
#     )
#     return get_peft_model(model, peft_config)

def _apply_peft_lora(
    model: torch.nn.Module,
    *,
    r: int,
    lora_alpha: int,
    target_modules: list[str],
    lora_dropout: float,
    bias: str,
) -> torch.nn.Module:
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    return get_peft_model(model, peft_config)

def _load_partial_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    logger: logging.Logger | None = None,
) -> tuple[int, list[str], list[str]]:
    payload = load_checkpoint(ckpt_path, map_location=map_location)
    state = _extract_model_state(payload)
    model_state = model.state_dict()

    filtered = {}
    skipped = []

    for k, v in state.items():
        if k not in model_state:
            skipped.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped.append(k)
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    step = 0
    if isinstance(payload, dict):
        step = int(payload.get("step", 0))

    if logger is not None:
        logger.info(
            "Loaded partial checkpoint from %s | matched=%d skipped=%d missing=%d unexpected=%d",
            str(ckpt_path),
            len(filtered),
            len(skipped),
            len(missing),
            len(unexpected),
        )

    return step, skipped, list(missing)

def _load_authors_baseline_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    logger: logging.Logger | None = None,
) -> None:
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        model_state = checkpoint

    if any(key.startswith("module.") for key in model_state.keys()):
        model_state = {
            key.replace("module.", ""): value
            for key, value in model_state.items()
        }

    model.load_state_dict(model_state, strict=True)

    if logger is not None:
        logger.info("Loaded authors' baseline checkpoint strictly from %s", str(ckpt_path))

def _freeze_backbone_except_head(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("unembed"):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    return trainable

def _freeze_for_lora_finetuning(model: torch.nn.Module) -> int:
    """
    Freeze everything except:
      - LoRA adapter params
      - task-specific decoder / head params
    """
    trainable = 0
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_head = name.startswith("unembed")

        if is_lora or is_head:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False

    return trainable


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _normalize_ar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns and "timestep" in df.columns:
        df = df.rename(columns={"timestep": "timestamp"})
    if "file_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "file_path"})

    required = {"timestamp", "file_path", "present"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"AR CSV missing required columns {sorted(missing)}. Found: {list(df.columns)}")

    return df


def _filter_df_to_date(df: pd.DataFrame, date_yyyymmdd: str, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    keep = out[ts_col].dt.strftime("%Y%m%d") == date_yyyymmdd
    out = out.loc[keep].copy()
    out.sort_values(ts_col, inplace=True)
    return out


def _write_temp_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _resolve_include_globs(date: str, hour_prefix: str, includes: list[str] | None) -> tuple[str, ...]:
    include_globs: list[str] = []

    if hour_prefix:
        hp = hour_prefix.strip()
        if len(hp) != 2 or not hp.isdigit() or not (0 <= int(hp) <= 23):
            raise ValueError(f"--hour-prefix must be between 00 and 23, got: {hour_prefix}")
        include_globs.append(f"{date}_{hp}*.nc")

    if includes:
        include_globs.extend([s.strip() for s in includes if s.strip()])

    return tuple(include_globs)


def _filter_sdo_index_csv(index_csv: str | Path, date_yyyymmdd: str, out_csv: Path) -> str:
    df = pd.read_csv(index_csv)
    ts_col = "timestep" if "timestep" in df.columns else "timestamp"
    if ts_col not in df.columns:
        raise ValueError(f"Could not find timestep/timestamp column in SDO index CSV: {index_csv}")
    filtered = _filter_df_to_date(df, date_yyyymmdd, ts_col)
    if len(filtered) == 0:
        raise RuntimeError(f"No SDO rows remain in {index_csv} for restrict-date={date_yyyymmdd}")
    return _write_temp_csv(filtered, out_csv)


def _filter_ar_split_csv(csv_path: str | Path, date_yyyymmdd: str, out_csv: Path) -> str:
    df = pd.read_csv(csv_path)
    df = _normalize_ar_columns(df)
    filtered = _filter_df_to_date(df, date_yyyymmdd, "timestamp")
    if len(filtered) == 0:
        raise RuntimeError(f"No AR rows remain in {csv_path} for restrict-date={date_yyyymmdd}")
    return _write_temp_csv(filtered, out_csv)


def _build_or_filter_sdo_index(
    *,
    sdo_data_root: str,
    existing_index_csv: str | None,
    effective_date: str,
    tmp_dir: Path,
) -> str:
    if existing_index_csv:
        return _filter_sdo_index_csv(
            existing_index_csv,
            effective_date,
            tmp_dir / "sdo_index_restricted.csv",
        )

    full_index_csv = tmp_dir / "sdo_index_full_local.csv"
    build_index_csv(
        IndexParams(
            dataset_root=Path(sdo_data_root),
            output_csv=full_index_csv,
            relative_paths=True,
            skip_unparseable=True,
            sort_by_time=True,
        )
    )
    return _filter_sdo_index_csv(
        full_index_csv,
        effective_date,
        tmp_dir / "sdo_index_restricted.csv",
    )


def _make_same_day_ar_splits(
    *,
    csv_paths: list[str],
    date_yyyymmdd: str,
    out_dir: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = _normalize_ar_columns(df)
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    full_df = _filter_df_to_date(full_df, date_yyyymmdd, "timestamp")
    full_df = full_df.loc[full_df["present"] == 1].copy()
    full_df = full_df.drop_duplicates(subset=["timestamp", "file_path"]).sort_values("timestamp")

    n = len(full_df)
    if n == 0:
        raise RuntimeError(f"No AR rows remain for restrict-date={date_yyyymmdd}")

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val

    if n >= 3 and n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_df = full_df.iloc[:n_train].copy()
    val_df = full_df.iloc[n_train:n_train + n_val].copy()
    test_df = full_df.iloc[n_train + n_val:].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "validation.csv"
    test_csv = out_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return [str(train_csv)], [str(val_csv)], [str(test_csv)]


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class ARSegDataset(HelioNetCDFDataset):
    def __init__(
        self,
        *,
        sdo_data_root_path: str,
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers: dict[str, Any] | None,
        channels: list[str],
        phase: str,
        ds_ar_index_paths: list[str],
        mask_root: str,
        num_mask_aia_channels: int = 0,
        use_latitude_in_learned_flow: bool = False,
        pooling: str | None = None,
        random_vert_flip: bool = False,
    ) -> None:
        super().__init__(
            sdo_data_root_path=sdo_data_root_path,
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            channels=channels,
            drop_hmi_probability=0.0,
            num_mask_aia_channels=num_mask_aia_channels,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            scalers=scalers,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
        )

        self.mask_root = Path(mask_root)

        all_data = [_normalize_ar_columns(pd.read_csv(p)) for p in ds_ar_index_paths]
        self.ar_index = pd.concat(all_data, ignore_index=True)
        self.ar_index = self.ar_index.loc[self.ar_index["present"] == 1, :].copy()
        self.ar_index["timestamp"] = pd.to_datetime(self.ar_index["timestamp"]).values.astype("datetime64[ns]")
        self.ar_index.sort_values("timestamp", inplace=True)

        helio_valid = pd.DataFrame({"valid_indices": self.valid_indices}).sort_values("valid_indices")
        merged = pd.merge(
            self.ar_index,
            helio_valid,
            how="inner",
            left_on="timestamp",
            right_on="valid_indices",
        )

        self.ar_valid_indices = merged.copy()
        self.valid_indices = [pd.Timestamp(x) for x in self.ar_valid_indices["valid_indices"]]
        self.adjusted_length = len(self.valid_indices)

        if self.adjusted_length == 0:
            raise RuntimeError(
                "ARSegDataset has length 0 after matching AR CSV timestamps with Helio valid indices. "
                "Check your SDO index, time settings, and AR split CSVs."
            )

        self.ar_valid_indices.set_index("valid_indices", inplace=True)

    def __len__(self) -> int:
        return self.adjusted_length

    def __getitem__(self, idx: int):
        base_dictionary, metadata = super().__getitem__(idx=idx)

        timestep = self.valid_indices[idx]
        row = self.ar_valid_indices.loc[timestep]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        raw_file_path = Path(str(row["file_path"]))
        mask_path = self.mask_root / raw_file_path

        if not mask_path.exists():
            parts = raw_file_path.parts
            if len(parts) > 0 and parts[0] == "data":
                alt_path = self.mask_root / Path(*parts[1:])
                if alt_path.exists():
                    mask_path = alt_path

        if not mask_path.exists():
            raise FileNotFoundError(
                f"AR mask file not found. Tried: "
                f"{self.mask_root / raw_file_path} and "
                f"{self.mask_root / Path(*raw_file_path.parts[1:]) if len(raw_file_path.parts) > 1 else 'N/A'}"
            )

        with h5py.File(mask_path, "r") as f:
            if "union_with_intersect" in f:
                mask = f["union_with_intersect"][...]
            else:
                keys = list(f.keys())
                if len(keys) != 1:
                    raise KeyError(
                        f"Could not find 'union_with_intersect' in {mask_path}. Available keys: {keys}"
                    )
                mask = f[keys[0]][...]

        mask = _coerce_mask_to_chw(mask)

        base_dictionary = dict(base_dictionary)
        base_dictionary["forecast"] = mask

        metadata = dict(metadata)
        metadata["mask_path"] = str(mask_path)
        return base_dictionary, metadata


# --------------------------------------------------------------------------------------
# Objective and metrics
# --------------------------------------------------------------------------------------
def _ensure_target_shape(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def dice_from_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return dice_from_probs(probs, target, eps=eps)


def dice_at_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def iou_at_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


@dataclass
class ARSegObjective:
    bce_weight: float = 1.0
    dice_loss_weight: float = 1.0
    use_pos_weight: bool = True
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

        logits = model(
            {
                "ts": batch_data["ts"],
                "time_delta_input": batch_data["time_delta_input"],
            }
        )
        target = _ensure_target_shape(logits, batch_data["forecast"])

        if self.use_pos_weight:
            num_pos = target.sum()
            num_neg = target.numel() - num_pos
            pos_weight = (num_neg / num_pos.clamp_min(1.0)).clamp(max=100.0)
            bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight,
            )
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
            print(
                "logits stats:     "
                f"mean={logits.mean().item():.6f}, "
                f"std={logits.std().item():.6f}, "
                f"min={logits.min().item():.6f}, "
                f"max={logits.max().item():.6f}"
            )
            print(
                "target stats:     "
                f"mean={target.mean().item():.6f}, "
                f"std={target.std().item():.6f}, "
                f"min={target.min().item():.6f}, "
                f"max={target.max().item():.6f}"
            )
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
        batch_data = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch_data.items()
        }

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(
                {
                    "ts": batch_data["ts"],
                    "time_delta_input": batch_data["time_delta_input"],
                }
            )
            target = _ensure_target_shape(logits, batch_data["forecast"])

            if use_pos_weight:
                num_pos = target.sum()
                num_neg = target.numel() - num_pos
                pos_weight = (num_neg / num_pos.clamp_min(1.0)).clamp(max=100.0)
                bce = F.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    pos_weight=pos_weight,
                )
            else:
                bce = F.binary_cross_entropy_with_logits(logits, target)

            soft_dice = dice_from_logits(logits, target)
            loss = bce

            preds = (torch.sigmoid(logits) >= threshold).float()
            tgt = target.float()

            inter = (preds * tgt).sum().item()
            union = (preds + tgt - preds * tgt).sum().item()
            pred_sum = preds.sum().item()
            gt_sum = tgt.sum().item()

        total_loss += float(loss.detach().cpu().item())
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
    best_metrics = None
    best_dice = -1.0

    for t in thresholds:
        metrics = evaluate_ar_seg(
            model,
            dataloader,
            device,
            threshold=float(t),
            use_pos_weight=use_pos_weight,
        )
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_threshold = float(t)
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _build_ar_model_from_surya_config(
    base_config: dict[str, Any],
    *,
    n_input_timestamps: int,
    ft_unembedding_type: str,
) -> torch.nn.Module:
    model_cfg = base_config["model"]
    data_cfg = base_config["data"]

    cfg = json.loads(json.dumps(base_config))
    cfg["model"]["ft_unembedding_type"] = ft_unembedding_type
    cfg["model"]["ft_out_chans"] = 1
    cfg["model"]["finetune"] = True

    model = HelioSpectformer2D(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=len(data_cfg["sdo_channels"]),
        embed_dim=model_cfg["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": 1,
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
        finetune=True,
        config=cfg,
    )
    return model

def _build_authors_baseline_model_from_surya_config(
    base_config: dict[str, Any],
    *,
    n_input_timestamps: int,
    ft_unembedding_type: str,
) -> torch.nn.Module:
    model_cfg = base_config["model"]
    data_cfg = base_config["data"]

    cfg = json.loads(json.dumps(base_config))

    # Match downstream_examples/ar_segmentation/finetune.py expectations
    cfg["model"]["model_type"] = "spectformer_lora"
    cfg["model"]["use_lora"] = True
    cfg["model"]["finetune"] = True
    cfg["model"]["ft_unembedding_type"] = ft_unembedding_type
    cfg["model"]["ft_out_chans"] = 1

    # Fill the names expected by finetune.py / infer.py if your base config uses different ones
    cfg["model"]["in_channels"] = len(data_cfg["sdo_channels"])
    cfg["model"]["spectral_blocks"] = model_cfg["n_spectral_blocks"]
    cfg["model"]["time_embedding"] = {
        "type": "linear",
        "time_dim": 1,
    }

    model = HelioSpectformer2D(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=cfg["model"]["in_channels"],
        embed_dim=model_cfg["embed_dim"],
        time_embedding=cfg["model"]["time_embedding"],
        depth=model_cfg["depth"],
        n_spectral_blocks=cfg["model"]["spectral_blocks"],
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
        finetune=True,
        config=cfg,
    )

    model = _apply_peft_lora(
        model,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
    )
    return model


# --------------------------------------------------------------------------------------
# Dataloaders
# --------------------------------------------------------------------------------------
def _build_ar_dataset(
    *,
    split_csvs: list[str],
    sdo_index_csv: str,
    sdo_data_root: str,
    mask_root: str,
    config: dict[str, Any],
    scalers: dict[str, Any],
    phase: str,
    n_input_timestamps: int,
) -> ARSegDataset:
    data_cfg = config["data"]

    if n_input_timestamps != 1:
        raise ValueError(
            f"AR segmentation is configured for a single same-time input, but got "
            f"n_input_timestamps={n_input_timestamps}"
        )

    used_time_deltas = [0]

    ds = ARSegDataset(
        sdo_data_root_path=sdo_data_root,
        index_path=sdo_index_csv,
        time_delta_input_minutes=used_time_deltas,
        time_delta_target_minutes=0,
        n_input_timestamps=n_input_timestamps,
        rollout_steps=0,
        scalers=scalers,
        channels=data_cfg["sdo_channels"],
        phase=phase,
        ds_ar_index_paths=split_csvs,
        mask_root=mask_root,
        num_mask_aia_channels=0,
        use_latitude_in_learned_flow=data_cfg.get("use_latitude_in_learned_flow", False),
        pooling=data_cfg.get("pooling", None),
        random_vert_flip=False,
    )
    return ds


def _build_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=None if num_workers == 0 else 2,
        collate_fn=custom_collate_fn,
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run AR segmentation downstream training/eval for Surya checkpoints.")

    # SDO image data
    parser.add_argument("--sdo-data-root", type=str, default="pretext_experiments/data/ar-seg-sdo")
    parser.add_argument(
        "--sdo-index-csv",
        type=str,
        default="",
        help="Optional existing SDO index CSV. If omitted and prepare/filter is needed, a fresh one is built locally.",
    )
    parser.add_argument("--prepare-sdo-data", action="store_true")
    parser.add_argument("--download-date", type=str, default="")
    parser.add_argument("--hour-prefix", type=str, default="")
    parser.add_argument("--include", action="append", default=None)

    # AR labels/masks
    parser.add_argument("--ar-data-root", type=str, default="pretext_experiments/data/surya-bench-ar-segmentation")
    parser.add_argument("--prepare-ar-data", action="store_true")
    parser.add_argument("--restrict-date", type=str, default="")
    parser.add_argument("--generate-ar-split", action="store_true")
    parser.add_argument("--ar-train-frac", type=float, default=0.8)
    parser.add_argument("--ar-val-frac", type=float, default=0.1)
    parser.add_argument("--ar-test-frac", type=float, default=0.1)

    # Optional explicit split CSVs. If omitted, defaults under ar-data-root are used.
    parser.add_argument("--train-csv", action="append", default=None)
    parser.add_argument("--val-csv", action="append", default=None)
    parser.add_argument("--test-csv", action="append", default=None)

    # Model / checkpoint
    parser.add_argument("--model-dir", type=str, default="pretext_experiments/data/Surya-1.0")
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="Checkpoint used to initialize the downstream AR model. "
            "If --baseline is set, this is resolved automatically unless provided explicitly.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use the authors' AR-segmentation Surya model from Hugging Face as initialization.",
    )
    parser.add_argument(
        "--baseline-model-dir",
        type=str,
        default="pretext_experiments/data/ar_segmentation_surya",
        help="Local directory for the authors' AR-segmentation Surya snapshot.",
    )
    parser.add_argument(
        "--finetune-mode",
        type=str,
        default="lora",
        choices=["full", "head_only", "lora"],
    )
    parser.add_argument("--ft-unembedding-type", type=str, default="linear", choices=["linear", "perceiver"])
    parser.add_argument("--n-input-timestamps", type=int, default=1)

    # Train
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        default=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    )
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-bias", type=str, default="none")

    # Loss / eval
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-pos-weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false")
    parser.set_defaults(use_pos_weight=True)
    parser.add_argument("--eval-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="*",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Validation thresholds to sweep when --eval-threshold < 0",
    )
    parser.add_argument("--debug-loss-once", action="store_true")

    # Misc
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-batches", type=int, default=1)
    parser.add_argument("--viz-save-dir", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--runs-dir", type=str, default="pretext_experiments/outputs/runs")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    base_seed = int(args.seed)
    rank = int(get_rank())
    set_global_seed(base_seed + rank)

    run_name = args.run_name.strip() or f"ar_seg_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = create_run_dir(args.runs_dir, run_name)

    logger = setup_surya_logger(run_dir, name="pretext_main")
    metrics_logger = JsonlLogger(Path(run_dir) / "logs" / "metrics.jsonl")

    restrict_date = args.restrict_date.strip() or None
    download_date = args.download_date.strip() or None

    if restrict_date is not None and (len(restrict_date) != 8 or not restrict_date.isdigit()):
        raise ValueError(f"--restrict-date must be YYYYMMDD, got: {restrict_date}")
    if download_date is not None and (len(download_date) != 8 or not download_date.isdigit()):
        raise ValueError(f"--download-date must be YYYYMMDD, got: {download_date}")

    effective_date = restrict_date or download_date

    # ------------------------------------------------------------------
    # Prepare SDO data if requested
    # ------------------------------------------------------------------
    if args.prepare_sdo_data:
        if effective_date is None:
            raise ValueError("--prepare-sdo-data requires either --download-date or --restrict-date.")

        include_globs = _resolve_include_globs(
            effective_date,
            args.hour_prefix,
            args.include,
        )

        out_dir = download_sdo_subset(
            DownloadParams(
                date=effective_date,
                local_root=Path(args.sdo_data_root),
                mirror_year_month_dirs=True,
                include_globs=include_globs,
            )
        )
        logger.info("Downloaded SDO data for %s into %s", effective_date, out_dir)

    # ------------------------------------------------------------------
    # Prepare AR data if requested
    # ------------------------------------------------------------------
    ar_data_root = Path(args.ar_data_root)
    if args.prepare_ar_data:
        ensure_ar_segmentation_data(
            local_dir=ar_data_root,
            restrict_date=restrict_date,
        )
        logger.info("Prepared AR segmentation labels under %s", ar_data_root)

    # Resolve split CSVs
    default_train = ar_data_root / "train.csv"
    default_val = ar_data_root / "validation.csv"
    default_test = ar_data_root / "test.csv"

    train_csvs = args.train_csv if args.train_csv else [str(default_train)]
    val_csvs = args.val_csv if args.val_csv else [str(default_val)]
    test_csvs = args.test_csv if args.test_csv else [str(default_test)]

    for p in train_csvs + val_csvs + test_csvs:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing AR split CSV: {p}")

    # Build/filter SDO index if needed
    effective_sdo_index_csv = args.sdo_index_csv.strip() or None
    effective_train_csvs = list(train_csvs)
    effective_val_csvs = list(val_csvs)
    effective_test_csvs = list(test_csvs)

    if effective_date is not None:
        tmp_dir = Path(run_dir) / "tmp_restricted"
        effective_sdo_index_csv = _build_or_filter_sdo_index(
            sdo_data_root=args.sdo_data_root,
            existing_index_csv=effective_sdo_index_csv,
            effective_date=effective_date,
            tmp_dir=tmp_dir,
        )

        if restrict_date is not None:
            if args.generate_ar_split:
                all_ar_csvs = list(dict.fromkeys(train_csvs + val_csvs + test_csvs))
                effective_train_csvs, effective_val_csvs, effective_test_csvs = _make_same_day_ar_splits(
                    csv_paths=all_ar_csvs,
                    date_yyyymmdd=restrict_date,
                    out_dir=tmp_dir / "ar_same_day_split",
                    train_frac=args.ar_train_frac,
                    val_frac=args.ar_val_frac,
                    test_frac=args.ar_test_frac,
                )
                logger.info(
                    "Generated same-day AR splits for %s | train=%s val=%s test=%s",
                    restrict_date,
                    effective_train_csvs,
                    effective_val_csvs,
                    effective_test_csvs,
                )
            else:
                effective_train_csvs = [
                    _filter_ar_split_csv(p, restrict_date, tmp_dir / f"train_{i}.csv")
                    for i, p in enumerate(train_csvs)
                ]
                effective_val_csvs = [
                    _filter_ar_split_csv(p, restrict_date, tmp_dir / f"val_{i}.csv")
                    for i, p in enumerate(val_csvs)
                ]
                effective_test_csvs = [
                    _filter_ar_split_csv(p, restrict_date, tmp_dir / f"test_{i}.csv")
                    for i, p in enumerate(test_csvs)
                ]
                logger.info("Applied restrict-date=%s to AR split CSVs and SDO index CSV.", restrict_date)

    if effective_sdo_index_csv is None:
        raise ValueError(
            "No usable SDO index CSV available. Pass --sdo-index-csv, or use "
            "--prepare-sdo-data with --download-date/--restrict-date so the script can build one."
        )

    if args.baseline:
        model_dir = Path(ensure_surya_base_model(local_dir=args.model_dir))
        baseline_model_dir = Path(
            ensure_ar_segmentation_surya_model(local_dir=args.baseline_model_dir)
        )

        if args.init_ckpt.strip():
            resolved_init_ckpt = Path(args.init_ckpt)
        else:
            resolved_init_ckpt = _find_single_checkpoint_file(baseline_model_dir)

        logger.info("Using authors' AR-segmentation baseline.")
        logger.info("Base model dir: %s", model_dir)
        logger.info("Baseline weights dir: %s", baseline_model_dir)
        logger.info("Resolved baseline init checkpoint: %s", resolved_init_ckpt)
    else:
        model_dir = Path(args.model_dir)
        if not args.init_ckpt.strip():
            raise ValueError("You must provide --init-ckpt unless --baseline is set.")
        resolved_init_ckpt = Path(args.init_ckpt)

    config = _load_yaml(model_dir / "config.yaml")
    scalers_info = _load_yaml(model_dir / "scalers.yaml")
    scalers = build_scalers(scalers_info)

    hparams = {
        "run_name": run_name,
        "task": "ar_segmentation",
        "seed": int(args.seed),
        "effective_seed": int(base_seed + rank),
        "batch_size": int(args.batch_size),
        "n_steps": int(args.n_steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "optim": args.optim,
        "n_input_timestamps": int(args.n_input_timestamps),
        "finetune_mode": args.finetune_mode,
        "ft_unembedding_type": args.ft_unembedding_type,
        "init_ckpt": str(resolved_init_ckpt),
        "baseline": bool(args.baseline),
        "sdo_data_root": args.sdo_data_root,
        "sdo_index_csv": str(effective_sdo_index_csv),
        "ar_data_root": str(ar_data_root),
        "download_date": download_date,
        "restrict_date": restrict_date,
        "train_csv": effective_train_csvs,
        "val_csv": effective_val_csvs,
        "test_csv": effective_test_csvs,
        "bce_weight": args.bce_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "use_pos_weight": args.use_pos_weight,
        "eval_threshold": args.eval_threshold,
        "threshold_sweep": [float(x) for x in args.threshold_sweep],
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_target_modules": list(args.lora_target_modules),
        "lora_dropout": float(args.lora_dropout),
        "lora_bias": str(args.lora_bias),
    }
    write_hparams(run_dir, hparams, filename="hparams.json")

    if args.baseline:
        model = _build_authors_baseline_model_from_surya_config(
            config,
            n_input_timestamps=args.n_input_timestamps,
            ft_unembedding_type=args.ft_unembedding_type,
        )
        logger.info("Built authors' LoRA AR-segmentation baseline model.")
        _load_authors_baseline_checkpoint(
            model,
            resolved_init_ckpt,
            map_location="cpu",
            logger=logger,
        )
        init_step = 0
        skipped_keys = []
        missing_keys = []
    else:
        model = _build_ar_model_from_surya_config(
            config,
            n_input_timestamps=args.n_input_timestamps,
            ft_unembedding_type=args.ft_unembedding_type,
        )
        logger.info("Built AR segmentation model.")

        init_step, skipped_keys, missing_keys = _load_partial_checkpoint(
            model,
            resolved_init_ckpt,
            map_location="cpu",
            logger=logger,
        )
        logger.info("Initialization checkpoint step: %d", init_step)

        if args.finetune_mode == "lora":
            model = _apply_peft_lora(
                model,
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                target_modules=list(args.lora_target_modules),
                lora_dropout=float(args.lora_dropout),
                bias=str(args.lora_bias),
            )
            logger.info(
                "Wrapped downstream AR model with LoRA | r=%d alpha=%d dropout=%.3f targets=%s",
                int(args.lora_r),
                int(args.lora_alpha),
                float(args.lora_dropout),
                list(args.lora_target_modules),
            )

    if args.eval_only:
        logger.info("Eval-only mode. No optimizer/training step will be used.")
    else:
        if args.finetune_mode == "head_only":
            trainable = _freeze_backbone_except_head(model)
            logger.info(
                "Using head_only mode. Trainable params: %d.",
                trainable,
            )

        elif args.finetune_mode == "lora":
            trainable = _freeze_for_lora_finetuning(model)
            logger.info(
                "Using LoRA finetuning mode. Trainable params: %d.",
                trainable,
            )

        elif args.finetune_mode == "full":
            for p in model.parameters():
                p.requires_grad = True
            logger.info(
                "Using full finetuning mode. Trainable params: %d",
                _count_trainable_params(model),
            )

        else:
            raise ValueError(f"Unsupported finetune_mode: {args.finetune_mode}")

    pin_memory = (args.device == "cuda" and torch.cuda.is_available())

    train_ds = _build_ar_dataset(
        split_csvs=effective_train_csvs,
        sdo_index_csv=str(effective_sdo_index_csv),
        sdo_data_root=args.sdo_data_root,
        mask_root=str(ar_data_root),
        config=config,
        scalers=scalers,
        phase="train",
        n_input_timestamps=args.n_input_timestamps,
    )
    val_ds = _build_ar_dataset(
        split_csvs=effective_val_csvs,
        sdo_index_csv=str(effective_sdo_index_csv),
        sdo_data_root=args.sdo_data_root,
        mask_root=str(ar_data_root),
        config=config,
        scalers=scalers,
        phase="valid",
        n_input_timestamps=args.n_input_timestamps,
    )
    test_ds = _build_ar_dataset(
        split_csvs=effective_test_csvs,
        sdo_index_csv=str(effective_sdo_index_csv),
        sdo_data_root=args.sdo_data_root,
        mask_root=str(ar_data_root),
        config=config,
        scalers=scalers,
        phase="test",
        n_input_timestamps=args.n_input_timestamps,
    )

    train_dl = _build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )
    val_dl = _build_dataloader(
        val_ds,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )
    test_dl = _build_dataloader(
        test_ds,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    logger.info("Dataset sizes | train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    objective = ARSegObjective(
        bce_weight=args.bce_weight,
        dice_loss_weight=args.dice_loss_weight,
        use_pos_weight=args.use_pos_weight,
        debug_once=args.debug_loss_once,
    )

    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(eval_device)

    if args.eval_only:
        if args.eval_threshold >= 0.0:
            chosen_threshold = float(args.eval_threshold)
            val_metrics = evaluate_ar_seg(
                model,
                val_dl,
                eval_device,
                threshold=chosen_threshold,
                use_pos_weight=args.use_pos_weight,
            )
        else:
            chosen_threshold, val_metrics = select_best_threshold(
                model,
                val_dl,
                eval_device,
                thresholds=[float(x) for x in args.threshold_sweep],
                use_pos_weight=args.use_pos_weight,
            )

        test_metrics = evaluate_ar_seg(
            model,
            test_dl,
            eval_device,
            threshold=chosen_threshold,
            use_pos_weight=args.use_pos_weight,
        )

        logger.info(
            "Eval-only | Validation | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f",
            val_metrics["loss"], val_metrics["bce"], val_metrics["soft_dice"],
            val_metrics["dice"], val_metrics["iou"], chosen_threshold,
        )
        logger.info(
            "Eval-only | Test | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f",
            test_metrics["loss"], test_metrics["bce"], test_metrics["soft_dice"],
            test_metrics["dice"], test_metrics["iou"], chosen_threshold,
        )

        eval_dir = Path(run_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        if args.visualize:
            viz_save_dir = (
                Path(args.viz_save_dir)
                if args.viz_save_dir.strip()
                else Path(run_dir) / "figures"
            )

            visualize_ar_predictions_from_dataloader(
                model=model,
                dataloader=test_dl,
                device=eval_device,
                save_dir=viz_save_dir,
                scalers=scalers,
                channels=config["data"]["sdo_channels"],
                threshold=chosen_threshold,
                max_batches=int(args.viz_batches),
            )

            logger.info("Saved AR visualizations to: %s", viz_save_dir)

        with (eval_dir / "ar_seg_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "eval_only",
                    "init_ckpt": str(resolved_init_ckpt),
                    "baseline": bool(args.baseline),
                    "download_date": download_date,
                    "restrict_date": restrict_date,
                    "threshold": chosen_threshold,
                    "val": val_metrics,
                    "test": test_metrics,
                },
                f,
                indent=2,
            )

        print("Eval-only complete.")
        return

    trainer_cfg = TrainerConfig(
        device=args.device,
        use_amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        keep_last_ckpt=True,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_mode=True,
        smoke_test=args.smoke_test,
        optim=args.optim,
    )

    trainer = Trainer(
        run_dir=run_dir,
        config=trainer_cfg,
        logger=logger,
        metrics_logger=metrics_logger,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check finetune_mode/head freezing.")

    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )

    final_ckpt = trainer.fit_n_steps(
        model=model,
        dataloader=train_dl,
        objective=objective,
        n_steps=args.n_steps,
        optimizer=optimizer,
        start_step=0,
        meta={
            "task": "ar_segmentation",
            "init_ckpt": str(resolved_init_ckpt),
            "baseline": bool(args.baseline),
            "init_step": int(init_step),
            "finetune_mode": args.finetune_mode,
            "ft_unembedding_type": args.ft_unembedding_type,
            "n_input_timestamps": int(args.n_input_timestamps),
            "download_date": download_date,
            "restrict_date": restrict_date,
            "skipped_init_keys": skipped_keys[:50],
            "missing_after_partial_load": missing_keys[:50],
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_target_modules": list(args.lora_target_modules),
            "lora_dropout": float(args.lora_dropout),
            "lora_bias": str(args.lora_bias),
        },
    )

    logger.info("Finished training. Final checkpoint: %s", final_ckpt)

    model.to(eval_device)

    if args.eval_threshold >= 0.0:
        chosen_threshold = float(args.eval_threshold)
        val_metrics = evaluate_ar_seg(
            model,
            val_dl,
            eval_device,
            threshold=chosen_threshold,
            use_pos_weight=args.use_pos_weight,
        )
    else:
        chosen_threshold, val_metrics = select_best_threshold(
            model,
            val_dl,
            eval_device,
            thresholds=[float(x) for x in args.threshold_sweep],
            use_pos_weight=args.use_pos_weight,
        )

    test_metrics = evaluate_ar_seg(
        model,
        test_dl,
        eval_device,
        threshold=chosen_threshold,
        use_pos_weight=args.use_pos_weight,
    )

    logger.info(
        "Validation | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f",
        val_metrics["loss"], val_metrics["bce"], val_metrics["soft_dice"],
        val_metrics["dice"], val_metrics["iou"], chosen_threshold,
    )
    logger.info(
        "Test | loss=%.6f bce=%.6f soft_dice=%.6f dice=%.6f iou=%.6f thr=%.3f",
        test_metrics["loss"], test_metrics["bce"], test_metrics["soft_dice"],
        test_metrics["dice"], test_metrics["iou"], chosen_threshold,
    )

    if args.visualize:
        viz_save_dir = (
            Path(args.viz_save_dir)
            if args.viz_save_dir.strip()
            else Path(run_dir) / "figures"
        )

        visualize_ar_predictions_from_dataloader(
            model=model,
            dataloader=test_dl,
            device=eval_device,
            save_dir=viz_save_dir,
            scalers=scalers,
            channels=config["data"]["sdo_channels"],
            threshold=chosen_threshold,
            max_batches=int(args.viz_batches),
        )

        logger.info("Saved AR visualizations to: %s", viz_save_dir)

    eval_dir = Path(run_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "task": "ar_segmentation",
        "init_ckpt": str(resolved_init_ckpt),
        "baseline": bool(args.baseline),
        "finetune_mode": args.finetune_mode,
        "ft_unembedding_type": args.ft_unembedding_type,
        "n_input_timestamps": int(args.n_input_timestamps),
        "download_date": download_date,
        "restrict_date": restrict_date,
        "threshold": chosen_threshold,
        "val": val_metrics,
        "test": test_metrics,
        "final_ckpt": str(final_ckpt),
    }
    with (eval_dir / "ar_seg_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    save_checkpoint(
        run_dir=run_dir,
        step=int(args.n_steps),
        model=model,
        optimizer=optimizer,
        scaler=trainer.scaler,
        meta={"alias": "best_or_final", **metrics_payload},
        filename="best.pt",
    )

    print("Run complete.")


if __name__ == "__main__":
    main()