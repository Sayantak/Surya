from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def _canonical_channel_name(name: str) -> str:
    s = str(name).strip().lower().replace("-", "").replace("_", "")

    if s.startswith("aia"):
        s = s[3:]
    if s in {"94", "094"}:
        return "0094"
    if s == "131":
        return "0131"
    if s == "171":
        return "0171"
    if s == "193":
        return "0193"
    if s == "211":
        return "0211"
    if s == "304":
        return "0304"
    if s == "335":
        return "0335"
    if s == "1600":
        return "1600"

    if s in {"hmi", "hmim"}:
        return "hmi"
    if s == "hmibx":
        return "hmi_bx"
    if s == "hmiby":
        return "hmi_by"
    if s == "hmibz":
        return "hmi_bz"
    if s == "hmiv":
        return "hmi_v"

    return s


def resolve_channel_indices(
    all_channels: Iterable[str],
    requested_channels: Iterable[str],
) -> list[int]:
    all_channels = list(all_channels)
    canon_to_idx = {
        _canonical_channel_name(ch): idx
        for idx, ch in enumerate(all_channels)
    }

    resolved: list[int] = []
    missing: list[str] = []

    for ch in requested_channels:
        key = _canonical_channel_name(ch)
        if key not in canon_to_idx:
            missing.append(str(ch))
        else:
            resolved.append(canon_to_idx[key])

    if missing:
        raise ValueError(
            f"Could not resolve requested channels {missing} against dataset channels {all_channels}"
        )

    out: list[int] = []
    seen: set[int] = set()
    for idx in resolved:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _clone_like(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    raise TypeError(f"Unsupported array type: {type(x)}")


def _zero_channels(
    x: torch.Tensor | np.ndarray,
    zero_channel_indices: list[int],
) -> torch.Tensor | np.ndarray:
    out = _clone_like(x)

    if out.ndim < 3:
        raise ValueError(f"Expected at least 3D tensor/array, got shape {tuple(out.shape)}")

    if len(zero_channel_indices) == 0:
        return out

    if torch.is_tensor(out):
        out[torch.as_tensor(zero_channel_indices, dtype=torch.long)] = 0
    else:
        out[zero_channel_indices] = 0
    return out


@dataclass(frozen=True)
class RandomBandMaskingSpec:
    min_masked_channels: int = 1
    max_masked_channels: int | None = None
    mask_all_timesteps: bool = True
    seed: int = 42
    include_hmi_as_target: bool = True


class RandomBandMaskingDataset(Dataset):
    """
    TerraMind-style random modality masking for Surya.

    For each sample:
      - randomly pick a subset of channels as targets
      - zero those channels in `ts`
      - keep full `forecast`
      - objective later computes loss only on masked target channels

    Expected base dataset:
      sample["ts"]       -> [C, T, H, W]
      sample["forecast"] -> [C, L, H, W] or [C, H, W]
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        min_masked_channels: int = 1,
        max_masked_channels: int | None = None,
        mask_all_timesteps: bool = True,
        seed: int = 42,
        include_hmi_as_target: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.channels = list(getattr(base_dataset, "channels"))
        self.spec = RandomBandMaskingSpec(
            min_masked_channels=int(min_masked_channels),
            max_masked_channels=max_masked_channels,
            mask_all_timesteps=bool(mask_all_timesteps),
            seed=int(seed),
            include_hmi_as_target=bool(include_hmi_as_target),
        )

        self._targetable_indices = self._build_targetable_indices()
        if len(self._targetable_indices) == 0:
            raise ValueError("No targetable channels available for random masking.")

        self._max_masked = (
            len(self._targetable_indices)
            if self.spec.max_masked_channels is None
            else int(self.spec.max_masked_channels)
        )

        if self.spec.min_masked_channels < 1:
            raise ValueError("min_masked_channels must be >= 1")

        if self._max_masked < self.spec.min_masked_channels:
            raise ValueError(
                f"max_masked_channels ({self._max_masked}) must be >= "
                f"min_masked_channels ({self.spec.min_masked_channels})"
            )

        if self._max_masked > len(self._targetable_indices):
            raise ValueError(
                f"max_masked_channels ({self._max_masked}) exceeds number of "
                f"targetable channels ({len(self._targetable_indices)})"
            )

    def _build_targetable_indices(self) -> list[int]:
        if self.spec.include_hmi_as_target:
            return list(range(len(self.channels)))

        targetable = []
        for idx, ch in enumerate(self.channels):
            if _canonical_channel_name(ch).startswith("hmi"):
                continue
            targetable.append(idx)
        return targetable

    def __len__(self) -> int:
        return len(self.base_dataset)

    @property
    def valid_indices(self):
        return getattr(self.base_dataset, "valid_indices", None)

    def transformation_inputs(self):
        return self.base_dataset.transformation_inputs()

    def _sample_target_indices(self, idx: int) -> list[int]:
        # Deterministic per-sample masking for reproducibility
        rng = np.random.default_rng(self.spec.seed + int(idx))

        n_mask = int(
            rng.integers(
                low=self.spec.min_masked_channels,
                high=self._max_masked + 1,
            )
        )

        masked = rng.choice(
            np.asarray(self._targetable_indices, dtype=np.int64),
            size=n_mask,
            replace=False,
        )
        masked = sorted(int(x) for x in masked.tolist())
        return masked

    def __getitem__(self, idx: int):
        sample, metadata = self.base_dataset[idx]

        if not isinstance(sample, dict):
            raise TypeError(f"Expected base dataset sample dict, got {type(sample)}")

        target_channel_indices = self._sample_target_indices(idx)
        target_channel_mask = np.zeros((len(self.channels),), dtype=np.float32)
        target_channel_mask[target_channel_indices] = 1.0

        ts = sample["ts"]

        if self.spec.mask_all_timesteps:
            masked_ts = _zero_channels(ts, zero_channel_indices=target_channel_indices)
        else:
            masked_ts = _clone_like(ts)
            latest_t = masked_ts.shape[1] - 1
            latest_frame = masked_ts[:, latest_t : latest_t + 1, ...]
            latest_frame = _zero_channels(latest_frame, zero_channel_indices=target_channel_indices)
            masked_ts[:, latest_t : latest_t + 1, ...] = latest_frame

        sample = dict(sample)
        sample["ts"] = masked_ts
        sample["target_channel_mask"] = target_channel_mask

        metadata = dict(metadata)
        metadata["all_channels"] = list(self.channels)
        metadata["target_channels_str"] = ",".join(self.channels[i] for i in target_channel_indices)

        return sample, metadata