from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from surya.datasets.helio import HelioNetCDFDataset

from pretext_experiments.pretext.data.utils import IndexParams, build_index_csv


def coerce_mask_to_chw(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
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


def normalize_ar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" not in df.columns and "timestep" in df.columns:
        df = df.rename(columns={"timestep": "timestamp"})
    if "file_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "file_path"})

    required = {"timestamp", "file_path", "present"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"AR CSV missing required columns {sorted(missing)}. Found: {list(df.columns)}"
        )

    return df


def filter_df_to_date(df: pd.DataFrame, date_yyyymmdd: str, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    keep = out[ts_col].dt.strftime("%Y%m%d") == date_yyyymmdd
    out = out.loc[keep].copy()
    out.sort_values(ts_col, inplace=True)
    return out


def write_temp_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)

def filter_sdo_index_csv(index_csv: str | Path, date_yyyymmdd: str, out_csv: Path) -> str:
    df = pd.read_csv(index_csv)
    ts_col = "timestep" if "timestep" in df.columns else "timestamp"
    if ts_col not in df.columns:
        raise ValueError(
            f"Could not find timestep/timestamp column in SDO index CSV: {index_csv}"
        )

    filtered = filter_df_to_date(df, date_yyyymmdd, ts_col)
    if len(filtered) == 0:
        raise RuntimeError(
            f"No SDO rows remain in {index_csv} for restrict-date={date_yyyymmdd}"
        )

    return write_temp_csv(filtered, out_csv)


def build_or_filter_sdo_index(
    *,
    sdo_data_root: str,
    existing_index_csv: str | None,
    effective_date: str,
    tmp_dir: Path,
) -> str:
    if existing_index_csv:
        return filter_sdo_index_csv(
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

    return filter_sdo_index_csv(
        full_index_csv,
        effective_date,
        tmp_dir / "sdo_index_restricted.csv",
    )


def filter_ar_split_csv(csv_path: str | Path, date_yyyymmdd: str, out_csv: Path) -> str:
    """
    Keep this helper for cases where the caller explicitly wants to filter a single CSV.
    It is *not* the right primitive for same-day restricted train/val/test creation.
    """
    df = pd.read_csv(csv_path)
    df = normalize_ar_columns(df)
    filtered = filter_df_to_date(df, date_yyyymmdd, "timestamp")
    if len(filtered) == 0:
        raise RuntimeError(f"No AR rows remain in {csv_path} for restrict-date={date_yyyymmdd}")
    return write_temp_csv(filtered, out_csv)


def make_same_day_ar_splits(
    *,
    csv_paths: Iterable[str | Path],
    date_yyyymmdd: str,
    out_dir: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """
    Merge all AR CSVs, restrict to the requested date, then create fresh train/val/test
    splits for that day. This matches the intended behavior for --restrict-date.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = normalize_ar_columns(df)
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    full_df = filter_df_to_date(full_df, date_yyyymmdd, "timestamp")
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
        scalers,
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

        all_data = [normalize_ar_columns(pd.read_csv(p)) for p in ds_ar_index_paths]
        self.ar_index = pd.concat(all_data, ignore_index=True)
        self.ar_index = self.ar_index.loc[self.ar_index["present"] == 1, :].copy()
        self.ar_index["timestamp"] = pd.to_datetime(self.ar_index["timestamp"]).values.astype(
            "datetime64[ns]"
        )
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

        mask = coerce_mask_to_chw(mask)

        base_dictionary = dict(base_dictionary)
        base_dictionary["forecast"] = mask

        metadata = dict(metadata)
        metadata["mask_path"] = str(mask_path)
        return base_dictionary, metadata