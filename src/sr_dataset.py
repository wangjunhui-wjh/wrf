import random
from pathlib import Path

import numpy as np
import xarray as xr


def list_months(hr_dir, lr_dir):
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    months = []
    for hr_path in sorted(hr_dir.glob("*.zarr")):
        lr_path = lr_dir / hr_path.name
        if lr_path.exists():
            months.append(hr_path.name.replace(".zarr", ""))
    return months


def split_months(months, val_months=None, test_months=None, exclude_months=None):
    months = [m for m in months]
    exclude_months = set(exclude_months or [])
    test_months = set(test_months or [])
    val_months = set(val_months or [])

    filtered = [m for m in months if m not in exclude_months and m not in test_months and m not in val_months]

    if not val_months:
        if filtered:
            val_months = {filtered[-1]}
            filtered = filtered[:-1]
        else:
            val_months = set()

    train_months = [m for m in months if m not in exclude_months and m not in test_months and m not in val_months]
    return train_months, sorted(list(val_months)), sorted(list(test_months))


def open_month_dataset(hr_dir, lr_dir, month):
    hr_path = Path(hr_dir) / f"{month}.zarr"
    lr_path = Path(lr_dir) / f"{month}.zarr"
    if not hr_path.exists() or not lr_path.exists():
        raise FileNotFoundError(f"Missing hr/lr zarr for month {month}")
    ds_hr = xr.open_zarr(hr_path, consolidated=True, decode_times=False)
    ds_lr = xr.open_zarr(lr_path, consolidated=True, decode_times=False)
    return ds_hr, ds_lr


def _find_spatial_dims(var):
    dims = list(var.dims)
    if "time" in dims:
        dims.remove("time")
    if len(dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {dims}")
    return dims[0], dims[1]


class ZarrSRDataset:
    def __init__(
        self,
        hr_dir,
        lr_dir,
        months,
        input_vars,
        target_vars,
        scale=4,
        crop_size=64,
        samples_per_epoch=2000,
        input_stats=None,
        target_stats=None,
        grid_static=None,
        static_vars=None,
        static_stats=None,
        seed=42,
    ):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.months = list(months)
        self.input_vars = list(input_vars)
        self.target_vars = list(target_vars)
        self.scale = int(scale)
        self.crop_size = int(crop_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.input_stats = input_stats or {}
        self.target_stats = target_stats or {}
        self.static_vars = list(static_vars or [])
        self.static_stats = static_stats or {}
        self.rng = random.Random(seed)

        self.datasets = []
        for month in self.months:
            ds_hr, ds_lr = open_month_dataset(self.hr_dir, self.lr_dir, month)
            self.datasets.append((ds_hr, ds_lr))

        if not self.datasets:
            raise ValueError("No datasets loaded for training.")

        sample_hr = self.datasets[0][0][self.target_vars[0]]
        y_dim, x_dim = _find_spatial_dims(sample_hr)
        self.hr_y_dim = y_dim
        self.hr_x_dim = x_dim
        self.hr_shape = (sample_hr.sizes[y_dim], sample_hr.sizes[x_dim])

        sample_lr = self.datasets[0][1][self.input_vars[0]]
        y_dim, x_dim = _find_spatial_dims(sample_lr)
        self.lr_y_dim = y_dim
        self.lr_x_dim = x_dim
        self.lr_shape = (sample_lr.sizes[y_dim], sample_lr.sizes[x_dim])

        self.crop_size_lr = self.crop_size // self.scale
        if self.crop_size_lr <= 0 or self.crop_size % self.scale != 0:
            raise ValueError("crop_size must be divisible by scale.")

        self.static_lr = None
        if self.static_vars:
            if not grid_static:
                raise ValueError("grid_static is required when static_vars is provided.")
            ds_static = xr.open_zarr(Path(grid_static))
            static_arrays = []
            for name in self.static_vars:
                if name not in ds_static:
                    raise KeyError(f"Static var {name} not found in grid_static.")
                da = ds_static[name]
                y_dim, x_dim = _find_spatial_dims(da)
                hr_arr = da.values.astype(np.float32)
                # downsample to LR grid
                lr_da = da.coarsen({y_dim: self.scale, x_dim: self.scale}, boundary="trim").mean()
                lr_arr = lr_da.values.astype(np.float32)
                stats = self.static_stats.get(name)
                if stats:
                    mean = stats.get("mean", 0.0)
                    std = stats.get("std", 1.0) or 1.0
                    lr_arr = (lr_arr - mean) / std
                else:
                    mean = float(np.mean(hr_arr))
                    std = float(np.std(hr_arr)) or 1.0
                    lr_arr = (lr_arr - mean) / std
                static_arrays.append(lr_arr)
            self.static_lr = np.stack(static_arrays, axis=0)

    def __len__(self):
        return self.samples_per_epoch

    def _normalize(self, arr, var_names, stats):
        if not stats:
            return arr
        out = arr.copy()
        for i, name in enumerate(var_names):
            info = stats.get(name)
            if not info:
                continue
            mean = info.get("mean", 0.0)
            std = info.get("std", 1.0)
            if std == 0:
                std = 1.0
            out[i] = (out[i] - mean) / std
        return out

    def __getitem__(self, idx):
        ds_hr, ds_lr = self.datasets[self.rng.randrange(0, len(self.datasets))]

        time_len = ds_hr.sizes["time"]
        t = self.rng.randrange(0, time_len)

        hr_h, hr_w = self.hr_shape
        lr_h, lr_w = self.lr_shape

        max_hr_y = hr_h - self.crop_size
        max_hr_x = hr_w - self.crop_size
        hr_y = self.rng.randrange(0, max_hr_y + 1, self.scale)
        hr_x = self.rng.randrange(0, max_hr_x + 1, self.scale)

        lr_y = hr_y // self.scale
        lr_x = hr_x // self.scale

        lr_slice = {
            "time": slice(t, t + 1),
            self.lr_y_dim: slice(lr_y, lr_y + self.crop_size_lr),
            self.lr_x_dim: slice(lr_x, lr_x + self.crop_size_lr),
        }
        hr_slice = {
            "time": slice(t, t + 1),
            self.hr_y_dim: slice(hr_y, hr_y + self.crop_size),
            self.hr_x_dim: slice(hr_x, hr_x + self.crop_size),
        }

        lr_arr = ds_lr[self.input_vars].isel(lr_slice).to_array().values.squeeze(1)
        hr_arr = ds_hr[self.target_vars].isel(hr_slice).to_array().values.squeeze(1)

        lr_arr = self._normalize(lr_arr, self.input_vars, self.input_stats)
        hr_arr = self._normalize(hr_arr, self.target_vars, self.target_stats)

        if self.static_lr is not None:
            static_patch = self.static_lr[
                :,
                lr_y:lr_y + self.crop_size_lr,
                lr_x:lr_x + self.crop_size_lr,
            ]
            lr_arr = np.concatenate([lr_arr, static_patch], axis=0)

        return lr_arr.astype(np.float32), hr_arr.astype(np.float32)
