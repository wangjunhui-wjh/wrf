import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.sr_dataset import ZarrSRDataset, list_months, split_months
from src.sr_losses import CharbonnierLoss, physics_loss
from src.sr_models import build_model


def load_stats(stats_path):
    if not stats_path:
        return {}
    path = Path(stats_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_month_list(value):
    if not value:
        return []
    return [m.strip() for m in value.split(",") if m.strip()]


def _get_stats(stats, name):
    info = stats.get(name, {})
    mean = info.get("mean", 0.0)
    std = info.get("std", 1.0)
    if std == 0:
        std = 1.0
    return mean, std


def _lr_target_norm(x, input_vars, target_vars, input_stats, target_stats):
    chunks = []
    for var in target_vars:
        if var not in input_vars:
            raise ValueError(f"Target var {var} not in input_vars, required for residual/scale loss.")
        idx = input_vars.index(var)
        x_var = x[:, idx:idx + 1]
        in_mean, in_std = _get_stats(input_stats, var)
        tar_mean, tar_std = _get_stats(target_stats, var)
        x_phys = x_var * in_std + in_mean
        x_tar = (x_phys - tar_mean) / tar_std
        chunks.append(x_tar)
    return torch.cat(chunks, dim=1)


def main():
    parser = argparse.ArgumentParser(description="Train SR models on WRF LR/HR zarrs.")
    parser.add_argument("--hr-dir", default="processed_data/hr", help="HR zarr directory")
    parser.add_argument("--lr-dir", default="processed_data/lr", help="LR zarr directory")
    parser.add_argument("--model", default="unet", choices=["espcn", "unet", "physr"], help="Model type")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--input-vars", default="U10,V10,T2,PSFC,PBLH", help="LR input vars")
    parser.add_argument("--target-vars", default="U10,V10", help="HR target vars")
    parser.add_argument("--static-vars", default="", help="Static vars from grid_static (e.g., HGT)")
    parser.add_argument("--grid-static", default="processed_data/grid_static.zarr", help="Grid static zarr")
    parser.add_argument("--crop-size", type=int, default=64, help="HR crop size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs")
    parser.add_argument("--pretrain-epochs", type=int, default=5, help="Pretrain epochs (data loss only)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-pde", type=float, default=0.1, help="Physics loss weight")
    parser.add_argument("--pde-div-only", action="store_true", help="Use divergence-only physics loss (ignore momentum term)")
    parser.add_argument("--pde-residual", action="store_true", help="Apply physics loss on residual (pred - bicubic upsample)")
    parser.add_argument("--pde-bicubic", action="store_true", help="Use bicubic upsample for residual physics loss base")
    parser.add_argument("--pde-highpass-k", type=int, default=0, help="High-pass filter size for physics loss (0=off)")
    parser.add_argument("--pde-start-frac", type=float, default=None,
                        help="Start physics loss at this fraction of epochs (e.g., 0.7). Overrides pretrain-epochs.")
    parser.add_argument("--lambda-scale", type=float, default=0.0, help="Scale consistency loss weight")
    parser.add_argument("--residual", action="store_true", help="Predict residual on top of bicubic upsample")
    parser.add_argument("--samples-per-epoch", type=int, default=2000, help="Training samples per epoch")
    parser.add_argument("--val-samples", type=int, default=400, help="Validation samples per epoch")
    parser.add_argument("--stats", default="", help="JSON stats for normalization")
    parser.add_argument("--exclude-months", default="2025-09", help="Comma-separated months to exclude")
    parser.add_argument("--val-months", default="", help="Comma-separated months for validation")
    parser.add_argument("--test-months", default="", help="Comma-separated months for test (not used in training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="processed_data/checkpoints", help="Checkpoint output root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    input_vars = [v.strip() for v in args.input_vars.split(",") if v.strip()]
    target_vars = [v.strip() for v in args.target_vars.split(",") if v.strip()]
    static_vars = [v.strip() for v in args.static_vars.split(",") if v.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_months = list_months(args.hr_dir, args.lr_dir)
    train_months, val_months, test_months = split_months(
        all_months,
        val_months=parse_month_list(args.val_months),
        test_months=parse_month_list(args.test_months),
        exclude_months=parse_month_list(args.exclude_months),
    )

    print(f"Train months: {train_months}")
    print(f"Val months: {val_months}")
    print(f"Test months (held-out): {test_months}")

    if not train_months:
        raise ValueError("No training months after filtering.")

    stats = load_stats(args.stats)
    input_stats = stats.get("inputs", {})
    target_stats = stats.get("targets", {})
    static_stats = stats.get("static", {})

    train_set = ZarrSRDataset(
        args.hr_dir,
        args.lr_dir,
        train_months,
        input_vars,
        target_vars,
        scale=args.scale,
        crop_size=args.crop_size,
        samples_per_epoch=args.samples_per_epoch,
        input_stats=input_stats,
        target_stats=target_stats,
        grid_static=args.grid_static,
        static_vars=static_vars,
        static_stats=static_stats,
        seed=args.seed,
    )

    val_set = ZarrSRDataset(
        args.hr_dir,
        args.lr_dir,
        val_months or train_months[-1:],
        input_vars,
        target_vars,
        scale=args.scale,
        crop_size=args.crop_size,
        samples_per_epoch=args.val_samples,
        input_stats=input_stats,
        target_stats=target_stats,
        grid_static=args.grid_static,
        static_vars=static_vars,
        static_stats=static_stats,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = build_model(args.model, len(input_vars) + len(static_vars), len(target_vars), scale=args.scale)
    model = model.to(args.device)

    loss_data = CharbonnierLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir) / f"{args.model}_x{args.scale}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss\n", encoding="utf-8")

    best_val = float("inf")

    def _pde_weight(epoch):
        if args.pde_start_frac is not None:
            start = int(np.floor(args.epochs * args.pde_start_frac))
        else:
            start = args.pretrain_epochs
        if epoch <= start:
            return 0.0
        span = max(1, args.epochs - start)
        return args.lambda_pde * (epoch - start) / span

    def _highpass(x, k):
        if k is None or k <= 1:
            return x
        pad = k // 2
        low = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        return x - low
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        pde_weight = _pde_weight(epoch)
        for lr_arr, hr_arr in train_loader:
            if torch.is_tensor(lr_arr):
                x = lr_arr.to(args.device)
            else:
                x = torch.from_numpy(lr_arr).to(args.device)
            if torch.is_tensor(hr_arr):
                y = hr_arr.to(args.device)
            else:
                y = torch.from_numpy(hr_arr).to(args.device)

            lr_target_norm = None
            if args.residual or args.lambda_scale > 0 or args.pde_residual:
                lr_target_norm = _lr_target_norm(x, input_vars, target_vars, input_stats, target_stats)
            pred_raw = model(x)
            pred = pred_raw
            if args.residual:
                base = torch.nn.functional.interpolate(
                    lr_target_norm, scale_factor=args.scale, mode="bilinear", align_corners=False
                )
                pred = pred_raw + base
            l_data = loss_data(pred, y)

            if args.model == "physr" and pde_weight > 0:
                if args.pde_div_only:
                    p_hr = None
                elif "PSFC" in input_vars:
                    idx = input_vars.index("PSFC")
                    p_lr = x[:, idx:idx + 1, :, :]
                    p_hr = torch.nn.functional.interpolate(p_lr, scale_factor=args.scale, mode="bilinear", align_corners=False)
                else:
                    p_hr = None
                if args.pde_residual:
                    if lr_target_norm is None:
                        lr_target_norm = _lr_target_norm(x, input_vars, target_vars, input_stats, target_stats)
                    base_mode = "bicubic" if args.pde_bicubic else "bilinear"
                    base_phy = torch.nn.functional.interpolate(
                        lr_target_norm, scale_factor=args.scale, mode=base_mode, align_corners=False
                    )
                    pred_for_pde = pred - base_phy
                else:
                    pred_for_pde = pred_raw if args.residual else pred
                pred_for_pde = _highpass(pred_for_pde, args.pde_highpass_k)
                pred_u = pred_for_pde[:, 0:1]
                pred_v = pred_for_pde[:, 1:2]
                l_phy = physics_loss(pred_u, pred_v, pressure=p_hr)
                loss = l_data + pde_weight * l_phy
            else:
                loss = l_data

            if args.lambda_scale > 0:
                pred_lr = torch.nn.functional.avg_pool2d(pred, kernel_size=args.scale, stride=args.scale)
                l_scale = loss_data(pred_lr, lr_target_norm)
                loss = loss + args.lambda_scale * l_scale

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for lr_arr, hr_arr in val_loader:
                if torch.is_tensor(lr_arr):
                    x = lr_arr.to(args.device)
                else:
                    x = torch.from_numpy(lr_arr).to(args.device)
                if torch.is_tensor(hr_arr):
                    y = hr_arr.to(args.device)
                else:
                    y = torch.from_numpy(hr_arr).to(args.device)
                lr_target_norm = None
                if args.residual or args.lambda_scale > 0:
                    lr_target_norm = _lr_target_norm(x, input_vars, target_vars, input_stats, target_stats)
                pred = model(x)
                if args.residual:
                    base = torch.nn.functional.interpolate(
                        lr_target_norm, scale_factor=args.scale, mode="bilinear", align_corners=False
                    )
                    pred = pred + base
                l = loss_data(pred, y)
                if args.lambda_scale > 0:
                    pred_lr = torch.nn.functional.avg_pool2d(pred, kernel_size=args.scale, stride=args.scale)
                    l_scale = loss_data(pred_lr, lr_target_norm)
                    l = l + args.lambda_scale * l_scale
                val_losses.append(l.item())

        scheduler.step()
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss},{val_loss}\n")

        ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

    print(f"Training done. Best val: {best_val:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
