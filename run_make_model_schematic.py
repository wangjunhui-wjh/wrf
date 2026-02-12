from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, text, fc="#f4f6f8", ec="#4a4a4a", fs=9, lw=1.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def _arrow(ax, x0, y0, x1, y1, lw=1.2):
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color="#4a4a4a",
    )
    ax.add_patch(arrow)


def make_figure(out_path: Path):
    fig = plt.figure(figsize=(14, 7), dpi=180)

    # ===== Left panel: four methods overview =====
    ax1 = fig.add_axes([0.04, 0.08, 0.44, 0.84])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.text(0.0, 0.98, "A) Downscaling pipelines (LR -> HR)", fontsize=12, weight="bold", va="top")

    _box(ax1, 0.03, 0.80, 0.28, 0.12, "LR WRF inputs\n(U10, V10, T2, PSFC, PBLH)", fc="#e8f1ff")
    _box(ax1, 0.03, 0.62, 0.28, 0.12, "Optional static inputs\n(HGT, NDVI, Land cover)", fc="#eef9ea")
    _arrow(ax1, 0.17, 0.80, 0.17, 0.74)
    _arrow(ax1, 0.17, 0.62, 0.17, 0.56)

    _box(ax1, 0.03, 0.44, 0.28, 0.12, "Input tensor", fc="#f8f8f8")
    _arrow(ax1, 0.31, 0.50, 0.42, 0.50)

    _box(ax1, 0.42, 0.74, 0.22, 0.10, "Bicubic", fc="#ffece6")
    _box(ax1, 0.42, 0.58, 0.22, 0.10, "ESPCN", fc="#ffece6")
    _box(ax1, 0.42, 0.42, 0.22, 0.10, "U-Net", fc="#ffece6")
    _box(ax1, 0.42, 0.26, 0.22, 0.10, "PI-WindSR", fc="#ffece6")

    _arrow(ax1, 0.53, 0.74, 0.53, 0.68)
    _arrow(ax1, 0.53, 0.58, 0.53, 0.52)
    _arrow(ax1, 0.53, 0.42, 0.53, 0.36)

    _box(ax1, 0.73, 0.74, 0.23, 0.10, "HR U10/V10 prediction", fc="#fff9e6")
    _box(ax1, 0.73, 0.58, 0.23, 0.10, "HR U10/V10 prediction", fc="#fff9e6")
    _box(ax1, 0.73, 0.42, 0.23, 0.10, "HR U10/V10 prediction", fc="#fff9e6")
    _box(ax1, 0.73, 0.26, 0.23, 0.10, "HR U10/V10 prediction", fc="#fff9e6")

    _arrow(ax1, 0.64, 0.79, 0.73, 0.79)
    _arrow(ax1, 0.64, 0.63, 0.73, 0.63)
    _arrow(ax1, 0.64, 0.47, 0.73, 0.47)
    _arrow(ax1, 0.64, 0.31, 0.73, 0.31)

    _box(
        ax1,
        0.03,
        0.03,
        0.93,
        0.17,
        "Evaluation outputs:\n"
        "1) WRF-HR grid metrics (RMSE / MAE / Bias / R)\n"
        "2) Station validation and MOS correction\n"
        "3) Case maps, transects, PSD, and divergence",
        fc="#f5f5f5",
        fs=8.5,
    )

    # ===== Right panel: PI-WindSR training details =====
    ax2 = fig.add_axes([0.52, 0.08, 0.44, 0.84])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(0.0, 0.98, "B) PI-WindSR architecture and losses", fontsize=12, weight="bold", va="top")

    _box(ax2, 0.02, 0.78, 0.22, 0.12, "LR input tensor", fc="#e8f1ff")
    _box(ax2, 0.31, 0.78, 0.22, 0.12, "Encoder\n(conv + downsample)", fc="#f0ecff")
    _box(ax2, 0.60, 0.78, 0.22, 0.12, "Bottleneck\n+ CBAM", fc="#f0ecff")
    _box(ax2, 0.31, 0.57, 0.22, 0.12, "Decoder\n(upsample + skip)", fc="#f0ecff")
    _box(ax2, 0.60, 0.57, 0.22, 0.12, "HR prediction\n(U10, V10)", fc="#fff9e6")

    _arrow(ax2, 0.24, 0.84, 0.31, 0.84)
    _arrow(ax2, 0.53, 0.84, 0.60, 0.84)
    _arrow(ax2, 0.71, 0.78, 0.42, 0.69)
    _arrow(ax2, 0.53, 0.63, 0.60, 0.63)

    _box(ax2, 0.05, 0.37, 0.26, 0.11, "L_rec\n(Charbonnier on U10/V10)", fc="#eef9ea")
    _box(ax2, 0.37, 0.37, 0.26, 0.11, "L_div\n(divergence penalty)", fc="#eef9ea")
    _box(ax2, 0.22, 0.22, 0.54, 0.10, "L_total = L_rec + lambda * L_div", fc="#eef9ea")

    # Route arrows around text to avoid overlap
    _arrow(ax2, 0.68, 0.57, 0.18, 0.48)
    _arrow(ax2, 0.74, 0.57, 0.50, 0.48)
    _arrow(ax2, 0.18, 0.37, 0.38, 0.32)
    _arrow(ax2, 0.50, 0.37, 0.60, 0.32)

    _box(ax2, 0.02, 0.04, 0.94, 0.14,
         "Training protocol:\n"
         "- Patch-based training, AdamW, 50 epochs\n"
         "- Warm-up on reconstruction loss, then activate physics term\n"
         "- Variants: baseline / divergence-only / residual / high-pass curriculum",
         fc="#f5f5f5", fs=9)
    # No figure number in-panel; numbering is handled by manuscript.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    out_path = Path("processed_data/summary/paper_figs_cn/图2_模型结构示意/figure2_model_schematic.png")
    make_figure(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
