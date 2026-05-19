"""
compareMeanMaps.py
------------------
Plots a 3-row spatial comparison:
  Row 1: Real average (solar + wind)
  Row 2: Copula synthetic average (solar + wind)
  Row 3: StyleGAN average (solar + wind)
  + a difference map row (synthetic - real) for both methods

Run after copulaMapGen.py:
    python compareMeanMaps.py
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import device, STAGE_SIZES, STATS_PATH
from Models.generator import Generator

COPULA_OUTPUT_DIR = Path("Resources/FakeData/20260506 - 90")
CHECKPOINT_PATH   = Path("Resources/FakeData/20260506 - 90/stage6.pt")
STATS_PATH        = Path("Resources/FakeData/20260506 - 90/stats_90.pt")
LAYER_NUM         = 6
N_SAMPLES         = 500   # how many StyleGAN samples to average over


# ---------------------------------------------------------------------------
# Load copula data
# ---------------------------------------------------------------------------

def load_copula():
    solar = np.load(COPULA_OUTPUT_DIR / "solar_synthetic.npy")  # (N, 121, 201)
    wind  = np.load(COPULA_OUTPUT_DIR / "wind_synthetic.npy")
    solar_test = np.load(COPULA_OUTPUT_DIR / "solar_test.npy")
    wind_test  = np.load(COPULA_OUTPUT_DIR / "wind_test.npy")
    return solar_test, wind_test, solar, wind


# ---------------------------------------------------------------------------
# Generate StyleGAN samples
# ---------------------------------------------------------------------------

def generate_stylegan(n_samples, stats):
    print(f"[StyleGAN] Generating {n_samples} samples...")
    g = Generator().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    g.load_state_dict(ckpt["g_running"])
    g.eval()

    chunks = []
    with torch.no_grad():
        for i in range(0, n_samples, 32):
            z = torch.randn(min(32, n_samples - i), 512, device=device)
            chunks.append(g(z, layer_num=LAYER_NUM, alpha=1.0).cpu())
    fake = torch.cat(chunks, dim=0).numpy()   # (N, 2, 121, 201) in [-1, 1]

    def denorm(arr, key):
        return (arr + 1) / 2 * (stats[f"{key}Max"] - stats[f"{key}Min"]) + stats[f"{key}Min"]

    return denorm(fake[:, 0], "solar"), denorm(fake[:, 1], "wind")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(solar_real, wind_real,
                    solar_cop,  wind_cop,
                    solar_gan,  wind_gan):

    # compute means
    mean_real_s = solar_real.mean(axis=0)
    mean_real_w = wind_real.mean(axis=0)
    mean_cop_s  = solar_cop.mean(axis=0)
    mean_cop_w  = wind_cop.mean(axis=0)
    mean_gan_s  = solar_gan.mean(axis=0)
    mean_gan_w  = wind_gan.mean(axis=0)

    # difference maps
    diff_cop_s  = mean_cop_s - mean_real_s
    diff_cop_w  = mean_cop_w - mean_real_w
    diff_gan_s  = mean_gan_s - mean_real_s
    diff_gan_w  = mean_gan_w - mean_real_w

    # shared colour limits per channel for rows 1-3
    s_vmin = min(mean_real_s.min(), mean_cop_s.min(), mean_gan_s.min())
    s_vmax = max(mean_real_s.max(), mean_cop_s.max(), mean_gan_s.max())
    w_vmin = min(mean_real_w.min(), mean_cop_w.min(), mean_gan_w.min())
    w_vmax = max(mean_real_w.max(), mean_cop_w.max(), mean_gan_w.max())

    # symmetric colour limits for difference maps
    d_s_lim = max(abs(diff_cop_s).max(), abs(diff_gan_s).max())
    d_w_lim = max(abs(diff_cop_w).max(), abs(diff_gan_w).max())

    rows = [
        ("Real (test set)",      mean_real_s, mean_real_w, None,       None,       False),
        ("PCA Copula",           mean_cop_s,  mean_cop_w,  None,       None,       False),
        ("StyleGAN",             mean_gan_s,  mean_gan_w,  None,       None,       False),
        ("Copula − Real (diff)", diff_cop_s,  diff_cop_w,  d_s_lim,    d_w_lim,    True),
        ("StyleGAN − Real (diff)", diff_gan_s, diff_gan_w, d_s_lim,   d_w_lim,    True),
    ]

    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
    fig.suptitle("Spatial Mean Comparison — Real vs PCA Copula vs StyleGAN",
                 fontsize=14, fontweight="bold")

    col_titles = ["Solar Energy Potential", "Wind Energy Potential"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=11, fontweight="bold")

    for ri, (label, s_map, w_map, d_s, d_w, is_diff) in enumerate(rows):
        for ci, (data, ch_name, d_lim, base_vmin, base_vmax) in enumerate([
            (s_map, "solar", d_s, s_vmin, s_vmax),
            (w_map, "wind",  d_w, w_vmin, w_vmax),
        ]):
            ax = axes[ri, ci]

            if is_diff:
                im = ax.imshow(data, cmap="RdBu_r", origin="lower",
                               vmin=-d_lim, vmax=d_lim)
            else:
                im = ax.imshow(data, cmap="viridis", origin="lower",
                               vmin=base_vmin, vmax=base_vmax)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if ci == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    path = COPULA_OUTPUT_DIR / "mean_map_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Done] Saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stats = torch.load(STATS_PATH, weights_only=False)

    solar_real, wind_real, solar_cop, wind_cop = load_copula()
    solar_gan,  wind_gan  = generate_stylegan(N_SAMPLES, stats)

    plot_comparison(solar_real, wind_real,
                    solar_cop,  wind_cop,
                    solar_gan,  wind_gan)