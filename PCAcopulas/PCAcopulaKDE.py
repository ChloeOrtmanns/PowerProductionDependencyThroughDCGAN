"""
copulaKDE.py
------------
KDE visualizations comparing Real data, PCA Copula, and StyleGAN.

Two plots:
  A) PCA latent space KDE — projects all three into PC1/PC2 and PC1/PC3 space.
     Evaluates the copula in the space it actually models.

  B) Pixel-wise 1D KDE — for representative pixels, compares the marginal
     distribution of values across all days for all three methods.

"""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import device, STAGE_SIZES, STATS_PATH, SPLITS_PATH
from Models.generator import Generator

COPULA_OUTPUT_DIR = Path("Resources/FakeData/20260506 - Continuation LR")
CHECKPOINT_PATH   = Path("Resources/FakeData/20260506 - Continuation LR/stage6_best.pt")
LAYER_NUM         = 6
STATS_PATH        = Path("Resources/FakeData/20260506 - 90/stats_90.pt")

# ---------------------------------------------------------------------------
# Load copula outputs
# ---------------------------------------------------------------------------

def load_copula_data():
    solar_synth = np.load(COPULA_OUTPUT_DIR / "solar_synthetic.npy")
    wind_synth  = np.load(COPULA_OUTPUT_DIR / "wind_synthetic.npy")
    solar_test  = np.load(COPULA_OUTPUT_DIR / "solar_test.npy")
    wind_test   = np.load(COPULA_OUTPUT_DIR / "wind_test.npy")
    print(f"[Data] Real test : {solar_test.shape[0]} days")
    print(f"[Data] Copula    : {solar_synth.shape[0]} days")
    return solar_test, wind_test, solar_synth, wind_synth


# ---------------------------------------------------------------------------
# Generate StyleGAN fakes
# ---------------------------------------------------------------------------

def generate_stylegan(n_samples, stats):
    """Load checkpoint and generate n_samples grids in physical units."""
    print(f"[StyleGAN] Loading checkpoint from {CHECKPOINT_PATH}...")
    g = Generator().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    g.load_state_dict(ckpt["g_running"])
    g.eval()

    chunks = []
    with torch.no_grad():
        for i in range(0, n_samples, 32):
            z = torch.randn(min(32, n_samples - i), 512, device=device)
            chunks.append(g(z, layer_num=LAYER_NUM, alpha=1.0).cpu())
    fake_t = torch.cat(chunks, dim=0).numpy()   # (N, 2, 121, 201) in [-1, 1]

    def denorm(arr, key):
        return (arr + 1) / 2 * (stats[f"{key}Max"] - stats[f"{key}Min"]) + stats[f"{key}Min"]

    solar_gan = denorm(fake_t[:, 0], "solar")
    wind_gan  = denorm(fake_t[:, 1], "wind")
    print(f"[StyleGAN] Generated {solar_gan.shape[0]} samples.")
    return solar_gan, wind_gan


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def stack_flat(solar, wind):
    N = solar.shape[0]
    return np.concatenate([solar.reshape(N, -1), wind.reshape(N, -1)], axis=1)


def make_kde_grid(points_list, n=80j):
    """
    points_list: list of (2, N) arrays
    Returns xx, yy, and a list of KDE grids, one per input.
    """
    x_min = min(p[0].min() for p in points_list)
    x_max = max(p[0].max() for p in points_list)
    y_min = min(p[1].min() for p in points_list)
    y_max = max(p[1].max() for p in points_list)

    xx, yy = np.mgrid[x_min:x_max:n, y_min:y_max:n]
    pos    = np.vstack([xx.ravel(), yy.ravel()])
    grids  = [gaussian_kde(p)(pos).reshape(xx.shape) for p in points_list]
    return xx, yy, grids


# ---------------------------------------------------------------------------
# Plot A: PCA latent space KDE
# ---------------------------------------------------------------------------

def plot_pca_kde(real_flat, cop_flat, gan_flat, n_components=10):
    print("[KDE-A] Fitting PCA for visualization...")
    scaler      = StandardScaler()
    real_scaled = scaler.fit_transform(real_flat)
    cop_scaled  = scaler.transform(cop_flat)
    gan_scaled  = scaler.transform(gan_flat)

    pca         = PCA(n_components=n_components)
    real_lat    = pca.fit_transform(real_scaled)
    cop_lat     = pca.transform(cop_scaled)
    gan_lat     = pca.transform(gan_scaled)

    def remove_outliers(arr, sigma=3):
        """Remove rows where any PC is more than sigma std devs from the mean."""
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        mask = np.all(np.abs(arr - mean) < sigma * std, axis=1)
        return arr[mask]

    real_lat = remove_outliers(real_lat)
    cop_lat  = remove_outliers(cop_lat)
    gan_lat  = remove_outliers(gan_lat)

    var = pca.explained_variance_ratio_ * 100
    print(f"[KDE-A] PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  PC3={var[2]:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA Latent Space KDE — Real vs PCA Copula vs StyleGAN", fontsize=13)

    for ax_idx, (da, db) in enumerate([(0, 1), (0, 2)]):
        ax = axes[ax_idx]
        rpc = real_lat[:, [da, db]].T
        cpc = cop_lat[:,  [da, db]].T
        gpc = gan_lat[:,  [da, db]].T

        xx, yy, (zr, zc, zg) = make_kde_grid([rpc, cpc, gpc])

        ax.contourf(xx, yy, zr, levels=8, cmap="Blues",   alpha=0.7)
        ax.contour( xx, yy, zc, levels=8, colors="tomato",  linewidths=1.5)
        ax.contour( xx, yy, zg, levels=8, colors="green",   linewidths=1.5,
                    linestyles="dashed")
        ax.set_xlabel(f"PC{da+1} ({var[da]:.1f}% var)")
        ax.set_ylabel(f"PC{db+1} ({var[db]:.1f}% var)")
        ax.set_title(f"PC{da+1} vs PC{db+1}")

    fig.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.7, label="Real (test set)"),
        Line2D([0], [0], color="tomato", linewidth=1.5, label="PCA Copula"),
        Line2D([0], [0], color="green",  linewidth=1.5, linestyle="dashed",
               label="StyleGAN"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    path = "Resources/FakeData/20260506 - 90/kde_A_pca_latent.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KDE-A] Saved to {path}")


# ---------------------------------------------------------------------------
# Plot B: Pixel-wise 1D KDE
# ---------------------------------------------------------------------------

def plot_pixel_kde(solar_test, wind_test,
                   solar_cop,  wind_cop,
                   solar_gan,  wind_gan):
    pixels = {
        "NW":     (100, 20),
        "NE":     (100, 180),
        "Centre": (60,  100),
        "SW":     (20,  20),
        "SE":     (20,  180),
    }

    fig, axes = plt.subplots(2, len(pixels), figsize=(4 * len(pixels), 8))
    fig.suptitle("Pixel-wise KDE — Real vs PCA Copula vs StyleGAN", fontsize=13)

    for col, (label, (row, ci)) in enumerate(pixels.items()):
        for ch, (ch_name, r_ch, c_ch, g_ch) in enumerate([
            ("Solar", solar_test, solar_cop, solar_gan),
            ("Wind",  wind_test,  wind_cop,  wind_gan),
        ]):
            ax = axes[ch, col]

            rv = r_ch[:, row, ci]
            cv = c_ch[:, row, ci]
            gv = g_ch[:, row, ci]

            x_min = min(rv.min(), cv.min(), gv.min())
            x_max = max(rv.max(), cv.max(), gv.max())
            xs    = np.linspace(x_min, x_max, 300)

            kde_r = gaussian_kde(rv)
            kde_c = gaussian_kde(cv)
            kde_g = gaussian_kde(gv)

            ax.fill_between(xs, kde_r(xs), alpha=0.3, color="steelblue")
            ax.plot(xs, kde_r(xs), color="steelblue",  linewidth=1.5)
            ax.plot(xs, kde_c(xs), color="tomato",     linewidth=1.5, linestyle="--")
            ax.plot(xs, kde_g(xs), color="green",      linewidth=1.5, linestyle=":")

            if col == 0:
                ax.set_ylabel(f"{ch_name}\nDensity", fontsize=9)
            if ch == 0:
                ax.set_title(label, fontsize=9)
            ax.set_xlabel("Production value", fontsize=8)
            ax.tick_params(labelsize=7)

            ks_cop, _ = ks_2samp(rv, cv)
            ks_gan, _ = ks_2samp(rv, gv)
            ax.text(0.97, 0.95,
                    f"KS cop={ks_cop:.3f}\nKS gan={ks_gan:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color="gray")

    fig.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.4, label="Real (test set)"),
        Line2D([0], [0], color="tomato", linewidth=1.5, linestyle="--", label="PCA Copula"),
        Line2D([0], [0], color="green",  linewidth=1.5, linestyle=":",  label="StyleGAN"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = "Resources/FakeData/20260506 - 90/kde_B_pixel.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KDE-B] Saved to {path}")

# ---------------------------------------------------------------------------
# Plot C: Native data space KDE  (summary-statistic 2D projections)
# ---------------------------------------------------------------------------

def plot_native_kde(solar_test, wind_test,
                    solar_cop,  wind_cop,
                    solar_gan,  wind_gan):
    """
    Three 2-D KDE panels in interpretable native-space projections:
      (i)  Solar mean  vs Solar std   — spread of daily solar fields
      (ii) Wind mean   vs Wind std    — spread of daily wind fields
      (iii) Solar mean vs Wind mean   — cross-variable coupling
    """
    def remove_outliers_2d(x, y, sigma=3):
        mask = (
            (np.abs(x - x.mean()) < sigma * x.std()) &
            (np.abs(y - y.mean()) < sigma * y.std())
        )
        return x[mask], y[mask]

    def day_stats(arr):
        """Return (N,) mean and std over the spatial grid for each day."""
        flat = arr.reshape(arr.shape[0], -1)
        return flat.mean(axis=1), flat.std(axis=1)

    sol_mu_r, sol_sig_r = day_stats(solar_test)
    sol_mu_c, sol_sig_c = day_stats(solar_cop)
    sol_mu_g, sol_sig_g = day_stats(solar_gan)

    win_mu_r, win_sig_r = day_stats(wind_test)
    win_mu_c, win_sig_c = day_stats(wind_cop)
    win_mu_g, win_sig_g = day_stats(wind_gan)

    panels = [
        ("Solar: mean vs std",
         sol_mu_r, sol_sig_r,
         sol_mu_c, sol_sig_c,
         sol_mu_g, sol_sig_g,
         "Daily mean solar production",
         "Daily std solar production"),
        ("Wind: mean vs std",
         win_mu_r, win_sig_r,
         win_mu_c, win_sig_c,
         win_mu_g, win_sig_g,
         "Daily mean wind production",
         "Daily std wind production"),
        ("Solar mean vs Wind mean",
         sol_mu_r, win_mu_r,
         sol_mu_c, win_mu_c,
         sol_mu_g, win_mu_g,
         "Daily mean solar production",
         "Daily mean wind production"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Native Data Space KDE — Real vs PCA Copula vs StyleGAN", fontsize=13)

    for ax, (title, xr, yr, xc, yc, xg, yg, xlabel, ylabel) in zip(axes, panels):
        xr, yr = remove_outliers_2d(xr, yr)
        xc, yc = remove_outliers_2d(xc, yc)
        xg, yg = remove_outliers_2d(xg, yg)

        rpc = np.vstack([xr, yr])
        cpc = np.vstack([xc, yc])
        gpc = np.vstack([xg, yg])

        xx, yy, (zr, zc, zg) = make_kde_grid([rpc, cpc, gpc], n=80j)

        ax.contourf(xx, yy, zr, levels=8, cmap="Blues",  alpha=0.7)
        ax.contour( xx, yy, zc, levels=8, colors="tomato",  linewidths=1.5)
        ax.contour( xx, yy, zg, levels=8, colors="green",   linewidths=1.5,
                    linestyles="dashed")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

    fig.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.7, label="Real (test set)"),
        Line2D([0], [0], color="tomato", linewidth=1.5,               label="PCA Copula"),
        Line2D([0], [0], color="green",  linewidth=1.5, linestyle="dashed", label="StyleGAN"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    path = "Resources/FakeData/20260506 - 90/kde_C_native.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KDE-C] Saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stats = torch.load(STATS_PATH, weights_only=False)

    solar_test, wind_test, solar_cop, wind_cop = load_copula_data()
    solar_gan,  wind_gan  = generate_stylegan(len(solar_test), stats)

    real_flat = stack_flat(solar_test, wind_test)
    cop_flat  = stack_flat(solar_cop,  wind_cop)
    gan_flat  = stack_flat(solar_gan,  wind_gan)

    # plot_pca_kde(real_flat, cop_flat, gan_flat)
    # plot_pixel_kde(solar_test, wind_test, solar_cop, wind_cop, solar_gan, wind_gan)
    plot_native_kde(solar_test, wind_test, solar_cop, wind_cop, solar_gan, wind_gan)

    print("\n[Done] Both KDE plots saved to Resources/CopulaOutput/")