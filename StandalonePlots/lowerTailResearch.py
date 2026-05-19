"""
Plots the lower tail in Histogram and Scatterplot for easier research.
Is made as a standalone python file and can run (technically) the coarsed Copula,
however right now it's made to fit the PCA copula.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import torch
from pathlib import Path
from config import device
from Models.generator import Generator

COPULA_OUTPUT_DIR = Path("Resources/FakeData/20260506 - 90")

solar_test = np.load(COPULA_OUTPUT_DIR / "solar_test.npy")  # or your full real array
wind_test  = np.load(COPULA_OUTPUT_DIR / "wind_test.npy")

solar_cop = np.load(COPULA_OUTPUT_DIR / "solar_synthetic.npy")  # (N, 121, 201)
wind_cop  = np.load(COPULA_OUTPUT_DIR / "wind_synthetic.npy")



# --- generate StyleGAN samples, extract same pixel ---
stats = torch.load(COPULA_OUTPUT_DIR / "stats_90.pt", weights_only=False)
g = Generator().to(device)
ckpt = torch.load(COPULA_OUTPUT_DIR / "stage6.pt" , map_location=device, weights_only=False)
g.load_state_dict(ckpt["g_running"])
g.eval()

# Generate
chunks = []
with torch.no_grad():
    for i in range(0, len(solar_test), 32):
        z = torch.randn(min(32, len(solar_test) - i), 512, device=device)
        chunks.append(g(z, layer_num=6, alpha=1.0).cpu())
fake = torch.cat(chunks, dim=0).numpy()  # (N, 2, 121, 201)

def denorm(arr, key):
    return (arr + 1) / 2 * (stats[f"{key}Max"] - stats[f"{key}Min"]) + stats[f"{key}Min"]

# Denorm full grids instead of single pixel
solar_gan = denorm(fake[:, 0, :, :], "solar")  # (N, 121, 201)
wind_gan  = denorm(fake[:, 1, :, :], "wind")

# Flatten to get one value per day (e.g. spatial mean)
real_daily  = solar_test.mean(axis=(1,2))   # (N,) — one number per day
cop_daily   = solar_cop.mean(axis=(1,2))
gan_daily   = solar_gan.mean(axis=(1,2))

# Subsample copula to same size as real
n_real = len(real_daily)
idx = np.random.choice(len(cop_daily), size=n_real, replace=False)
cop_daily_sub = cop_daily[idx]

# Define a low-energy threshold (e.g. bottom 10%)
threshold = np.percentile(real_daily, 10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: full histogram
axes[0].hist(real_daily, bins=50, alpha=0.5, label="Real",    color="steelblue", density=True)
axes[0].hist(cop_daily_sub,  bins=50, alpha=0.5, label="Copula",  color="tomato",    density=True)
axes[0].hist(gan_daily,  bins=50, alpha=0.5, label="StyleGAN",color="green",     density=True)
axes[0].axvline(threshold, color="black", linestyle="--", label=f"10th pct ({threshold:.2f})")
axes[0].set_title("Full distribution — Daily mean solar")
axes[0].legend()

# Right: zoom into left tail only
axes[1].hist(real_daily[real_daily < threshold], bins=30, alpha=0.5, label="Real",     color="steelblue", density=True)
axes[1].hist(cop_daily_sub[cop_daily_sub   < threshold], bins=30, alpha=0.5, label="Copula",   color="tomato",    density=True)
axes[1].hist(gan_daily[gan_daily   < threshold], bins=30, alpha=0.5, label="StyleGAN", color="green",     density=True)
axes[1].set_title(f"Left tail (below 10th percentile) — Daily mean solar")
axes[1].legend()

plt.tight_layout()
plt.savefig(COPULA_OUTPUT_DIR / "low_energy_tail_solar_limited.png", dpi=150)

# Wind daily mean
real_wind_daily = wind_test.mean(axis=(1,2))
cop_wind_daily  = wind_cop.mean(axis=(1,2))
gan_wind_daily  = wind_gan.mean(axis=(1,2))

cop_wind_daily_sub = cop_wind_daily[idx]  # add this

wind_threshold = np.percentile(real_wind_daily, 10)

# --- Wind histogram ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(real_wind_daily, bins=50, alpha=0.5, label="Real",     color="steelblue", density=True)
axes[0].hist(cop_wind_daily_sub,  bins=50, alpha=0.5, label="Copula",   color="tomato",    density=True)
axes[0].hist(gan_wind_daily,  bins=50, alpha=0.5, label="StyleGAN", color="green",     density=True)
axes[0].axvline(wind_threshold, color="black", linestyle="--", label=f"10th pct ({wind_threshold:.2f})")
axes[0].set_title("Full distribution — Daily mean wind")
axes[0].legend()

axes[1].hist(real_wind_daily[real_wind_daily < wind_threshold], bins=30, alpha=0.5, label="Real",     color="steelblue", density=True)
axes[1].hist(cop_wind_daily_sub[cop_wind_daily_sub   < wind_threshold], bins=30, alpha=0.5, label="Copula",   color="tomato",    density=True)
axes[1].hist(gan_wind_daily[gan_wind_daily   < wind_threshold], bins=30, alpha=0.5, label="StyleGAN", color="green",     density=True)
axes[1].set_title("Left tail (below 10th percentile) — Daily mean wind")
axes[1].legend()

plt.tight_layout()
plt.savefig(COPULA_OUTPUT_DIR / "low_energy_tail_wind_limited.png", dpi=150)

# --- Joint low-energy days ---
real_joint = (real_daily < threshold) & (real_wind_daily < wind_threshold)
cop_joint  = (cop_daily_sub  < threshold) & (cop_wind_daily_sub  < wind_threshold)
gan_joint  = (gan_daily  < threshold) & (gan_wind_daily  < wind_threshold)

print(f"Real     — joint low days: {real_joint.sum()} ({real_joint.mean()*100:.1f}%)")
print(f"Copula   — joint low days: {cop_joint.sum()}  ({cop_joint.mean()*100:.1f}%)")
print(f"StyleGAN — joint low days: {gan_joint.sum()}  ({gan_joint.mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

pairs = [
    (axes[0], real_daily,  real_wind_daily, "Real"),
    (axes[1], cop_daily_sub,   cop_wind_daily_sub,  "Copula"),
    (axes[2], gan_daily,   gan_wind_daily,  "StyleGAN"),
]

for ax, s, w, title in pairs:
    ax.scatter(s, w, alpha=0.3, s=10, color="steelblue")
    ax.axvline(threshold,      color="red",    linestyle="--", linewidth=1, label="Solar 10th pct")
    ax.axhline(wind_threshold, color="orange", linestyle="--", linewidth=1, label="Wind 10th pct")
    
    # Highlight joint low days
    joint = (s < threshold) & (w < wind_threshold)
    ax.scatter(s[joint], w[joint], alpha=0.7, s=15, color="red", label=f"Joint low ({joint.mean()*100:.1f}%)")
    
    ax.set_xlabel("Daily mean solar")
    ax.set_ylabel("Daily mean wind")
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(COPULA_OUTPUT_DIR / "joint_low_energy_limited.png", dpi=150)

# Define zoom limits based on real data lower tail
x_lim = np.percentile(real_daily, 15)
y_lim = np.percentile(real_wind_daily, 15)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Joint Low-Energy Days — Zoomed into Lower Extremes", fontsize=13)

pairs = [
    (axes[0], real_daily,  real_wind_daily, "Real"),
    (axes[1], cop_daily_sub,   cop_wind_daily_sub,  "Copula"),
    (axes[2], gan_daily,   gan_wind_daily,  "StyleGAN"),
]

for ax, s, w, title in pairs:
    # Only plot points in the lower tail
    mask = (s < x_lim) & (w < y_lim)  # zoom window
    ax.scatter(s, w, alpha=0.3, s=10, color="steelblue")
    joint = (s < threshold) & (w < wind_threshold)
    ax.scatter(s[joint], w[joint], alpha=0.7, s=15, color="red",
               label=f"Joint low ({joint.mean()*100:.1f}%)")
    ax.axvline(threshold,      color="red",    linestyle="--", linewidth=1)
    ax.axhline(wind_threshold, color="orange", linestyle="--", linewidth=1)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel("Daily mean solar")
    ax.set_ylabel("Daily mean wind")
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(COPULA_OUTPUT_DIR / "joint_low_energy_zoomed_limited.png", dpi=150)