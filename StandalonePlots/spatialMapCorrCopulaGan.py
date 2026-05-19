import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import xarray as xr
from scipy.stats import rankdata, norm
from scipy.stats import multivariate_normal
import pyvinecopulib as pv

from config import PATHS_PATH, device
from Models.generator import Generator

# --- settings ---
CHECKPOINT  = Path("Resources/FakeData/20260506 - 90/stage6.pt")
STATS_PATH  = Path("Resources/FakeData/20260506 - 90/stats_90.pt")
SPLITS_PATH = Path("Resources/FakeData/20260506 - 90/splits_90.pt")
LAYER_NUM   = 6
N_GAN       = 500
STEP        = 10
OUT_PATH    = Path("Resources/FakeData/20260506 - 90/spatial_correlation_map_PCACopula_GAN.png")
paths = torch.load(PATHS_PATH, weights_only=False)

ds_solar = xr.open_mfdataset(paths["solarPaths"], combine="by_coords")
ds_wind  = xr.open_mfdataset(paths["windPaths"],  combine="by_coords")
ds_solar, ds_wind = xr.align(ds_solar, ds_wind)

solar_all = ds_solar["Solar Energy Potential"].values  # (T, 121, 201)
wind_all  = ds_wind["Wind Energy Potential"].values
ds_solar.close()
ds_wind.close()

# Apply same filter as copulaMapGen.py
total_cells     = 121 * 201
zero_solar      = (solar_all == 0).sum(axis=(1, 2))
zero_wind       = (wind_all  == 0).sum(axis=(1, 2))
valid_mask      = (zero_solar / total_cells <= 0.90) & (zero_wind / total_cells <= 0.90)
solar_all       = solar_all[valid_mask]
wind_all        = wind_all[valid_mask]

# Load the same split indices StyleGAN and copula used
splits        = torch.load(SPLITS_PATH, weights_only=False)
train_indices = splits["train"]
test_indices  = splits["test"]

solar_test  = solar_all[test_indices]   # real test data
wind_test   = wind_all[test_indices]
solar_train = solar_all[train_indices]  # real training data
wind_train  = wind_all[train_indices]

M = solar_test.shape[0]

# --- load lon/lat ---

ds = xr.open_mfdataset(paths["solarPaths"], combine="by_coords")
try:
    lat = ds["Solar Energy Potential"].coords["latitude"].values
    lon = ds["Solar Energy Potential"].coords["longitude"].values
except KeyError:
    lat = ds["Solar Energy Potential"].coords["lat"].values
    lon = ds["Solar Energy Potential"].coords["lon"].values
ds.close()

# --- generate GAN images ---
stats = torch.load(STATS_PATH, weights_only=False)
g = Generator().to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
g.load_state_dict(ckpt["g_running"])
g.eval()

print(f"Generating {N_GAN} GAN images...")
chunks = []
with torch.no_grad():
    for i in range(0, N_GAN, 32):
        z = torch.randn(min(32, N_GAN - i), 512, device=device)
        chunks.append(g(z, layer_num=LAYER_NUM, alpha=1.0).cpu())
fake = torch.cat(chunks, dim=0).numpy()   # (N_GAN, 2, 121, 201)

def denorm(arr, key):
    return (arr + 1) / 2 * (stats[f"{key}Max"] - stats[f"{key}Min"]) + stats[f"{key}Min"]

gan_solar = denorm(fake[:, 0], "solar")   # (N_GAN, 121, 201)
gan_wind  = denorm(fake[:, 1], "wind")

# --- coarse grid ---
rows = np.arange(0, 121, STEP)
cols = np.arange(0, 201, STEP)

rho_real   = np.full((len(rows), len(cols)), np.nan)
rho_gan    = np.full((len(rows), len(cols)), np.nan)
rho_copula = np.full((len(rows), len(cols)), np.nan)

print("Computing spatial correlation maps (this may take a few minutes)...")
for ri, r in enumerate(rows):
    for ci, c in enumerate(cols):
        # --- real ---
        s_real = solar_test[:, r, c]
        w_real = wind_test[:,  r, c]
        rho_real[ri, ci] = np.corrcoef(s_real, w_real)[0, 1]

        # --- GAN ---
        rho_gan[ri, ci] = np.corrcoef(gan_solar[:, r, c], gan_wind[:, r, c])[0, 1]

        # --- t-copula fitted on training data, evaluated on test-sized sample ---
        s_train = solar_train[:, r, c]
        w_train = wind_train[:,  r, c]

        data_u = pv.to_pseudo_obs(np.column_stack([s_train, w_train]))
        controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.student])
        cop = pv.Bicop(family=pv.BicopFamily.student)
        cop.fit(data=data_u, controls=controls)

        # sample M points from copula, invert marginals via ECDF
        u_samples = cop.simulate(n=M)

        def invert_ecdf(u_new, x_ref):
            q = np.sort(x_ref)
            i = (u_new * len(q)).astype(int).clip(0, len(q) - 1)
            return q[i]

        cop_s = invert_ecdf(u_samples[:, 0], s_train)
        cop_w = invert_ecdf(u_samples[:, 1], w_train)
        rho_copula[ri, ci] = np.corrcoef(cop_s, cop_w)[0, 1]

    print(f"  row {r}/120 done")

# --- lon/lat grids for plotting ---
lon_grid, lat_grid = np.meshgrid(lon[cols], lat[rows])

# --- errors ---
err_cop = rho_copula - rho_real   # should be ~0 everywhere (sanity check)
err_gan = rho_gan    - rho_real   # this is the interesting one

# --- plot ---
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
fig.suptitle("Pixel-wise Solar–Wind Correlation: Real vs Copula vs StyleGAN", fontsize=13)

vmin, vmax = -0.6, 0.2
err_abs = max(np.nanmax(np.abs(err_cop)), np.nanmax(np.abs(err_gan)))

def scatter_map(ax, values, vmin, vmax, cmap, title, label="ρ (solar, wind)"):
    sc = ax.scatter(lon_grid, lat_grid, c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax, s=80, marker="s", linewidths=0)
    plt.colorbar(sc, ax=ax, shrink=0.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())

scatter_map(axes[0], rho_real,
            vmin, vmax, "RdBu_r",
            f"Real Data\nρ mean={np.nanmean(rho_real):.3f}")

scatter_map(axes[1], rho_copula,
            vmin, vmax, "RdBu_r",
            f"t-Copula\nρ mean={np.nanmean(rho_copula):.3f}")

scatter_map(axes[2], rho_gan,
            vmin, vmax, "RdBu_r",
            f"StyleGAN\nρ mean={np.nanmean(rho_gan):.3f}")

scatter_map(axes[3], err_cop,
            -err_abs, err_abs, "PiYG",
            f"Copula − Real\nMAE={np.nanmean(np.abs(err_cop)):.3f}",
            label="ρ error")

scatter_map(axes[4], err_gan,
            -err_abs, err_abs, "PiYG",
            f"GAN − Real\nMAE={np.nanmean(np.abs(err_gan)):.3f}",
            label="ρ error")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved → {OUT_PATH}")
print(f"\nSummary:")
print(f"  Real    ρ: mean={np.nanmean(rho_real):.3f},    std={np.nanstd(rho_real):.3f}")
print(f"  Copula  ρ: mean={np.nanmean(rho_copula):.3f},  std={np.nanstd(rho_copula):.3f}")
print(f"  GAN     ρ: mean={np.nanmean(rho_gan):.3f},     std={np.nanstd(rho_gan):.3f}")
print(f"  Copula error — MAE={np.nanmean(np.abs(err_cop)):.3f}, "
      f"max={np.nanmax(np.abs(err_cop)):.3f}")
print(f"  GAN    error — MAE={np.nanmean(np.abs(err_gan)):.3f}, "
      f"max={np.nanmax(np.abs(err_gan)):.3f}")