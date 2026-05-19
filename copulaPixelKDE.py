import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
from pathlib import Path
from scipy.stats import gaussian_kde, rankdata, norm
from scipy.stats import multivariate_normal
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import pyvinecopulib as pv

from config import STATS_PATH, SPLITS_PATH, PATHS_PATH, device, STAGE_SIZES
from Models.generator import Generator

# --- settings ---
PIXEL      = (60, 100)          # (row, col) — centre pixel
CHECKPOINT = Path("Resources/FakeData/20260506 - 90/stage6.pt")
LAYER_NUM  = 6
STATS_PATH = Path("Resources/FakeData/20260506 - 90/stats_90.pt")
SPLITS_PATH = Path("Resources/FakeData/20260506 - 90/splits_90.pt")

# --- load real data ---
solar_test  = np.load("Resources/FakeData/20260506 - 90/solar_test.npy")    # (M, 121, 201)
wind_test   = np.load("Resources/FakeData/20260506 - 90/wind_test.npy")
solar_train = np.load("Resources/FakeData/20260506 - 90/solar_synthetic.npy")
wind_train  = np.load("Resources/FakeData/20260506 - 90/wind_synthetic.npy")

# --- extract lon/lat from original NetCDF files ---
paths = torch.load(PATHS_PATH, weights_only=False)
ds = xr.open_mfdataset(paths["solarPaths"], combine="by_coords")
try:
    lat = ds["Solar Energy Potential"].coords["latitude"].values   # (121,)
    lon = ds["Solar Energy Potential"].coords["longitude"].values  # (201,)
except KeyError:
    # fallback: try short names
    lat = ds["Solar Energy Potential"].coords["lat"].values
    lon = ds["Solar Energy Potential"].coords["lon"].values
ds.close()

# --- extract pixel time series ---
r, c = PIXEL
real_s = solar_test[:, r, c]
real_w = wind_test[:,  r, c]

# --- fit pixel-level Gaussian copula on training data ---
train_s = solar_train[:, r, c]
train_w = wind_train[:,  r, c]

def fit_and_sample_pixel_copula(xA, xB, n_samples):
    n   = len(xA)
    eps = 1e-6
    uA  = rankdata(xA) / (n + 1)
    uB  = rankdata(xB) / (n + 1)
    zA  = norm.ppf(np.clip(uA, eps, 1 - eps))
    zB  = norm.ppf(np.clip(uB, eps, 1 - eps))
    rho = np.corrcoef(zA, zB)[0, 1]
    cov = np.array([[1, rho], [rho, 1]])
    z_samples = multivariate_normal(mean=[0, 0], cov=cov).rvs(n_samples)
    u_samples = norm.cdf(z_samples)

    def invert_ecdf(u_new, x_ref):
        q = np.sort(x_ref)
        i = (u_new * len(q)).astype(int).clip(0, len(q) - 1)
        return q[i]

    return invert_ecdf(u_samples[:, 0], xA), invert_ecdf(u_samples[:, 1], xB)


def fit_and_sample_pixel_tcopula(xA, xB, n_samples):
    # PIT to pseudo-observations (same idea as your rankdata, but library-handled)
    data = np.column_stack([xA, xB])
    u = pv.to_pseudo_obs(data)  # shape (n, 2), values in (0,1)

    # Fit t-copula by MLE
    controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.student])
    cop = pv.Bicop(family=pv.BicopFamily.student)
    cop.fit(data=u, controls=controls)
    
    print(cop)  # prints rho and nu — keep this, it's informative
    # e.g. "Student, parameters = 0.42  8.3"
    # nu=8.3 means meaningful tail dependence; nu>30 ~ Gaussian

    # Sample and invert marginals
    u_samples = cop.simulate(n=n_samples)  # shape (n_samples, 2), uniform

    def invert_ecdf(u_new, x_ref):
        q = np.sort(x_ref)
        i = (u_new * len(q)).astype(int).clip(0, len(q) - 1)
        return q[i]

    return invert_ecdf(u_samples[:, 0], xA), invert_ecdf(u_samples[:, 1], xB)



"""Gaussian Copula"""
# cop_s, cop_w = fit_and_sample_pixel_copula(train_s, train_w, len(real_s))

"""t-Gaussian Copula"""
cop_s, cop_w = fit_and_sample_pixel_tcopula(train_s, train_w, len(real_s))


# --- generate StyleGAN samples, extract same pixel ---
stats = torch.load(STATS_PATH, weights_only=False)
g = Generator().to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
g.load_state_dict(ckpt["g_running"])
g.eval()

chunks = []
with torch.no_grad():
    for i in range(0, len(real_s), 32):
        z = torch.randn(min(32, len(real_s) - i), 512, device=device)
        chunks.append(g(z, layer_num=LAYER_NUM, alpha=1.0).cpu())
fake = torch.cat(chunks, dim=0).numpy()

def denorm(arr, key):
    return (arr + 1) / 2 * (stats[f"{key}Max"] - stats[f"{key}Min"]) + stats[f"{key}Min"]

gan_s = denorm(fake[:, 0, r, c], "solar")
gan_w = denorm(fake[:, 1, r, c], "wind")

# --- 2D KDE ---
def kde2d(x, y, n=80j):
    xmn, xmx = min(x.min(), real_s.min()), max(x.max(), real_s.max())
    ymn, ymx = min(y.min(), real_w.min()), max(y.max(), real_w.max())
    xx, yy   = np.mgrid[xmn:xmx:n, ymn:ymx:n]
    pos      = np.vstack([xx.ravel(), yy.ravel()])
    zz       = gaussian_kde(np.vstack([x, y]))(pos).reshape(xx.shape)
    return xx, yy, zz

xx, yy, zr = kde2d(real_s, real_w)
xx, yy, zc = kde2d(cop_s,  cop_w)
xx, yy, zg = kde2d(gan_s,  gan_w)

# --- spatial map setup ---
solar_map = solar_test.mean(axis=0)   # (121, 201) — mean solar as background

pix_lon = lon[c]
pix_lat = lat[r]
dlon    = abs(lon[1] - lon[0])
dlat    = abs(lat[1] - lat[0])

# --- plot ---
fig, (ax_map, ax_kde) = plt.subplots(1, 2, figsize=(16, 6))

# -- left: spatial map with highlighted pixel --
im = ax_map.pcolormesh(lon, lat, solar_map, cmap="YlOrRd", shading="auto")
plt.colorbar(im, ax=ax_map, label="Mean Solar Irradiance")

rect = Rectangle(
    (pix_lon - dlon / 2, pix_lat - dlat / 2),
    dlon, dlat,
    linewidth=2, edgecolor="cyan", facecolor="none", zorder=5
)
ax_map.add_patch(rect)
ax_map.plot(pix_lon, pix_lat, "c+", markersize=10, markeredgewidth=2, zorder=6)

ax_map.set_xlabel("Longitude")
ax_map.set_ylabel("Latitude")
ax_map.set_title(f"Mean Solar Irradiance\nSelected pixel ({r}, {c}) — "
                 f"lon={pix_lon:.2f}°, lat={pix_lat:.2f}°")

# -- right: KDE comparison --
ax_kde.contourf(xx, yy, zr, levels=8, cmap="Blues",  alpha=0.7)
ax_kde.contour( xx, yy, zc, levels=8, colors="tomato",  linewidths=1.5)
ax_kde.contour( xx, yy, zg, levels=8, colors="green",   linewidths=1.5, linestyles="dashed")

ax_kde.set_xlabel(f"Solar at pixel ({r}, {c})")
ax_kde.set_ylabel(f"Wind at pixel ({r}, {c})")
ax_kde.set_title(f"Pixel-level Gaussian Copula vs StyleGAN — pixel ({r}, {c})\n"
                 f"ρ real={np.corrcoef(real_s, real_w)[0,1]:.2f}  "
                 f"cop={np.corrcoef(cop_s,  cop_w)[0,1]:.2f}  "
                 f"gan={np.corrcoef(gan_s,  gan_w)[0,1]:.2f}")

ax_kde.legend(handles=[
    Patch(facecolor="steelblue", alpha=0.7, label="Real"),
    Line2D([0], [0], color="tomato", linewidth=1.5, label="Pixel Copula"),
    Line2D([0], [0], color="green",  linewidth=1.5, linestyle="dashed", label="StyleGAN"),
])

plt.tight_layout()
plt.savefig("Resources/FakeData/20260506 - 90/pixel_tcopula_kde_60_100.png", dpi=150)
plt.close()
print("Saved pixel_tcopula_kde.png")