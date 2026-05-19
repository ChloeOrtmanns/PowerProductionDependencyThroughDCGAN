"""
copulaMapGen_coarse.py
----------------------
Fits a Gaussian Copula on spatially coarsened (solar, wind) grids — NO PCA.

Instead of compressing with PCA, the grid is spatially averaged into
BLOCK x BLOCK cells, reducing 121x201 → ~12x20 = 240 cells per channel,
giving 480 dimensions total. This is feasible for a Gaussian copula with
~1500 training days (>3:1 obs-to-dim ratio).

Pipeline:
1. Load + filter raw data  (same logic as main.py)
2. Load train/test indices from SPLITS_PATH  (same split as StyleGAN)
3. Coarsen grids spatially with block averaging
4. Flatten + standardise coarsened training grids
5. Fit a GaussianMultivariate copula directly (no PCA)
6. Sample synthetic coarse grids and save
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from skimage.measure import block_reduce
from sklearn.preprocessing import StandardScaler
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate

from config import PATHS_PATH, SPLITS_PATH

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SPLITS_PATH       = Path("Resources/FakeData/20260506 - 90/splits_90.pt")

# Spatial block size — averages BLOCK x BLOCK pixels into one cell
# BLOCK=10 → 121x201 becomes 12x20 = 240 cells per channel → 480 dims total
# BLOCK=20 → 121x201 becomes 6x10  =  60 cells per channel → 120 dims total
BLOCK             = 5

FILTER_PERCENTAGE = 0.90

COPULA_OUTPUT_DIR = Path(f"Resources/FakeData/20260506 - 90/CoarseCoopula{BLOCK}")
COPULA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_paths():
    paths = torch.load(PATHS_PATH, weights_only=False)
    return paths["solarPaths"], paths["windPaths"]


def load_and_filter_data():
    """Load raw NetCDF files, align, and drop low-production days."""
    solar_paths, wind_paths = _load_paths()

    ds_solar = xr.open_mfdataset(solar_paths, combine="by_coords")
    ds_wind  = xr.open_mfdataset(wind_paths,  combine="by_coords")
    ds_solar, ds_wind = xr.align(ds_solar, ds_wind)

    solar_np = ds_solar["Solar Energy Potential"].values   # (T, 121, 201)
    wind_np  = ds_wind["Wind Energy Potential"].values     # (T, 121, 201)
    times    = ds_solar["Solar Energy Potential"].time.values

    ds_solar.close()
    ds_wind.close()

    total_cells     = 121 * 201
    zero_solar      = (solar_np == 0).sum(axis=(1, 2))
    zero_wind       = (wind_np  == 0).sum(axis=(1, 2))
    valid_mask      = (
        (zero_solar / total_cells <= FILTER_PERCENTAGE) &
        (zero_wind  / total_cells <= FILTER_PERCENTAGE)
    )

    solar_np = solar_np[valid_mask]
    wind_np  = wind_np[valid_mask]
    times    = times[valid_mask]

    print(f"[Copula] Days after filtering: {len(solar_np)}")
    return solar_np, wind_np, times


# ---------------------------------------------------------------------------
# Spatial coarsening
# ---------------------------------------------------------------------------

def coarsen(solar_np, wind_np, block=BLOCK):
    """
    Spatially average BLOCK x BLOCK pixels into one cell.
    (N, 121, 201) → (N, H_c, W_c)  where H_c = ceil(121/block), W_c = ceil(201/block)
    """
    solar_c = block_reduce(solar_np, block_size=(1, block, block), func=np.mean)
    wind_c  = block_reduce(wind_np,  block_size=(1, block, block), func=np.mean)
    print(f"[Copula] Coarsened grid: {solar_np.shape[1:]}"
          f" → {solar_c.shape[1:]}  ({solar_c.shape[1] * solar_c.shape[2]} cells/channel,"
          f" {solar_c.shape[1] * solar_c.shape[2] * 2} dims total)")
    return solar_c, wind_c


# ---------------------------------------------------------------------------
# Stack / unstack helpers
# ---------------------------------------------------------------------------

def _stack(solar_c, wind_c):
    """(N, H_c, W_c) x2  →  (N, 2*H_c*W_c)"""
    N = solar_c.shape[0]
    return np.concatenate(
        [solar_c.reshape(N, -1), wind_c.reshape(N, -1)], axis=1
    )


def _unstack(flat, H_c, W_c):
    """(N, 2*H_c*W_c)  →  solar (N, H_c, W_c), wind (N, H_c, W_c)"""
    N = flat.shape[0]
    return (
        flat[:, :H_c * W_c].reshape(N, H_c, W_c),
        flat[:, H_c * W_c:].reshape(N, H_c, W_c),
    )


# ---------------------------------------------------------------------------
# Copula pipeline (no PCA)
# ---------------------------------------------------------------------------

from scipy.stats import rankdata

def fit_copula(solar_c, wind_c, train_indices):
    H_c, W_c = solar_c.shape[1], solar_c.shape[2]
    n_dims   = 2 * H_c * W_c

    full_flat  = _stack(solar_c, wind_c)
    train_flat = full_flat[train_indices]
    N = len(train_flat)

    print(f"[Copula] Fitting on {N} training days, {n_dims} dimensions (no PCA).")

    # Explicit PIT — no Gaussian marginal assumption
    train_pit = np.zeros_like(train_flat)
    for i in range(n_dims):
        ranks = rankdata(train_flat[:, i])
        train_pit[:, i] = (ranks - 0.5) / N

    col_names = [f"f{i}" for i in range(n_dims)]
    copula    = GaussianMultivariate()  # no GaussianUnivariate forced
    copula.fit(pd.DataFrame(train_pit, columns=col_names))
    print("[Copula] GaussianMultivariate copula fitted.")

    return copula, train_flat, H_c, W_c  # return train_flat for inversion


def sample_copula(copula, train_flat, H_c, W_c, n_samples):
    print(f"[Copula] Sampling {n_samples} synthetic days...")
    sampled = copula.sample(n_samples).to_numpy()  # (n_samples, n_dims) in [0,1]

    # Invert PIT via empirical quantiles — no Gaussian assumption
    flat = np.zeros_like(sampled)
    for i in range(sampled.shape[1]):
        q = np.sort(train_flat[:, i])
        idx = (sampled[:, i] * len(q)).astype(int).clip(0, len(q) - 1)
        flat[:, i] = q[idx]

    solar, wind = _unstack(flat, H_c, W_c)
    return np.clip(solar, 0, None), np.clip(wind, 0, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load data
    solar_np, wind_np, _ = load_and_filter_data()

    # 2. Load the same train/test split StyleGAN uses
    splits        = torch.load(SPLITS_PATH, weights_only=False)
    train_indices = splits["train"]
    test_indices  = splits["test"]
    print(f"[Copula] Train: {len(train_indices)} | Test: {len(test_indices)}")

    # 3. Coarsen spatially (no PCA)
    solar_c, wind_c = coarsen(solar_np, wind_np)

    # 4. Fit
    copula, train_flat, H_c, W_c = fit_copula(solar_c, wind_c, train_indices)


    # 5. Sample
    solar_synth, wind_synth = sample_copula(copula, train_flat, H_c, W_c, n_samples=len(train_indices))

    # 6. Pull out real coarse test grids for evaluation
    solar_test = solar_c[test_indices]
    wind_test  = wind_c[test_indices]

    # 7. Save
    np.save(COPULA_OUTPUT_DIR / "solar_synthetic_coarse.npy", solar_synth)
    np.save(COPULA_OUTPUT_DIR / "wind_synthetic_coarse.npy",  wind_synth)
    np.save(COPULA_OUTPUT_DIR / "solar_test_coarse.npy",      solar_test)
    np.save(COPULA_OUTPUT_DIR / "wind_test_coarse.npy",       wind_test)

    print(f"\n[Copula] Done. Files saved to {COPULA_OUTPUT_DIR}/")
    print(f"  solar_synthetic : {solar_synth.shape}")
    print(f"  wind_synthetic  : {wind_synth.shape}")
    print(f"  solar_test      : {solar_test.shape}")
    print(f"  wind_test       : {wind_test.shape}")


if __name__ == "__main__":
    main()