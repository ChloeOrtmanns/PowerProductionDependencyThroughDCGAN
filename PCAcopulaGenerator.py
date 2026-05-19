"""
copulaMapGen.py
---------------
Standalone script: fits a PCA + Gaussian Copula on the training split and
generates synthetic (solar, wind) grids for comparison against StyleGAN.

Run independently while StyleGAN trains:
python copulaMapGen.py

Pipeline:
1. Load + filter raw data  (same logic as main.py)
2. Load train/test indices from SPLITS_PATH  (same split as StyleGAN)
3. Flatten + standardise training grids
4. Reduce dimensionality with PCA  (fit on train only)
5. Fit a GaussianMultivariate copula in PCA latent space  (fit on train only)
6. Sample synthetic grids and save to Resources/CopulaOutput/
"""
                    
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copulas.multivariate import GaussianMultivariate

from config import STATS_PATH, SPLITS_PATH, PATHS_PATH

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

COPULA_OUTPUT_DIR  = Path("Resources/FakeData/20260506 - Continuation LR")
COPULA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_PATH = Path(Path("Resources/FakeData/20260506 - 90/splits_90.pt"))

# Number of PCA components.
# Tip for your thesis: print pca.explained_variance_ratio_.cumsum()[-1] and
# try 20 / 50 / 100 to show how reconstruction quality trades off with stability.
N_COMPONENTS       = 50

# Must match the filterPercentage used in main.py
FILTER_PERCENTAGE  = 0.90


# ---------------------------------------------------------------------------
# Data loading  (copied from main.py so this file is fully self-contained)
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
                                
    total_cells      = 121 * 201
    zero_solar       = (solar_np == 0).sum(axis=(1, 2))
    zero_wind        = (wind_np  == 0).sum(axis=(1, 2))
    solar_zero_mask  = (zero_solar / total_cells <= FILTER_PERCENTAGE)
    wind_zero_mask   = (zero_wind  / total_cells <= FILTER_PERCENTAGE)
    valid_mask       = solar_zero_mask & wind_zero_mask
                        
    solar_np = solar_np[valid_mask]
    wind_np  = wind_np[valid_mask]
    times    = times[valid_mask]
                                    
    ds_solar.close()
    ds_wind.close()
                                            
    print(f"[Copula] Days after filtering: {len(solar_np)}")
    return solar_np, wind_np, times
                                                                                    
                                                                                    
# ---------------------------------------------------------------------------
# PCA + Copula pipeline
# ---------------------------------------------------------------------------
                                                                                    
def _stack(solar_np, wind_np):
    """(N,121,201) x2  ->  (N, 2*121*201)"""
    N = solar_np.shape[0]
    return np.concatenate(
        [solar_np.reshape(N, -1), wind_np.reshape(N, -1)], axis=1
    )
                        
                        
def _unstack(flat, H=121, W=201):
    """(N, 2*H*W)  ->  solar (N,H,W), wind (N,H,W)"""
    N = flat.shape[0]
    return flat[:, :H*W].reshape(N, H, W), flat[:, H*W:].reshape(N, H, W)
                                    
                                    
def fit_copula(solar_np, wind_np, train_indices, n_components=N_COMPONENTS):
    """Fit scaler -> PCA -> GaussianMultivariate on training data only."""
    print(f"[Copula] Fitting on {len(train_indices)} training days "
        f"with {n_components} PCA components.")
                    
    full_flat   = _stack(solar_np, wind_np)
    train_flat  = full_flat[train_indices]
            
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
                    
    pca          = PCA(n_components=n_components, random_state=42)
    train_latent = pca.fit_transform(train_scaled)
                            
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"[Copula] PCA explains {explained * 100:.1f}% of training variance.")
                                    
    col_names = [f"pc{i}" for i in range(n_components)]
                                        
    # Force Gaussian marginals instead of auto-detecting distributions.
    # Auto-detection caused unstable fits (warnings + Inf samples) on some
    # PCA components. Since PCA components are linear combinations of
    # standardised data, Gaussian marginals are a well-justified assumption.
    from copulas.univariate import GaussianUnivariate
    copula    = GaussianMultivariate(distribution=GaussianUnivariate)
    copula.fit(pd.DataFrame(train_latent, columns=col_names))
    print("[Copula] GaussianMultivariate copula fitted.")
                                
    return copula, pca, scaler
                                                                                                                                                                            
                                                                                                                                                                                                                    
def sample_copula(copula, pca, scaler, n_samples):
    """Sample from the copula and reconstruct full-resolution grids."""
    print(f"[Copula] Sampling {n_samples} synthetic days...")
    latent      = copula.sample(n_samples)
    flat        = scaler.inverse_transform(pca.inverse_transform(latent.values))
    solar, wind = _unstack(flat)
                    
    # Clip negatives: PCA reconstruction can produce tiny negative values.
    # Energy production is physically bounded at zero.
    return np.clip(solar, 0, None), np.clip(wind, 0, None)
                                                                                                            
def main():
    # 1. Load data
    solar_np, wind_np, _ = load_and_filter_data()
            
    # 2. Load the same train/test split StyleGAN uses
    splits        = torch.load(SPLITS_PATH, weights_only=False)
    train_indices = splits["train"]
    test_indices  = splits["test"]
    print(f"[Copula] Train: {len(train_indices)} | Test: {len(test_indices)}")
                    
    # 3. Fit
    copula, pca, scaler = fit_copula(solar_np, wind_np, train_indices)
        
    # 4. Sample (same number of days as training set)
    solar_synth, wind_synth = sample_copula(
        copula, pca, scaler, n_samples=len(train_indices)
    )
                    
    # 5. Pull out real test grids for evaluation
    solar_test = solar_np[test_indices]
    wind_test  = wind_np[test_indices]
            
    # 6. Save
    np.save(COPULA_OUTPUT_DIR / "solar_synthetic.npy", solar_synth)
    np.save(COPULA_OUTPUT_DIR / "wind_synthetic.npy",  wind_synth)
    np.save(COPULA_OUTPUT_DIR / "solar_test.npy",      solar_test)
    np.save(COPULA_OUTPUT_DIR / "wind_test.npy",       wind_test)
    
    print(f"\n[Copula] Done. Files saved to {COPULA_OUTPUT_DIR}/")
    print(f"  solar_synthetic : {solar_synth.shape}")
    print(f"  wind_synthetic  : {wind_synth.shape}")
    print(f"  solar_test      : {solar_test.shape}")
    print(f"  wind_test       : {wind_test.shape}")

if __name__ == "__main__":
    main()