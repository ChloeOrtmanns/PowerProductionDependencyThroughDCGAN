"""
Inspects the found PCA components to see what they specifically mean
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

COPULA_OUTPUT_DIR = Path("Resources/FakeData/20260506 - 90")

# 1. Load real test data (already saved by copulaMapGen.py)
solar_test = np.load(COPULA_OUTPUT_DIR / "solar_test.npy")
wind_test  = np.load(COPULA_OUTPUT_DIR / "wind_test.npy")

# 2. Flatten and stack into (N, 48642)
N = solar_test.shape[0]
flat = np.concatenate([solar_test.reshape(N, -1), wind_test.reshape(N, -1)], axis=1)

# 3. Fit PCA on real data
scaler = StandardScaler()
scaled = scaler.fit_transform(flat)
pca    = PCA(n_components=3)
pca.fit(scaled)

# 4. Plot loadings for PC1, PC2, PC3
H, W      = 121, 201
n_pixels  = H * W

for pc_idx in range(3):
    loading       = pca.components_[pc_idx]         # 48642 numbers
    solar_loading = loading[:n_pixels].reshape(H, W)
    wind_loading  = loading[n_pixels:].reshape(H, W)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"PC{pc_idx+1} loadings  ({pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance)")

    for ax, data, title in zip(axes, [solar_loading, wind_loading], ["Solar", "Wind"]):
        im = ax.imshow(data, origin="lower", cmap="RdBu_r")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(COPULA_OUTPUT_DIR / f"pc{pc_idx+1}_loadings.png", dpi=150)
    plt.close()
    print(f"Saved PC{pc_idx+1} loadings")