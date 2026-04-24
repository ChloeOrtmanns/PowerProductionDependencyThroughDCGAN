from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
from scipy.stats import norm, rankdata
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_copula_map(real_imgs, stats, n_samples=4):
    import numpy as np
    from scipy.stats import norm, rankdata
    from scipy.stats import multivariate_normal

    def denorm(x, vmin, vmax):
        return (x + 1) / 2 * (vmax - vmin) + vmin

    H, W = 121, 201
    # --- work on coarse grid to keep covariance matrix manageable ---
    H_c, W_c = 20, 30  # coarse grid — matrix is only 600x600
    
    real_solar = denorm(real_imgs[:, 0], stats["solarMin"], stats["solarMax"])
    real_wind  = denorm(real_imgs[:, 1], stats["windMin"],  stats["windMax"])

    # --- Step 1: fit length scale from real data ---
    # estimate correlation length in coarse grid units
    # real grid: 121x201, coarse: 20x30 → scale factor ~6
    length_scale_coarse = 5.0  # ~30 pixels on full grid, ~150km

    # --- Step 2: build coarse coordinate grid ---
    yy, xx = np.meshgrid(np.arange(H_c), np.arange(W_c), indexing='ij')
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)  # (600, 2)
    N = len(coords)  # 600 — manageable

    # --- Step 3: build RBF covariance matrix on coarse grid ---
    def rbf_cov(coords, length_scale):
        diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
        sq_dist = (diff**2).sum(axis=-1)                # (N, N)
        return np.exp(-sq_dist / (2 * length_scale**2))

    K = rbf_cov(coords, length_scale_coarse)
    K += 1e-6 * np.eye(N)  # numerical stability
    L = np.linalg.cholesky(K)  # (600, 600) — fast

    # --- Step 4: fit copula rho from real data ---
    real_flat = real_imgs.permute(0,2,3,1).reshape(-1, 2).numpy()
    eps = 1e-6
    uA = rankdata(real_flat[:, 0]) / (len(real_flat) + 1)
    uB = rankdata(real_flat[:, 1]) / (len(real_flat) + 1)
    zA = norm.ppf(np.clip(uA, eps, 1-eps))
    zB = norm.ppf(np.clip(uB, eps, 1-eps))
    rho = np.corrcoef(zA, zB)[0, 1]
    print(f"Fitted copula rho: {rho:.3f}")

    solar_out = np.zeros((n_samples, H, W))
    wind_out  = np.zeros((n_samples, H, W))

    for i in range(n_samples):
        # --- Step 5: sample two independent GRFs on coarse grid ---
        z1 = L @ np.random.randn(N)  # solar field
        z2 = L @ np.random.randn(N)  # wind field (independent)

        # --- Step 6: impose copula dependency ---
        z2_coupled = rho * z1 + np.sqrt(1 - rho**2) * z2

        # reshape to coarse grid
        solar_coarse = z1.reshape(H_c, W_c)
        wind_coarse  = z2_coupled.reshape(H_c, W_c)

        # --- Step 7: upsample to full grid ---
        solar_fine = torch.nn.functional.interpolate(
            torch.tensor(solar_coarse).float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        wind_fine = torch.nn.functional.interpolate(
            torch.tensor(wind_coarse).float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        # --- Step 8: invert marginals to real scale ---
        def invert_ecdf(z_field, x_ref):
            u = norm.cdf(z_field)  # GRF values → uniform
            quantiles = np.sort(x_ref.numpy().ravel())
            indices = (u * len(quantiles)).astype(int).clip(0, len(quantiles)-1)
            return quantiles[indices]

        solar_out[i] = invert_ecdf(solar_fine, real_solar)
        wind_out[i]  = invert_ecdf(wind_fine,  real_wind)

    # --- Step 9: plot ---
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    fig.suptitle("Gaussian Copula + GRF Generated Maps", fontsize=13)

    for i in range(n_samples):
        axes[0, i].imshow(solar_out[i], cmap="viridis", origin="lower")
        axes[0, i].set_title(f"Sample {i+1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(wind_out[i],  cmap="viridis", origin="lower")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Solar")
    axes[1, 0].set_ylabel("Wind")

    plt.tight_layout()
    path = "Resources/samples/copula_maps.png"
    plt.savefig(path, dpi=150)
    plt.close()
    wandb.log({"copula_maps": wandb.Image(path, caption=
        f"GRF+Copula generated maps | coarse {H_c}x{W_c} → upsampled to {H}x{W} | "
        f"copula rho={rho:.3f}")})

    return solar_out, wind_out