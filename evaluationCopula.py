"""
evaluateCopula.py
-----------------
Standalone evaluation script for the Gaussian Copula baseline.
Loads the synthetic grids produced by copulaMapGen.py and runs the exact
same metrics as StyleGAN, so results are directly comparable.

Run after copulaMapGen.py has finished:
    python evaluateCopula.py

Output:
    - Resources/CopulaOutput/metrics_copula.png   (same dashboard as StyleGAN)
    - Logged to wandb project "StyleGAN-full" under run name "copula_baseline"
"""

from pathlib import Path

import numpy as np
import torch
import wandb
from scipy.stats import wasserstein_distance

from config import STATS_PATH
from Evaluation.metrics import (
    kolmogorov_smirnov,
    compute_tail_metrics,
    plot_metrics_dashboard,
)

COPULA_OUTPUT_DIR = Path("Resources/FakeData/20260506 - 90")
STATS_PATH = Path("Resources/FakeData/20260506 - 90/stats_90.pt")
SPLITS_PATH = Path("Resources/FakeData/20260507 - 50/splits_90.pt")


# ---------------------------------------------------------------------------
# Load copula outputs
# ---------------------------------------------------------------------------

def load_copula_outputs():
    """
    Load the .npy files saved by copulaMapGen.py and stack into
    (N, 2, H, W) tensors that match what metrics.py expects.
    """
    solar_synth = np.load(COPULA_OUTPUT_DIR / "solar_synthetic.npy")  # (N, 121, 201)
    wind_synth  = np.load(COPULA_OUTPUT_DIR / "wind_synthetic.npy")   # (N, 121, 201)
    solar_test  = np.load(COPULA_OUTPUT_DIR / "solar_test.npy")       # (M, 121, 201)
    wind_test   = np.load(COPULA_OUTPUT_DIR / "wind_test.npy")        # (M, 121, 201)

    # Stack channels: (N, 2, H, W)
    fake_np = np.stack([solar_synth, wind_synth], axis=1)
    real_np = np.stack([solar_test,  wind_test],  axis=1)

    # metrics.py works with tensors in [-1, 1], so we need to normalise.
    # We use the same stats StyleGAN uses (computed on train split).
    stats = torch.load(STATS_PATH, weights_only=False)

    def normalise(arr, ch):
        key = "solar" if ch == 0 else "wind"
        mn  = stats[f"{key}Min"]
        mx  = stats[f"{key}Max"]
        return 2.0 * (arr - mn) / (mx - mn) - 1.0

    fake_01 = np.stack([normalise(fake_np[:, 0], 0),
                        normalise(fake_np[:, 1], 1)], axis=1)
    real_01 = np.stack([normalise(real_np[:, 0], 0),
                        normalise(real_np[:, 1], 1)], axis=1)

    fake_t = torch.tensor(fake_01, dtype=torch.float32)
    real_t = torch.tensor(real_01, dtype=torch.float32)

    print(f"[Eval] Real  (test) : {real_t.shape}")
    print(f"[Eval] Fake (copula): {fake_t.shape}")

    return real_t, fake_t, real_np, fake_np, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wandb.init(
        project="StyleGAN-full",
        name="copula_baseline_90",
        config={
            "model":        "GaussianCopula",
            "pca_components": 50,
            "marginals":    "GaussianUnivariate",
        }
    )

    real_t, fake_t, real_np, fake_np, stats = load_copula_outputs()

    # --- Wasserstein per channel ---
    for ch, name in enumerate(["solar", "wind"]):
        w = wasserstein_distance(
            real_t[:, ch].numpy().flatten(),
            fake_t[:, ch].numpy().flatten(),
        )
        wandb.log({f"{name}_wasserstein": w})
        print(f"[Eval] {name} Wasserstein: {w:.4f}")

    # --- Spatial autocorrelation ---
    for ch, name in enumerate(["solar", "wind"]):
        r = real_t[:, ch]
        f = fake_t[:, ch]
        wandb.log({
            f"{name}_autocorr_real": (r * torch.roll(r, 1, -1)).mean().item(),
            f"{name}_autocorr_fake": (f * torch.roll(f, 1, -1)).mean().item(),
        })

    # --- Solar-wind channel correlation ---
    def channel_corr(t):
        solar = t[:, 0].flatten()
        wind  = t[:, 1].flatten()
        return torch.corrcoef(torch.stack([solar, wind]))[0, 1].item()

    wandb.log({
        "solar_wind_corr_real": channel_corr(real_t),
        "solar_wind_corr_fake": channel_corr(fake_t),
    })

    # --- Mean and std per channel ---
    for ch, name in enumerate(["solar", "wind"]):
        wandb.log({
            f"{name}_mean_real": real_t[:, ch].mean().item(),
            f"{name}_mean_fake": fake_t[:, ch].mean().item(),
            f"{name}_std_real":  real_t[:, ch].std().item(),
            f"{name}_std_fake":  fake_t[:, ch].std().item(),
        })

    # --- KS test + tail metrics (reuse metrics.py directly) ---
    # metrics.py expects raw (unnormalised) numpy arrays for tail metrics
    ks_results   = kolmogorov_smirnov(real_t.numpy(), fake_t.numpy())
    tail_results = compute_tail_metrics(real_t.numpy(), fake_t.numpy(), q=0.95)
    wandb.log(ks_results)
    wandb.log(tail_results)

    # --- Full dashboard plot (same function as StyleGAN) ---
    # plot_metrics_dashboard saves the png and logs a wandb table internally.
    # We pass layer_num=0 as a neutral placeholder since this is not a GAN stage.
    path = plot_metrics_dashboard(real_t, fake_t, stats, layer_num=0, log_to_wandb=False)

    # Move the saved plot to CopulaOutput so it doesn't overwrite StyleGAN plots
    import shutil
    dest = COPULA_OUTPUT_DIR / "metrics_copula.png"
    shutil.copy(path, dest)
    wandb.log({"copula_metrics_dashboard": wandb.Image(str(dest))})
    print(f"[Eval] Dashboard saved to {dest}")

    wandb.finish()


if __name__ == "__main__":
    main()