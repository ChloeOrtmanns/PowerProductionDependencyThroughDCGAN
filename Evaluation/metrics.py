import wandb
import torch
import numpy
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance
from torchmetrics.functional import structural_similarity_index_measure as ssim

import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_metrics(g_running, real_imgs, stats, layer_num, phase, device):
    """
    Generate fake images and compute domain-specific metrics.
    real_imgs: (B, 2, H, W) tensor already on device, in [-1, 1]
    """
    g_running.eval()
    with torch.no_grad():
        z = torch.randn(real_imgs.size(0), 512, device=device)
        fake_imgs = g_running(z, layer_num=layer_num, alpha=1.0).cpu()

    real_imgs = real_imgs.cpu()

    # --- per-channel Wasserstein distance ---
    for ch, name in enumerate(["solar", "wind"]):
        real_vals = real_imgs[:, ch].numpy().flatten()
        fake_vals = fake_imgs[:, ch].numpy().flatten()
        w_dist = wasserstein_distance(real_vals, fake_vals)
        wandb.log({f"{name}_wasserstein": w_dist,
                "layer_num": layer_num, "phase": phase})

    # --- spatial autocorrelation (horizontal shift by 1 pixel) ---
    for ch, name in enumerate(["solar", "wind"]):
        real_ch   = real_imgs[:, ch]
        fake_ch   = fake_imgs[:, ch]
        real_auto = (real_ch * torch.roll(real_ch, 1, dims=-1)).mean().item()
        fake_auto = (fake_ch * torch.roll(fake_ch, 1, dims=-1)).mean().item()
        wandb.log({f"{name}_autocorr_real": real_auto,
                f"{name}_autocorr_fake": fake_auto,
                "layer_num": layer_num, "phase": phase})

    # --- solar-wind channel correlation ---
    def channel_corr(imgs):
        solar = imgs[:, 0].flatten()
        wind  = imgs[:, 1].flatten()
        return torch.corrcoef(torch.stack([solar, wind]))[0, 1].item()

    real_corr = channel_corr(real_imgs)
    fake_corr = channel_corr(fake_imgs)
    wandb.log({"solar_wind_corr_real": real_corr,
            "solar_wind_corr_fake": fake_corr,
            "layer_num": layer_num, "phase": phase})

    # --- mean and std per channel ---
    for ch, name in enumerate(["solar", "wind"]):
        wandb.log({
            f"{name}_mean_real": real_imgs[:, ch].mean().item(),
            f"{name}_mean_fake": fake_imgs[:, ch].mean().item(),
            f"{name}_std_real":  real_imgs[:, ch].std().item(),
            f"{name}_std_fake":  fake_imgs[:, ch].std().item(),
            "layer_num": layer_num, "phase": phase,
        })

    # --- SSIM per channel ---
    # SSIM expects values in [0, 1], so shift from [-1, 1]
    real_01 = (real_imgs + 1) / 2
    fake_01 = (fake_imgs + 1) / 2

    ssim_solar = ssim(fake_01[:, 0:1], real_01[:, 0:1], data_range=1.0).item()
    ssim_wind  = ssim(fake_01[:, 1:2], real_01[:, 1:2], data_range=1.0).item()

    wandb.log({
        "ssim_solar": ssim_solar,
        "ssim_wind":  ssim_wind,
        "layer_num":  layer_num,
        "phase":      phase,
    })

    g_running.train(False)
    return fake_imgs

def plot_metrics_dashboard(real_imgs, fake_imgs, stats, layer_num):
    real_imgs = real_imgs.cpu()
    fake_imgs = fake_imgs.cpu()

    metrics = {}

    # --- Wasserstein ---
    for ch, name in enumerate(["solar", "wind"]):
        r = real_imgs[:, ch].numpy().flatten()
        f = fake_imgs[:, ch].numpy().flatten()
        metrics[f"{name}_wasserstein"] = wasserstein_distance(r, f)

    # --- autocorr ---
    for ch, name in enumerate(["solar", "wind"]):
        r = real_imgs[:, ch]
        f = fake_imgs[:, ch]

        metrics[f"{name}_autocorr_real"] = (r * torch.roll(r, 1, -1)).mean().item()
        metrics[f"{name}_autocorr_fake"] = (f * torch.roll(f, 1, -1)).mean().item()

    # --- correlation ---
    sr = real_imgs[:, 0].flatten()
    wr = real_imgs[:, 1].flatten()
    sf = fake_imgs[:, 0].flatten()
    wf = fake_imgs[:, 1].flatten()

    metrics["corr_real"] = torch.corrcoef(torch.stack([sr, wr]))[0,1].item()
    metrics["corr_fake"] = torch.corrcoef(torch.stack([sf, wf]))[0,1].item()

    # --- add mean and std ---
    for ch, name in enumerate(["solar", "wind"]):
        metrics[f"{name}_mean_real"] = real_imgs[:, ch].mean().item()
        metrics[f"{name}_mean_fake"] = fake_imgs[:, ch].mean().item()
        metrics[f"{name}_std_real"]  = real_imgs[:, ch].std().item()
        metrics[f"{name}_std_fake"]  = fake_imgs[:, ch].std().item()

    # ---------------- PLOTS ----------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1. Wasserstein
    axes[0, 0].bar(["solar", "wind"],
                   [metrics["solar_wasserstein"], metrics["wind_wasserstein"]])
    axes[0, 0].set_title("Wasserstein Distance")

    # 2. Autocorr
    axes[0, 1].bar(
        ["solar_real", "solar_fake", "wind_real", "wind_fake"],
        [metrics["solar_autocorr_real"], metrics["solar_autocorr_fake"],
         metrics["wind_autocorr_real"],  metrics["wind_autocorr_fake"]]
    )
    axes[0, 1].set_title("Spatial Autocorrelation")

    # 3. Correlation
    axes[0, 2].bar(["real", "fake"], [metrics["corr_real"], metrics["corr_fake"]])
    axes[0, 2].set_title("Solar-Wind Correlation")

    # row 1: mean and std
    axes[1, 0].bar(
        ["solar_real", "solar_fake", "wind_real", "wind_fake"],
        [metrics["solar_mean_real"], metrics["solar_mean_fake"],
         metrics["wind_mean_real"],  metrics["wind_mean_fake"]]
    )
    axes[1, 0].set_title("Mean per Channel")

    axes[1, 1].bar(
        ["solar_real", "solar_fake", "wind_real", "wind_fake"],
        [metrics["solar_std_real"], metrics["solar_std_fake"],
         metrics["wind_std_real"],  metrics["wind_std_fake"]]
    )
    axes[1, 1].set_title("Std per Channel")

    axes[1, 2].axis("off")  # empty for now, could add SSIM here later

    plt.tight_layout()

    path = Path(f"Resources/samples/metrics_stage{layer_num}.png")
    plt.savefig(path)
    plt.close()

    wandb.log({
        "metrics_table": wandb.Table(
            columns=["metric", "real", "fake"],
            data=[
                ["solar_wasserstein",  metrics["solar_wasserstein"],  None],
                ["wind_wasserstein",   metrics["wind_wasserstein"],   None],
                ["solar_autocorr",     metrics["solar_autocorr_real"], metrics["solar_autocorr_fake"]],
                ["wind_autocorr",      metrics["wind_autocorr_real"],  metrics["wind_autocorr_fake"]],
                ["solar_wind_corr",    metrics["corr_real"],           metrics["corr_fake"]],
                ["solar_mean",         metrics["solar_mean_real"],     metrics["solar_mean_fake"]],
                ["wind_mean",          metrics["wind_mean_real"],      metrics["wind_mean_fake"]],
                ["solar_std",          metrics["solar_std_real"],      metrics["solar_std_fake"]],
                ["wind_std",           metrics["wind_std_real"],       metrics["wind_std_fake"]],
            ]
        )
    })
    return path