"""
Bunch of mathematical metrics, plotted on a dashboard compatible with wandb
"""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance, ks_2samp
from torchmetrics.functional import structural_similarity_index_measure as ssim

def kolmogorov_smirnov(real_np, fake_np):
    """
    KS test per channel (pixel-wise) and spatially (per-location mean across samples).
    Returns dict of statistic/pvalue pairs.
    """
    results = {}
    for ch, name in enumerate(["solar", "wind"]):
        r = real_np[:, ch].flatten()
        f = fake_np[:, ch].flatten()
        stat, pval = ks_2samp(r, f)
        results[f"{name}_ks_stat"] = stat
        results[f"{name}_ks_pval"] = pval

        # spatial KS: per grid-cell distribution across samples
        # real/fake shape: (B, H, W) → flatten H*W locations
        r_spatial = real_np[:, ch]   # (B, H, W)
        f_spatial = fake_np[:, ch]
        H, W = r_spatial.shape[1], r_spatial.shape[2]
        spatial_stats = []
        # subsample locations to keep it fast (every 4th pixel)
        for hi in range(0, H, 4):
            for wi in range(0, W, 4):
                s, _ = ks_2samp(r_spatial[:, hi, wi], f_spatial[:, hi, wi])
                spatial_stats.append(s)
        results[f"{name}_ks_spatial_mean"] = float(np.mean(spatial_stats))

    return results

def tail_dependence(x, y, q=0.95):
    """
    Upper and lower tail dependence coefficients between two 1D arrays.
    Upper TDC: P(Y > q_y | X > q_x)  — co-occurrence of high extremes
    Lower TDC: P(Y < q_y | X < q_x)  — co-occurrence of low extremes
    q: quantile threshold (0.95 = top 5%)
    """
    upper_x = np.quantile(x, q)
    upper_y = np.quantile(y, q)
    lower_x = np.quantile(x, 1 - q)
    lower_y = np.quantile(y, 1 - q)

    mask_upper_x = x > upper_x
    mask_lower_x = x < lower_x

    upper_tdc = np.mean(y[mask_upper_x] > upper_y) if mask_upper_x.sum() > 0 else float("nan")
    lower_tdc = np.mean(y[mask_lower_x] < lower_y) if mask_lower_x.sum() > 0 else float("nan")

    return upper_tdc, lower_tdc

def causal_tail_coeff(x, y, q=0.95):
    """
    Causal Tail Coefficient:
      CTC(X→Y): given X is extreme, how often is Y also extreme?
      CTC(Y→X): given Y is extreme, how often is X also extreme?
    Asymmetry = CTC(X→Y) - CTC(Y→X): positive means X drives Y's tail more than reverse.
    Uses both upper and lower tails, averaged.
    """
    def _ctc(driver, response, quantile):
        upper_d = np.quantile(driver, quantile)
        upper_r = np.quantile(response, quantile)
        lower_d = np.quantile(driver, 1 - quantile)
        lower_r = np.quantile(response, 1 - quantile)

        mask_up = driver > upper_d
        mask_lo = driver < lower_d

        ctc_up = np.mean(response[mask_up] > upper_r) if mask_up.sum() > 0 else float("nan")
        ctc_lo = np.mean(response[mask_lo] < lower_r) if mask_lo.sum() > 0 else float("nan")

        return float(np.nanmean([ctc_up, ctc_lo]))

    ctc_xy = _ctc(x, y, q)   # solar drives wind
    ctc_yx = _ctc(y, x, q)   # wind drives solar
    return ctc_xy, ctc_yx, ctc_xy - ctc_yx

def compute_tail_metrics(real_np, fake_np, q=0.95):
    """
    Compute TDC and CTC for:
      - solar vs wind cross-channel  (per pixel, flattened)
      - spatial neighbors (each pixel vs its right neighbor)
    """
    results = {}

    for ch, name in enumerate(["solar", "wind"]):
        r = real_np[:, ch]   # (B, H, W)
        f = fake_np[:, ch]

        # ── spatial neighbor TDC/CTC (pixel vs right neighbor) ──
        r_flat  = r[:, :, :-1].flatten()   # all pixels except last column
        r_right = r[:, :, 1:].flatten()    # their right neighbors
        f_flat  = f[:, :, :-1].flatten()
        f_right = f[:, :, 1:].flatten()

        ru, rl = tail_dependence(r_flat, r_right, q)
        fu, fl = tail_dependence(f_flat, f_right, q)
        results[f"{name}_spatial_upper_tdc_real"] = ru
        results[f"{name}_spatial_upper_tdc_fake"] = fu
        results[f"{name}_spatial_lower_tdc_real"] = rl
        results[f"{name}_spatial_lower_tdc_fake"] = fl

        rxy, ryx, rasym = causal_tail_coeff(r_flat, r_right, q)
        fxy, fyx, fasym = causal_tail_coeff(f_flat, f_right, q)
        results[f"{name}_spatial_ctc_real"] = rasym   # asymmetry: pixel→neighbor
        results[f"{name}_spatial_ctc_fake"] = fasym

    # ── cross-channel TDC/CTC (solar vs wind, pixel-wise) ──
    r_solar = real_np[:, 0].flatten()
    r_wind  = real_np[:, 1].flatten()
    f_solar = fake_np[:, 0].flatten()
    f_wind  = fake_np[:, 1].flatten()

    ru, rl = tail_dependence(r_solar, r_wind, q)
    fu, fl = tail_dependence(f_solar, f_wind, q)
    results["cross_upper_tdc_real"] = ru
    results["cross_upper_tdc_fake"] = fu
    results["cross_lower_tdc_real"] = rl
    results["cross_lower_tdc_fake"] = fl

    rxy, ryx, rasym = causal_tail_coeff(r_solar, r_wind, q)
    fxy, fyx, fasym = causal_tail_coeff(f_solar, f_wind, q)
    results["cross_ctc_solar_drives_wind_real"] = rxy
    results["cross_ctc_solar_drives_wind_fake"] = fxy
    results["cross_ctc_wind_drives_solar_real"] = ryx
    results["cross_ctc_wind_drives_solar_fake"] = fyx
    results["cross_ctc_asymmetry_real"] = rasym
    results["cross_ctc_asymmetry_fake"] = fasym

    return results

########################################################################################

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

    H = fake_01.shape[2]
    W = fake_01.shape[3]
    if min(H, W) >= 11:
        ssim_solar = ssim(fake_01[:, 0:1], real_01[:, 0:1], data_range=1.0).item()
        ssim_wind  = ssim(fake_01[:, 1:2], real_01[:, 1:2], data_range=1.0).item()
    else:
        ssim_solar = float("nan")
        ssim_wind  = float("nan")

    wandb.log({
        "ssim_solar": ssim_solar,
        "ssim_wind":  ssim_wind,
        "layer_num":  layer_num,
        "phase":      phase,
    })

    g_running.train(False)
    return fake_imgs

def plot_metrics_dashboard(real_imgs, fake_imgs, stats, layer_num, log_to_wandb=True):
    real_imgs = real_imgs.cpu()
    fake_imgs = fake_imgs.cpu()
    real_np   = real_imgs.numpy()  # ← add this
    fake_np   = fake_imgs.numpy()  # ← add this

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

    # ── new metrics ───────────────────────────────────────────────
    ks      = kolmogorov_smirnov(real_np, fake_np)
    tail    = compute_tail_metrics(real_np, fake_np, q=0.95)
    metrics.update(ks)
    metrics.update(tail)

    # ---------------- PLOTS ----------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

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

    # [1,2] KS statistic (pixel-wise) — NEW
    axes[1, 2].bar(
        ["solar", "wind"],
        [metrics["solar_ks_stat"], metrics["wind_ks_stat"]],
        color=["#f4a261", "#457b9d"]
    )
    # add p-value as text annotation
    for j, name in enumerate(["solar", "wind"]):
        pval = metrics[f"{name}_ks_pval"]
        label = f"p={pval:.2e}" if pval > 0 else "p≈0"
        axes[1, 2].text(j, metrics[f"{name}_ks_stat"] + 0.002, label,
                        ha="center", fontsize=8)
    axes[1, 2].set_title("KS Statistic (pixel-wise)\n(lower = better, p≈0 means distributions differ)")

    # [2,0] Tail Dependence Coefficient — cross-channel — NEW
    labels = ["upper\nreal", "upper\nfake", "lower\nreal", "lower\nfake"]
    vals   = [metrics["cross_upper_tdc_real"], metrics["cross_upper_tdc_fake"],
              metrics["cross_lower_tdc_real"], metrics["cross_lower_tdc_fake"]]
    bars = axes[2, 0].bar(labels, vals, color=["#2a9d8f", "#e76f51", "#2a9d8f", "#e76f51"])
    for bar, a in zip(bars, [1, 1, 0.5, 0.5]):
        bar.set_alpha(a)
    axes[2, 0].set_title("Cross-channel TDC (solar↔wind)\n(upper=high extremes, lower=low extremes)")
    axes[2, 0].set_ylim(0, 1)

    # [2,1] Spatial TDC (pixel vs right neighbor) — NEW
    x      = np.arange(2)
    width  = 0.2
    for ci, (ch_name, color_r, color_f) in enumerate([
        ("solar", "#f4a261", "#e9c46a"),
        ("wind",  "#457b9d", "#90e0ef"),
    ]):
        axes[2, 1].bar(x + ci * width * 2,
                       [metrics[f"{ch_name}_spatial_upper_tdc_real"],
                        metrics[f"{ch_name}_spatial_lower_tdc_real"]],
                       width=width, label=f"{ch_name} real", color=color_r)
        axes[2, 1].bar(x + ci * width * 2 + width,
                       [metrics[f"{ch_name}_spatial_upper_tdc_fake"],
                        metrics[f"{ch_name}_spatial_lower_tdc_fake"]],
                       width=width, label=f"{ch_name} fake", color=color_f)
    axes[2, 1].set_xticks(x + width * 1.5)
    axes[2, 1].set_xticklabels(["upper tail", "lower tail"])
    axes[2, 1].set_title("Spatial TDC (pixel→neighbor)\n(fake should match real)")
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].legend(fontsize=7)

    # [2,2] Causal Tail Coefficient asymmetry — NEW
    # positive = solar drives wind's tail, negative = wind drives solar's tail
    ctc_labels = ["solar→wind\nreal", "solar→wind\nfake",
                  "wind→solar\nreal", "wind→solar\nfake",
                  "asymmetry\nreal",  "asymmetry\nfake"]
    ctc_vals   = [metrics["cross_ctc_solar_drives_wind_real"],
                  metrics["cross_ctc_solar_drives_wind_fake"],
                  metrics["cross_ctc_wind_drives_solar_real"],
                  metrics["cross_ctc_wind_drives_solar_fake"],
                  metrics["cross_ctc_asymmetry_real"],
                  metrics["cross_ctc_asymmetry_fake"]]
    colors = ["#2a9d8f", "#e76f51", "#2a9d8f", "#e76f51", "#264653", "#e9c46a"]
    axes[2, 2].bar(ctc_labels, ctc_vals, color=colors)
    axes[2, 2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2, 2].set_title("Causal Tail Coefficient\n(asymmetry: + means solar drives wind)")
    axes[2, 2].tick_params(axis="x", labelsize=7)

    plt.tight_layout()

    path = Path(f"Resources/samples/metrics_stage{layer_num}.png")
    plt.savefig(path)
    plt.close()

    if log_to_wandb:
        wandb.log({
            "metrics_table": wandb.Table(
                columns=["metric", "real", "fake"],
                data=[
                    ["solar_wasserstein",           metrics["solar_wasserstein"],                  None],
                    ["wind_wasserstein",            metrics["wind_wasserstein"],                   None],
                    ["solar_autocorr",              metrics["solar_autocorr_real"],                metrics["solar_autocorr_fake"]],
                    ["wind_autocorr",               metrics["wind_autocorr_real"],                 metrics["wind_autocorr_fake"]],
                    ["solar_wind_corr",             metrics["corr_real"],                          metrics["corr_fake"]],
                    ["solar_mean",                  metrics["solar_mean_real"],                    metrics["solar_mean_fake"]],
                    ["wind_mean",                   metrics["wind_mean_real"],                     metrics["wind_mean_fake"]],
                    ["solar_std",                   metrics["solar_std_real"],                     metrics["solar_std_fake"]],
                    ["wind_std",                    metrics["wind_std_real"],                      metrics["wind_std_fake"]],
                    ["solar_ks_stat",               metrics["solar_ks_stat"],                      None],
                    ["wind_ks_stat",                metrics["wind_ks_stat"],                       None],
                    ["solar_ks_spatial_mean",       metrics["solar_ks_spatial_mean"],              None],
                    ["wind_ks_spatial_mean",        metrics["wind_ks_spatial_mean"],               None],
                    ["cross_upper_tdc",             metrics["cross_upper_tdc_real"],               metrics["cross_upper_tdc_fake"]],
                    ["cross_lower_tdc",             metrics["cross_lower_tdc_real"],               metrics["cross_lower_tdc_fake"]],
                    ["solar_spatial_upper_tdc",     metrics["solar_spatial_upper_tdc_real"],       metrics["solar_spatial_upper_tdc_fake"]],
                    ["solar_spatial_lower_tdc",     metrics["solar_spatial_lower_tdc_real"],       metrics["solar_spatial_lower_tdc_fake"]],
                    ["wind_spatial_upper_tdc",      metrics["wind_spatial_upper_tdc_real"],        metrics["wind_spatial_upper_tdc_fake"]],
                    ["wind_spatial_lower_tdc",      metrics["wind_spatial_lower_tdc_real"],        metrics["wind_spatial_lower_tdc_fake"]],
                    ["ctc_solar_drives_wind",       metrics["cross_ctc_solar_drives_wind_real"],   metrics["cross_ctc_solar_drives_wind_fake"]],
                    ["ctc_wind_drives_solar",       metrics["cross_ctc_wind_drives_solar_real"],   metrics["cross_ctc_wind_drives_solar_fake"]],
                    ["ctc_asymmetry",               metrics["cross_ctc_asymmetry_real"],           metrics["cross_ctc_asymmetry_fake"]],
                ]
            )
        })
    return path