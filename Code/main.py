# ---RANDOM-NOTES--------------------------------------------------------------------------------------------------------------
# Most of the code is based off https://ym2132.github.io/Progressive_GAN#the-gulrajani-generator-g-network

# ---IMPORTS-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from scipy.stats import wasserstein_distance
import wandb
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm
import numpy as np

from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from Models.generator import Generator
from Models.discriminator import Discriminator
from Models.energyDataset import EnergyDataset, get_dataloader

# ---VARIABLES-----------------------------------------------------------------------------------------------------------------
statsPath      = Path("Resources/stats.pt")
splitsPath     = Path("Resources/splits.pt")
pathsPath      = Path("Resources/paths.pt")

STAGE_SIZES = {
    1: (4,   7),
    2: (8,   13),
    3: (16,  26),
    4: (32,  52),
    5: (64,  104),
    6: (121, 201),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using PyTorch {torch.__version__}.")
print(f"Using device {device}.")

if device.type == "cuda":
    _ = torch.empty(1, device=device).normal_()

# ---FUNCTIONS-----------------------------------------------------------------------------------------------------------------    

def savePathsLists():
    solarPaths = list(Path("../../../mnt/mvbc-gan/Data/Power/Solar").glob("*.nc"))
    windPaths  = list(Path("../../../mnt/mvbc-gan/Data/Power/Wind").glob("*.nc"))

    torch.save({
        "solarPaths": solarPaths,
        "windPaths": windPaths
    }, pathsPath)

def calculateSaveStatsSplits():
    solarPaths, windPaths = pathsLists()

    # load arrays first — dataset now expects numpy, not paths
    dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
    dsWind  = xr.open_mfdataset(windPaths,  combine="by_coords")
    dsSolar, dsWind = xr.align(dsSolar, dsWind)
    solar_np = dsSolar["Solar Energy Potential"].values
    wind_np  = dsWind["Wind Energy Potential"].values

    times = dsSolar["Solar Energy Potential"].time.values

    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "idx":  range(len(times)),
        "time": pd.to_datetime(times),
    })
    df["year"]   = df["time"].dt.year
    df["month"]  = df["time"].dt.month
    df["season"] = df["month"].apply(lambda m:
        "DJF" if m in [12,1,2] else
        "MAM" if m in [3,4,5]  else
        "JJA" if m in [6,7,8]  else "SON")

    test_indices, train_indices = [], []
    for year in df["year"].unique():
        for season in ["DJF", "MAM", "JJA", "SON"]:
            mask    = (df["year"] == year) & (df["season"] == season)
            indices = df[mask]["idx"].values
            if len(indices) == 0:
                continue
            n_test = min(9, len(indices))
            chosen = np.random.choice(indices, size=n_test, replace=False)
            test_indices.extend(chosen.tolist())
            train_indices.extend([i for i in indices if i not in chosen])

    torch.save({"train": train_indices, "test": test_indices}, splitsPath)
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

    # create dataset with numpy arrays
    fullDataset = EnergyDataset(solar_np, wind_np, stats=None)
    print(len(fullDataset))

    solarMin = fullDataset.solar[train_indices].min()
    solarMax = fullDataset.solar[train_indices].max()
    windMin  = fullDataset.wind[train_indices].min()
    windMax  = fullDataset.wind[train_indices].max()

    torch.save({
        "solarMin": float(solarMin), "solarMax": float(solarMax),
        "windMin":  float(windMin),  "windMax":  float(windMax),
    }, statsPath)
    print(test_indices)
    print(train_indices)

def pathsLists():
    paths = torch.load(pathsPath, weights_only=False)
    return paths["solarPaths"], paths["windPaths"]

def save_checkpoint(g, d, g_opt, d_opt, layer_num, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "layer_num": layer_num,
        "g_state":   g.state_dict(),
        "d_state":   d.state_dict(),
        "g_opt":     g_opt.state_dict(),
        "d_opt":     d_opt.state_dict(),
    }, filepath)
    print(f"Saved checkpoint → {filepath}")

def load_checkpoint(filepath, g, d, g_opt, d_opt):
    ckpt = torch.load(filepath, map_location=device)
    g.load_state_dict(ckpt["g_state"])
    d.load_state_dict(ckpt["d_state"])
    g_opt.load_state_dict(ckpt["g_opt"])
    d_opt.load_state_dict(ckpt["d_opt"])
    print(f"Loaded checkpoint from {filepath} (layer {ckpt['layer_num']})")
    return ckpt["layer_num"]

# We init a function to calculate the EMA for our parameters.
# This is implemented using two model, one the training G and the other
# a G we do not train but only use to hold the weights after perfoming the EMA
def EMA(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def show_images(imgs, stats, tag, layer_num):
    """
    imgs: tensor of shape (B, 2, H, W), values in [-1, 1]
    Saves a matplotlib figure and logs it to wandb.
    """
    # Take the first sample from the batch
    sample = imgs[0].cpu()
    solar = (sample[0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]
    wind  = (sample[1] + 1) / 2 * (stats["windMax"]  - stats["windMin"])  + stats["windMin"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title("Fake Solar")
    im0 = axes[0].imshow(solar, cmap='viridis', origin='lower')
    plt.colorbar(im0, ax=axes[0])

    axes[1].set_title("Fake Wind")
    im1 = axes[1].imshow(wind, cmap='viridis', origin='lower')
    plt.colorbar(im1, ax=axes[1])

    path = f"Resources/samples/{tag}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

    wandb.log({tag: wandb.Image(path), "layer_num": layer_num})

def get_batch_size(layer_num):
    return {1: 64, 2: 64, 3: 32, 4: 16, 5: 8, 6: 4}[layer_num]

def get_total_iters(layer_num):
    return {
        1: 10000,
        2: 15000,
        3: 25000,
        4: 40000,
        5: 50000,
        6: 75000,
    }[layer_num]


def plot_real_vs_fake(real_imgs, fake_imgs, stats, layer_num):
    real_imgs = real_imgs.cpu()
    fake_imgs = fake_imgs.cpu()

    n = min(4, real_imgs.size(0))

    fig, axes = plt.subplots(4, n, figsize=(4*n, 10))

    for i in range(n):

        # --- denormalize ---
        real_solar = (real_imgs[i, 0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]
        fake_solar = (fake_imgs[i, 0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]

        real_wind = (real_imgs[i, 1] + 1) / 2 * (stats["windMax"] - stats["windMin"]) + stats["windMin"]
        fake_wind = (fake_imgs[i, 1] + 1) / 2 * (stats["windMax"] - stats["windMin"]) + stats["windMin"]

        # compute shared scale per sample per channel
        vmin_s = min(real_solar.min(), fake_solar.min())
        vmax_s = max(real_solar.max(), fake_solar.max())
        vmin_w = min(real_wind.min(), fake_wind.min())
        vmax_w = max(real_wind.max(), fake_wind.max())

        axes[0, i].imshow(real_solar, cmap="viridis", origin="lower", vmin=vmin_s, vmax=vmax_s)
        axes[1, i].imshow(fake_solar, cmap="viridis", origin="lower", vmin=vmin_s, vmax=vmax_s)
        axes[2, i].imshow(real_wind,  cmap="viridis", origin="lower", vmin=vmin_w, vmax=vmax_w)
        axes[3, i].imshow(fake_wind,  cmap="viridis", origin="lower", vmin=vmin_w, vmax=vmax_w)

        # --- solar row ---
        axes[0, i].imshow(real_solar, cmap="viridis", origin="lower")
        axes[0, i].set_title("Real" if i == 0 else "")
        axes[0, i].axis("off")

        axes[1, i].imshow(fake_solar, cmap="viridis", origin="lower")
        axes[1, i].set_title("Fake" if i == 0 else "")
        axes[1, i].axis("off")

        # --- wind row ---
        axes[2, i].imshow(real_wind, cmap="viridis", origin="lower")
        axes[2, i].axis("off")

        axes[3, i].imshow(fake_wind, cmap="viridis", origin="lower")
        axes[3, i].axis("off")

    axes[0, 0].set_ylabel("Solar Real")
    axes[1, 0].set_ylabel("Solar Fake")
    axes[2, 0].set_ylabel("Wind Real")
    axes[3, 0].set_ylabel("Wind Fake")

    plt.tight_layout()

    path = f"Resources/samples/compare_stage{layer_num}.png"
    plt.savefig(path)
    plt.close()

    # in plot_real_vs_fake, replace the wandb.log line at the bottom:
    wandb.log({
        f"compare_stage{layer_num}": wandb.Image(path, caption=
            f"Stage {layer_num} | Rows: Real Solar / Fake Solar / Real Wind / Fake Wind | "
            f"Each column is one test sample | colorscale shared per row per sample")
    })




def plot_histograms(real_imgs, fake_imgs, stats, layer_num):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ch, name in enumerate(["solar", "wind"]):
        real_vals = real_imgs[:, ch].cpu().numpy().flatten()
        fake_vals = fake_imgs[:, ch].cpu().numpy().flatten()

        axes[ch].hist(real_vals, bins=80, alpha=0.5, label="Real", density=True)
        axes[ch].hist(fake_vals, bins=80, alpha=0.5, label="Fake", density=True)
        axes[ch].set_title(name)
        axes[ch].legend()

    path = f"Resources/samples/hist_stage{layer_num}.png"
    plt.savefig(path)
    plt.close()

    # in plot_histograms:
    wandb.log({
        f"hist_stage{layer_num}": wandb.Image(path, caption=
            f"Stage {layer_num} | Pixel value distributions in [-1,1] | "
            f"Fake should match Real shape — gaps indicate the generator misses certain value ranges")
    })



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

    path = f"Resources/samples/metrics_stage{layer_num}.png"
    plt.savefig(path)
    plt.close()

    wandb.log({
    "metrics_dashboard": wandb.Image(path, caption=
        f"Stage {layer_num} | Wasserstein: lower=better | "
        f"Autocorr: fake should match real | Solar-Wind corr: real≈-0.5 physically expected"),
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



# To get the params for the mapping network we look for parameters with "mapping" in the name
def get_params_with_lr(model):
    mapping_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'mapping' in name:  # Adjust this condition based on your actual naming convention
            mapping_params.append(param)
        else:
            other_params.append(param)
    return mapping_params, other_params

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



def main():
    #--only run this function once
    # savePathsLists()
    # calculateSaveStatsSplits()

    # Load once
    solarPaths, windPaths = pathsLists()
    stats  = torch.load(statsPath,  weights_only=False)
    splits = torch.load(splitsPath, weights_only=False)

    dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
    dsWind  = xr.open_mfdataset(windPaths,  combine="by_coords")
    dsSolar, dsWind = xr.align(dsSolar, dsWind)
    solar_np = dsSolar["Solar Energy Potential"].values  # (T, 121, 201)
    wind_np  = dsWind["Wind Energy Potential"].values    # (T, 121, 201)

    wandb.init(
        project="pggan-energy",
        name="202604182300",
        config={
            "batchSize":   "64->4",
            "total_iters": "10k -> 75 k",
            "lr":          0.001,
            "betas":       (0.0, 0.99),
            "gp_lambda":   10,
            "stages":      6,
        }
    )
    

    #---DATASET-SPLITTING---------------------------------------------------------------------------------------------------------
    ##so right now, I split it into 0.8-0.2 and if I'm correct, i just take the first 80% of the years for training,
    ##and use the remaining years for validation. Idealiter, I want a test with equal seasons and also varying through all the years
    ##so rn I have data from 1975-2005 (i can technically add 2006-2025 to it)

    ##if I'd take 10% instead, so 0.9-0.1, that means every year should have +- 36 days to test on, the rest is training.
    ##divide that through the 4 seasons every year, it means I need 9 random days from every season every year to build my test data on

    test_loader = get_dataloader(
        solar_np, wind_np, stats, splits,
        layer_num=6, STAGE_SIZES=STAGE_SIZES,
        batch_size=32, split="test"
    )

    # ---MODELS----------------------------------------------------
    g         = Generator().to(device)
    g_running = Generator().to(device)
    d         = Discriminator().to(device)
    g_running.train(False)

    mapping_params, other_params = get_params_with_lr(g)
    g_optimizer = torch.optim.Adam([
        {'params': other_params,   'lr': 0.001},
        {'params': mapping_params, 'lr': 0.001 * 0.01},  # 100x lower for mapping network
    ], betas=(0.0, 0.99))
    d_optimizer = torch.optim.Adam(d.parameters(), lr=0.001, betas=(0.0, 0.99))

    EMA(g_running, g, 0)

    resume_stage = 1        # set to the last successfully saved stage
    resume = False          # set to False for a fresh run

    if resume:
        start_stage = resume_stage + 1
        load_checkpoint(
            Path(f"Resources/checkpoints/stage{resume_stage}.pt"),
            g, d, g_optimizer, d_optimizer
        )
    else:
        start_stage = 1

    # before the stage loop
    global_step = 0

    # ---PROGRESSIVE LOOP--------------------------------------
    for layer_num in range(start_stage, 7):
        batchSize = get_batch_size(layer_num)
        total_iters = get_total_iters(layer_num)
        alpha = 0.0

        data_loader = get_dataloader(
            solar_np, wind_np, stats, splits,
            layer_num, STAGE_SIZES, batch_size=batchSize, split="train"
        )
        dataset = iter(data_loader)
        print(f"\n--- Stage {layer_num}: {STAGE_SIZES[layer_num]} ---")

        for phase in ["fadein", "stabilise"]:
            pbar = tqdm(range(total_iters), desc=f"stage {layer_num} {phase}")

            for i in pbar:
                # current alpha: ramps 0→1 during fadein, fixed 1.0 during stabilise
                a = alpha if phase == "fadein" else 1.0

                # --- real batch ---
                try:
                    real_imgs = next(dataset)
                except StopIteration:
                    dataset = iter(data_loader)
                    real_imgs = next(dataset)

                real_imgs    = real_imgs.to(device)
                batch_size_i = real_imgs.size(0)

                # --- train D ---
                d.zero_grad()

                real_preds      = d(real_imgs, layer_num=layer_num, alpha=a)
                real_preds_mean = real_preds.mean() - 0.001 * (real_preds ** 2).mean()

                z = torch.randn(batch_size_i, 512, device=device)
                fake_imgs = g(z, layer_num=layer_num, alpha=a)
                
                fake_preds_mean = d(fake_imgs.detach(), layer_num=layer_num, alpha=a).mean()

                eps   = torch.rand(batch_size_i, 1, 1, 1, device=device)
                x_hat = (eps * real_imgs + (1 - eps) * fake_imgs.detach()).requires_grad_(True)
                grads = torch.autograd.grad(
                    outputs=d(x_hat, layer_num=layer_num, alpha=a),
                    inputs=x_hat,
                    grad_outputs=torch.ones(batch_size_i, 1, device=device),
                    create_graph=True, retain_graph=True
                )[0]
                # flatten all non-batch dims before computing norm
                GP = ((grads.view(batch_size_i, -1).norm(2, dim=1) - 1) ** 2).mean()
                d_loss = (fake_preds_mean - real_preds_mean) + 10 * GP
                d_loss.backward()
                d_optimizer.step()

                # --- train G ---
                g.zero_grad()
                z         = torch.randn(batch_size_i, 512, device=device)  # ← was g(batch_size_i, ...)
                fake_imgs = g(z, layer_num=layer_num, alpha=a)
                g_loss    = -d(fake_imgs, layer_num=layer_num, alpha=a).mean()
                g_loss.backward()
                g_optimizer.step()

                EMA(g_running, g)

                if phase == "fadein":
                    alpha = min(1.0, alpha + 1 / total_iters)

                pbar.set_description(
                    f"stage {layer_num} {phase} | d: {d_loss.item():.3f} "
                    f"g: {g_loss.item():.3f} α: {a:.2f}"
                )

                # inside the iteration loop, replace the log block:
                global_step += 1
                if global_step % 500 == 0:
                    wandb.log({
                        "d_loss":      d_loss.item(),
                        "g_loss":      g_loss.item(),
                        "alpha":       a,
                        "layer_num":   layer_num,
                        "phase":       phase,
                        "global_step": global_step,
                    }, step=global_step)

            # --- end of phase: samples + metrics ---
            with torch.no_grad():
                z_sample            = torch.randn(8, 512, device=device)
                sample_imgs         = g(z_sample, layer_num=layer_num, alpha=1.0)
                sample_imgs_ema     = g_running(z_sample, layer_num=layer_num, alpha=1.0)

            show_images(sample_imgs,     stats, f"stage{layer_num}_{phase}_g",   layer_num)
            show_images(sample_imgs_ema, stats, f"stage{layer_num}_{phase}_ema", layer_num)

            # domain metrics on test batch batch vs g_running
            ###evaluate_metrics gets a (160, 2, 121, 201) tensor but at early stages like 1–5 the test loader resizes to 
            ###lower resolutions, so test_batch will be (160, 2, 4, 7) at stage 1 for example. The generator also outputs
            ###at that resolution so the comparison is still valid, just worth being aware of.
            test_imgs = []
            test_iter = iter(test_loader)
            for _ in range(5):  # 5 × 32 = 160 samples
                try:
                    test_imgs.append(next(test_iter))
                except StopIteration:
                    break
            test_batch = torch.cat(test_imgs, dim=0).to(device)
            evaluate_metrics(g_running, test_batch, stats, layer_num, phase, device)

        save_checkpoint(g, d, g_optimizer, d_optimizer, layer_num,
                        Path(f"Resources/checkpoints/stage{layer_num}.pt"))

    wandb.finish()


def get_fixed_season_indices(times, splits):
    import pandas as pd

    df = pd.DataFrame({
        "idx": range(len(times)),
        "time": pd.to_datetime(times)
    })

    df["month"] = df["time"].dt.month
    df["season"] = df["month"].apply(lambda m:
        "DJF" if m in [12,1,2] else
        "MAM" if m in [3,4,5] else
        "JJA" if m in [6,7,8] else "SON")

    test_set = set(splits["test"])

    season_indices = {}
    for _, row in df.iterrows():
        if row["idx"] in test_set:
            s = row["season"]
            if s not in season_indices:
                season_indices[s] = row["idx"]
        if len(season_indices) == 4:
            break

    return season_indices  # dict: {"DJF": idx, ...}

def plot_seasonal_grid(g, solar_np, wind_np, stats, splits, times,
                      layer_num, device):
    import torch.nn.functional as F

    season_map = {
        "DJF": "Winter",
        "MAM": "Spring",
        "JJA": "Summer",
        "SON": "Autumn"
    }

    season_indices = get_fixed_season_indices(times, splits)

    target_h, target_w = STAGE_SIZES[layer_num]

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    col_titles = ["Real Solar", "Fake Solar", "Real Wind", "Fake Wind"]

    # fix noise for reproducibility
    torch.manual_seed(42)

    g.eval()

    z_fixed = torch.randn(1, 512, device=device)

    for row, season_key in enumerate(["DJF", "MAM", "JJA", "SON"]):
        idx = season_indices.get(season_key)
        if idx is None:
            continue
        # --- real sample ---
        real = torch.stack([
            torch.tensor(solar_np[idx], dtype=torch.float32),
            torch.tensor(wind_np[idx],  dtype=torch.float32)
        ], dim=0).unsqueeze(0)  # (1,2,H,W)

        # resize to current stage
        if (target_h, target_w) != (121, 201):
            real = F.interpolate(real, size=(target_h, target_w),
                                mode="bilinear", align_corners=False)

        # normalize to [-1,1]
        real[:, 0] = (real[:, 0] - stats["solarMin"]) / (stats["solarMax"] - stats["solarMin"]) * 2 - 1
        real[:, 1] = (real[:, 1] - stats["windMin"])  / (stats["windMax"]  - stats["windMin"])  * 2 - 1

        real = real.to(device)

        # --- fake sample ---
        with torch.no_grad():
            fake = g(z_fixed, layer_num=layer_num, alpha=1.0)

        # --- denormalize ---
        def denorm(x, vmin, vmax):
            return (x + 1) / 2 * (vmax - vmin) + vmin

        real_solar = denorm(real[0, 0].cpu(), stats["solarMin"], stats["solarMax"])
        fake_solar = denorm(fake[0, 0].cpu(), stats["solarMin"], stats["solarMax"])

        real_wind  = denorm(real[0, 1].cpu(), stats["windMin"], stats["windMax"])
        fake_wind  = denorm(fake[0, 1].cpu(), stats["windMin"], stats["windMax"])

        # --- plot row ---
        # compute shared scales per row
        vmin_s = min(real_solar.min(), fake_solar.min())
        vmax_s = max(real_solar.max(), fake_solar.max())

        vmin_w = min(real_wind.min(), fake_wind.min())
        vmax_w = max(real_wind.max(), fake_wind.max())

        for col, data in enumerate([real_solar, fake_solar, real_wind, fake_wind]):
            ax = axes[row, col]

            if col < 2:  # solar
                im = ax.imshow(data, cmap="viridis", origin="lower",
                            vmin=vmin_s, vmax=vmax_s)
            else:        # wind
                im = ax.imshow(data, cmap="viridis", origin="lower",
                            vmin=vmin_w, vmax=vmax_w)

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(col_titles[col])

            if col == 0:
                ax.set_ylabel(season_map[season_key])

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    path = f"Resources/samples/seasonal_stage{layer_num}.png"
    plt.savefig(path)
    plt.close()

    wandb.log({f"seasonal_stage{layer_num}": wandb.Image(path)})

def get_fixed_test_batch(dataset, indices, device):
    samples = [dataset[i] for i in indices]
    batch = torch.stack(samples, dim=0).to(device)
    return batch


def evaluate_from_checkpoint(checkpoint_path, layer_num):
    wandb.init(project="pggan-energy", name=f"eval_stage{layer_num}")

    solarPaths, windPaths = pathsLists()
    stats  = torch.load(statsPath,  weights_only=False)
    splits = torch.load(splitsPath, weights_only=False)

    dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
    dsWind  = xr.open_mfdataset(windPaths,  combine="by_coords")
    dsSolar, dsWind = xr.align(dsSolar, dsWind)

    solar_np = dsSolar["Solar Energy Potential"].values
    wind_np  = dsWind["Wind Energy Potential"].values

    # --- load ALL test data (stratified across years/seasons) ---
    test_loader = get_dataloader(
        solar_np, wind_np, stats, splits,
        layer_num=layer_num, STAGE_SIZES=STAGE_SIZES,
        batch_size=32, split="test"
    )
    real_imgs = torch.cat(list(test_loader), dim=0).to(device)
    print(f"Evaluating on {real_imgs.size(0)} test samples")

    # --- load model ---
    g = Generator().to(device)
    d = Discriminator().to(device)
    mapping_params, other_params = get_params_with_lr(g)
    g_opt = torch.optim.Adam([
        {'params': other_params,   'lr': 0.001},
        {'params': mapping_params, 'lr': 0.001 * 0.01},
    ], betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(d.parameters(), lr=0.001, betas=(0.0, 0.99))

    load_checkpoint(checkpoint_path, g, d, g_opt, d_opt)
    g.eval()

    torch.manual_seed(42)  # reproducible fakes
    fake_imgs = evaluate_metrics(g, real_imgs, stats, layer_num, "eval_only", device)

    plot_real_vs_fake(real_imgs.cpu(), fake_imgs, stats, layer_num)
    plot_histograms(real_imgs.cpu(), fake_imgs, stats, layer_num)
    plot_metrics_dashboard(real_imgs.cpu(), fake_imgs, stats, layer_num)

    wandb.finish()

if __name__ == "__main__":
    # main()
    evaluate_from_checkpoint(
        Path("Resources/checkpoints/stage6.pt"),
        layer_num=6
    )