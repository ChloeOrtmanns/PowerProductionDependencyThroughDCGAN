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
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from config import device, STAGE_SIZES, BATCH_SIZES, TOTAL_ITERS, N_CRITIC, GP_LAMBDA, LR
from config import STATS_PATH, SPLITS_PATH, PATHS_PATH, CHECKPOINT_DIR

from Checkpoints.checkpointUtils import save_checkpoint, load_checkpoint
from Models.generator import Generator
from Models.discriminator import Discriminator
from Models.energyDataset import EnergyDataset, get_dataloader
from Models.copulaMapGen import generate_copula_map
from Evaluation.metrics import evaluate_metrics
from Evaluation.earlyStop import early_stop_metric

print(f"Using PyTorch {torch.__version__}.")
print(f"Using device {device}.")

if device.type == "cuda":
    _ = torch.empty(1, device=device).normal_()

filterPercentage = 0.9
SPLITS_PATH = Path("Resources/FakeData/20260506 - 90/splits_90.pt")
STATS_PATH = Path("Resources/FakeData/20260506 - 90/stats_90.pt")

# ---FUNCTIONS-----------------------------------------------------------------------------------------------------------------    
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.003, smoothing=3):
        self.patience  = patience
        self.min_delta = min_delta
        self.smoothing = smoothing  # average last N checks
        self.history   = []
        self.best      = float("inf")
        self.counter   = 0

    def step(self, metric) -> bool:
        self.history.append(metric)
        smoothed = np.mean(self.history[-self.smoothing:])
        if smoothed < self.best - self.min_delta:
            self.best    = smoothed
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def savePathsLists():
    solarPaths = list(Path("../../../mnt/mvbc-gan/Data/Power/Solar").glob("*.nc"))
    windPaths  = list(Path("../../../mnt/mvbc-gan/Data/Power/Wind").glob("*.nc"))

    torch.save({
        "solarPaths": solarPaths,
        "windPaths": windPaths
    }, PATHS_PATH)

def calculateSaveStatsSplits(solar_np, wind_np, times):

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

    torch.save({"train": train_indices, "test": test_indices}, SPLITS_PATH)
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
    }, Path(STATS_PATH))
    print(test_indices)
    print(train_indices)

def pathsLists():
    paths = torch.load(PATHS_PATH, weights_only=False)
    return paths["solarPaths"], paths["windPaths"]

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


def loadAndFilterData():
    # Load once
    solarPaths, windPaths = pathsLists()

    dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
    dsWind  = xr.open_mfdataset(windPaths,  combine="by_coords")
    dsSolar, dsWind = xr.align(dsSolar, dsWind)
    solar_np = dsSolar["Solar Energy Potential"].values  # (T, 121, 201)
    wind_np  = dsWind["Wind Energy Potential"].values    # (T, 121, 201)
    times = dsSolar["Solar Energy Potential"].time.values

    # print(f"Solar min value: {solar_np.min()}")
    # print(f"Solar max value: {solar_np.max()}")
    # print(f"Wind min value:  {wind_np.min()}")
    # print(f"Wind max value:  {wind_np.max()}")
    # print(f"Solar NaNs: {np.isnan(solar_np).sum()}")
    # print(f"Wind NaNs:  {np.isnan(wind_np).sum()}")

    # --- Filter bad days ---
    total_cells = 121 * 201

    zero_counts_solar = (solar_np == 0).sum(axis=(1, 2))
    zero_counts_wind  = (wind_np  == 0).sum(axis=(1, 2))

    solar_zero_mask = (zero_counts_solar / total_cells <= filterPercentage)
    wind_zero_mask  = (zero_counts_wind  / total_cells <= filterPercentage)

    print(f"Solar zero days (>50%): {(~solar_zero_mask).sum()}")
    print(f"Wind zero days (>50%):  {(~wind_zero_mask).sum()}")

    solar_negative_days = (solar_np < 0).any(axis=(1, 2)).sum()
    wind_negative_days  = (wind_np  < 0).any(axis=(1, 2)).sum()

    valid_mask = solar_zero_mask & wind_zero_mask

    solar_np = solar_np[valid_mask]
    wind_np  = wind_np[valid_mask]
    times    = times[valid_mask]

    print(f"Days remaining: {len(solar_np)} / {valid_mask.shape[0]}")
    print(f"Days dropped:   {(~valid_mask).sum()}")
    print(f"Solar days with negative values: {solar_negative_days}")
    print(f"Wind days with negative values:  {wind_negative_days}")

    dsSolar.close()
    dsWind.close()

    return solar_np, wind_np, times

def main():
    #--only run this function once
    # savePathsLists()
    solar_np, wind_np, times = loadAndFilterData()
    calculateSaveStatsSplits(solar_np, wind_np, times)
    
    stats  = torch.load(STATS_PATH,  weights_only=False)
    splits = torch.load(SPLITS_PATH, weights_only=False)

    wandb.init(
        project="StyleGAN-full",
        name=f"202605080200_{filterPercentage: .0%}_continuation",
        config={
            "batchSize":    str(BATCH_SIZES),
            "total_iters":  str(TOTAL_ITERS),
            "n_critics":    str(N_CRITIC),
            "lr":           str(LR),
            "betas":        (0.0, 0.99),
            "gp_lambda":    str(GP_LAMBDA),
            "stages":       6,
        }
    )
    
    #---DATASET-SPLITTING---------------------------------------------------------------------------------------------------------
    ##so right now, I split it into 0.8-0.2 and if I'm correct, i just take the first 80% of the years for training,
    ##and use the remaining years for validation. Idealiter, I want a test with equal seasons and also varying through all the years
    ##so rn I have data from 1975-2005 (i can technically add 2006-2025 to it)

    ##if I'd take 10% instead, so 0.9-0.1, that means every year should have +- 36 days to test on, the rest is training.
    ##divide that through the 4 seasons every year, it means I need 9 random days from every season every year to build my test data on

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

    resume_stage = 6
    resume = True          # set to False for a fresh run

    if resume:
        start_stage = resume_stage
        load_checkpoint(
            Path("Resources/FakeData/20260506 - 90") / f"stage{resume_stage}.pt",
            g, d, g_optimizer, d_optimizer
        )
        global_step = 260000
        # cool down LR for resumed stage
        
    else:
        start_stage = 1
        # before the stage loop
        global_step = 0

    

    # ---PROGRESSIVE LOOP--------------------------------------
    for layer_num in range(start_stage, 7):
        batchSize = BATCH_SIZES[layer_num]
        total_iters = TOTAL_ITERS[layer_num]
        alpha = 0.0

        if resume:
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = LR[6] * 0.5   # halve the LR
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = LR[6] * 0.5
        else:
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = LR[layer_num]
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = LR[layer_num]

        test_loader = get_dataloader(
            solar_np, wind_np, stats, splits,
            layer_num=layer_num, STAGE_SIZES=STAGE_SIZES,
            batch_size=32, split="test"
        )

        data_loader = get_dataloader(
            solar_np, wind_np, stats, splits,
            layer_num, STAGE_SIZES, batch_size=batchSize, split="train"
        )
        dataset = iter(data_loader)
        print(f"\n--- Stage {layer_num}: {STAGE_SIZES[layer_num]} ---")

        for phase in (["stabilise"] if resume else ["fadein", "stabilise"]):
            pbar = tqdm(range(total_iters), desc=f"stage {layer_num} {phase}")

            # --- only arm the stopper for stage 6 stabilise ---          # <-- EARLY STOP
            use_early_stop = (layer_num == 6 and phase == "stabilise")    # <-- EARLY STOP
            stopper = EarlyStopper(patience=10, min_delta=0.003, smoothing=3)  # or even 0.03
            check_every = 4000
            stopped_early  = False                                        # <-- EARLY STOP

            for i in pbar:
                # current alpha: ramps 0→1 during fadein, fixed 1.0 during stabilise
                a = alpha if phase == "fadein" else 1.0

                # --- train D n_critic times ---
                for _ in range(N_CRITIC[layer_num]):
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
                    d_loss = (fake_preds_mean - real_preds_mean) + GP_LAMBDA[layer_num] * GP
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

                # --- early stop check ---                                          # <-- EARLY STOP
                if use_early_stop and (i + 1) % check_every == 0:                 # <-- EARLY STOP
                    val_metric  = early_stop_metric(g_running, test_loader, layer_num, device)
                    should_stop = stopper.step(val_metric)
                    wandb.log({
                        "early_stop/wasserstein": val_metric,
                        "early_stop/counter":     stopper.counter,
                        "early_stop/best":        stopper.best,
                    }, step=global_step)
                    # --- save best checkpoint ---
                    if stopper.counter == 0:   # counter resets to 0 only when a new best is found
                        save_checkpoint(g, d, g_optimizer, d_optimizer, g_running,
                                        layer_num,
                                        CHECKPOINT_DIR / f"stage{layer_num}_best.pt")
                        print(f"[Best] New best Wasserstein: {stopper.best:.4f} — checkpoint saved.")

                    if should_stop:                                                # <-- EARLY STOP
                        print(f"\n[EarlyStop] No improvement for {stopper.patience} "
                            f"checks. Stopping stage 6 stabilise at iter {i+1}.")
                        save_checkpoint(g, d, g_optimizer, d_optimizer, g_running, # <-- EARLY STOP
                                        layer_num,                                 # <-- EARLY STOP
                                        CHECKPOINT_DIR / f"stage{layer_num}_earlystop.pt")
                        stopped_early = True                                       # <-- EARLY STOP
                        break

            if stopped_early:                                                      # <-- EARLY STOP
                break  # skip the remaining phases (none here, but safe to have)  # <-- EARLY STOP

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

        save_checkpoint(g, d, g_optimizer, d_optimizer, g_running, layer_num, CHECKPOINT_DIR / f"stage{layer_num}.pt")


    wandb.finish()


if __name__ == "__main__":
    main()