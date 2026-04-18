# ---RANDOM-NOTES--------------------------------------------------------------------------------------------------------------
# Most of the code is based off https://ym2132.github.io/Progressive_GAN#the-gulrajani-generator-g-network

# ---IMPORTS-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import wandb

import matplotlib.pyplot as plt

from Models.generator import Generator
from Models.discriminator import Discriminator
from Models.energyDataset import EnergyDataset, get_dataloader

# ---VARIABLES-----------------------------------------------------------------------------------------------------------------
statsPath      = Path("Resources/stats.pt")
splitsPath     = Path("Resources/splits.pt")
pathsPath      = Path("Resources/paths.pt")

batchSize  = 4       # keep small for PGGAN — paper uses 16 max
total_iters = 50000

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
    fullDataset = EnergyDataset(
        solarPaths,
        windPaths,
        stats=None
    )
    trainSize = int(0.8 * len(fullDataset))
    trainIndices = list(range(0, trainSize))
    testIndices  = list(range(trainSize, len(fullDataset)))
    torch.save({
        "train": trainIndices,
        "test": testIndices
        }, splitsPath)

    solarMin = fullDataset.solar.isel(time=trainIndices).min().compute().item()
    solarMax = fullDataset.solar.isel(time=trainIndices).max().compute().item()

    windMin  = fullDataset.wind.isel(time=trainIndices).min().compute().item()
    windMax  = fullDataset.wind.isel(time=trainIndices).max().compute().item()

    torch.save({
        "solarMin": solarMin,
        "solarMax": solarMax,
        "windMin": windMin,
        "windMax": windMax
    }, statsPath)

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
    im0 = axes[0].imshow(solar, cmap='viridis')
    plt.colorbar(im0, ax=axes[0])

    axes[1].set_title("Fake Wind")
    im1 = axes[1].imshow(wind, cmap='viridis')
    plt.colorbar(im1, ax=axes[1])

    path = f"Resources/samples/{tag}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

    wandb.log({tag: wandb.Image(path), "layer_num": layer_num})


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
        name="202604180020",
        config={
            "batchSize":   batchSize,
            "total_iters": total_iters,
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


    # ---MODELS----------------------------------------------------
    g         = Generator().to(device)
    g_running = Generator().to(device)
    d         = Discriminator().to(device)
    g_running.train(False)

    with torch.no_grad():
        test_out = g(2, layer_num=1, alpha=1.0)
        print("G output shape:", test_out.shape)
        print("G output min/max:", test_out.min().item(), test_out.max().item())
        
        test_real = next(iter(get_dataloader(solar_np, wind_np, stats, splits, 1, STAGE_SIZES, batch_size=2)))
        test_real = test_real.to(device)
        test_pred = d(test_real, layer_num=1, alpha=1.0)
        print("D output on real:", test_pred)
        
        test_pred_fake = d(test_out, layer_num=1, alpha=1.0)
        print("D output on fake:", test_pred_fake)

    # The LR is quite high but this is specified in the paper
    # A high LR has the benefit of allowing us to explore more of the gradient space
    g_optimizer = torch.optim.Adam(g.parameters(), lr=0.001, betas=(0.0, 0.99))
    d_optimizer = torch.optim.Adam(d.parameters(), lr=0.001, betas=(0.0, 0.99))

    EMA(g_running, g, 0)

    g_losses = []
    d_losses = []

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

    # ---PROGRESSIVE LOOP--------------------------------------
    for layer_num in range(start_stage, 7):
        alpha = 0.0

        data_loader = get_dataloader(
            solar_np, wind_np, stats, splits,
            layer_num,
            STAGE_SIZES,
            batch_size=batchSize
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

                fake_imgs       = g(batch_size_i, layer_num=layer_num, alpha=a)
                
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
                fake_imgs  = g(batch_size_i, layer_num=layer_num, alpha=a)
                g_loss     = -d(fake_imgs, layer_num=layer_num, alpha=a).mean()
                g_loss.backward()
                g_optimizer.step()

                EMA(g_running, g)

                if phase == "fadein":
                    alpha = min(1.0, alpha + 1 / total_iters)

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                pbar.set_description(
                    f"stage {layer_num} {phase} | d: {d_loss.item():.3f} "
                    f"g: {g_loss.item():.3f} α: {a:.2f}"
                )

                # log to wandb every 100 steps
                if i % 100 == 0:
                    wandb.log({
                        "d_loss":     d_loss.item(),
                        "g_loss":     g_loss.item(),
                        "alpha":      a,
                        "layer_num":  layer_num,
                        "phase":      phase,
                        "step":       i,
                    })

            # samples at end of each phase
            with torch.no_grad():
                sample_imgs     = g(8, layer_num=layer_num, alpha=1.0)
                sample_imgs_ema = g_running(8, layer_num=layer_num, alpha=1.0)
            show_images(sample_imgs,     stats, f"stage{layer_num}_{phase}_g",   layer_num)
            show_images(sample_imgs_ema, stats, f"stage{layer_num}_{phase}_ema", layer_num)

        save_checkpoint(g, d, g_optimizer, d_optimizer, layer_num,
                        Path(f"Resources/checkpoints/stage{layer_num}.pt"))

    wandb.finish()


if __name__ == "__main__":
    main()


