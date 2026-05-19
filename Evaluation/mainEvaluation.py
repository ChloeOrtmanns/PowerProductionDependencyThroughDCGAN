"""
Runs some of the python files in Evaluation/ and logs then to wandb. Can be tweaked by commenting certain functions.
"""

from pathlib import Path
import torch
import xarray as xr
import wandb

from config import device, STAGE_SIZES, STATS_PATH, SPLITS_PATH, PATHS_PATH

from Models.generator import Generator
from Models.discriminator import Discriminator
from Models.energyDataset import get_dataloader
from Evaluation.metrics import evaluate_metrics, plot_metrics_dashboard
from Evaluation.histogramPlot import plot_histograms
from Evaluation.cloudPlots import plot_kde_full

CHECKPOINT_DIR = Path("Resources/FakeData/20260507 - 50/")
STATS_PATH = Path("Resources/FakeData/20260507 - 50/stats.pt")
SPLITS_PATH = Path("Resources/FakeData/20260507 - 50/splits.pt")


def pathsLists():
    paths = torch.load(PATHS_PATH, weights_only=False)
    return paths["solarPaths"], paths["windPaths"]


def get_params_with_lr(model):
    mapping_params, other_params = [], []
    for name, param in model.named_parameters():
        if 'mapping' in name:
            mapping_params.append(param)
        else:
            other_params.append(param)
    return mapping_params, other_params


def load_checkpoint(filepath, g, d, g_opt, d_opt):
    ckpt = torch.load(filepath, map_location=device)
    g.load_state_dict(ckpt["g_running"])
    d.load_state_dict(ckpt["d"])
    g_opt.load_state_dict(ckpt["g_optimizer"])
    d_opt.load_state_dict(ckpt["d_optimizer"])
    print(f"Loaded checkpoint from {filepath} (layer {ckpt['layer_num']})")
    return ckpt["layer_num"]


def evaluate_from_checkpoint(checkpoint_path, layer_num):
    wandb.init(project="StyleGAN-full", name="20260507 50 metrics", id=wandb.util.generate_id())

    # --- load data ---
    solarPaths, windPaths = pathsLists()
    stats  = torch.load(STATS_PATH,  weights_only=False)
    splits = torch.load(SPLITS_PATH, weights_only=False)

    dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
    dsWind  = xr.open_mfdataset(windPaths,  combine="by_coords")
    dsSolar, dsWind = xr.align(dsSolar, dsWind)

    times    = dsSolar["Solar Energy Potential"].time.values
    solar_np = dsSolar["Solar Energy Potential"].values
    wind_np  = dsWind["Wind Energy Potential"].values

    # --- load test data ---
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

    # --- generate fakes in batches to avoid OOM ---
    torch.manual_seed(42)
    fake_chunks = []
    with torch.no_grad():
        for i in range(0, real_imgs.size(0), 32):
            z = torch.randn(min(32, real_imgs.size(0) - i), 512, device=device)
            fake_chunks.append(g(z, layer_num=layer_num, alpha=1.0).cpu())
    fake_imgs = torch.cat(fake_chunks, dim=0)
    torch.cuda.empty_cache()

    real_imgs_cpu = real_imgs.cpu()

    # # --- plots & metrics ---
    pathMetrics  = plot_metrics_dashboard(real_imgs_cpu, fake_imgs, stats, layer_num)

    # pathHist     = plot_histograms(real_imgs_cpu, fake_imgs, stats, layer_num)
    # pathCloudAll   = plot_kde_full(real_imgs_cpu, fake_imgs, stats, layer_num, "all")
    # pathCloudSmall = plot_kde_full(real_imgs_cpu, fake_imgs, stats, layer_num, "small")

    wandb.log({
        "metrics_dashboard": wandb.Image(pathMetrics, caption=
            f"Stage {layer_num} | Wasserstein: lower=better | "
            f"Autocorr: fake should match real | Solar-Wind corr: real≈-0.5 physically expected"),
        f"hist_stage{layer_num}": wandb.Image(pathHist, caption=
            f"Stage {layer_num} | Pixel value distributions in [-1,1] | "
            f"Fake should match Real shape"),
        f"copula_kde_full_stage{layer_num}_all":   wandb.Image(pathCloudAll,   caption="KDE of all pixels"),
        f"copula_kde_full_stage{layer_num}_small": wandb.Image(pathCloudSmall, caption="KDE of 5x5 pixels"),
    })

    wandb.finish()


if __name__ == "__main__":
    evaluate_from_checkpoint(
        CHECKPOINT_DIR / "stage6.pt",
        layer_num=6
    )