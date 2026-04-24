from pathlib import Path
import torch
import xarray as xr
import wandb

from config import device, STAGE_SIZES, STATS_PATH, SPLITS_PATH, PATHS_PATH, CHECKPOINT_DIR

from Checkpoints.checkpointUtils import load_checkpoint
from Models.generator import Generator          
from Models.discriminator import Discriminator  
from Models.energyDataset import get_dataloader
from Evaluation.metrics import evaluate_metrics, plot_metrics_dashboard
from Evaluation.histogramPlot import plot_histograms
from Evaluation.realFakePlot import plot_real_vs_fake
from Evaluation.cloudPlots import plot_kde_full

def pathsLists():
    paths = torch.load(PATHS_PATH, weights_only=False)
    return paths["solarPaths"], paths["windPaths"]

def get_params_with_lr(model):
    mapping_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'mapping' in name:  # Adjust this condition based on your actual naming convention
            mapping_params.append(param)
        else:
            other_params.append(param)
    return mapping_params, other_params

def load_checkpoint(filepath, g, d, g_opt, d_opt):
    ckpt = torch.load(filepath, map_location=device)
    g.load_state_dict(ckpt["g_state"])
    d.load_state_dict(ckpt["d_state"])
    g_opt.load_state_dict(ckpt["g_opt"])
    d_opt.load_state_dict(ckpt["d_opt"])
    print(f"Loaded checkpoint from {filepath} (layer {ckpt['layer_num']})")
    return ckpt["layer_num"]

def evaluate_from_checkpoint(checkpoint_path, layer_num):
    wandb.init(project="pggan-energy", name=f"vs_hist_cloudsmall-test", id=wandb.util.generate_id())

    solarPaths, windPaths = pathsLists()
    stats  = torch.load(STATS_PATH,  weights_only=False)
    splits = torch.load(SPLITS_PATH, weights_only=False)

    dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
    dsWind  = xr.open_mfdataset(windPaths,  combine="by_coords")
    dsSolar, dsWind = xr.align(dsSolar, dsWind)

    times = dsSolar["Solar Energy Potential"].time.values

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

    load_checkpoint(CHECKPOINT_DIR/ f"stage6.pt", g, d, g_opt, d_opt)
    g.eval()

    torch.manual_seed(42)  # reproducible fakes
    fake_imgs = evaluate_metrics(g, real_imgs, stats, layer_num, "eval_only", device)

    pathVs = plot_real_vs_fake(real_imgs.cpu(), fake_imgs, stats, layer_num)
    pathHist = plot_histograms(real_imgs.cpu(), fake_imgs, stats, layer_num)
    pathMetrics = plot_metrics_dashboard(real_imgs.cpu(), fake_imgs, stats, layer_num)
    pathCloud = plot_kde_full(real_imgs.cpu(), fake_imgs, stats, layer_num)

    wandb.log({
        "metrics_dashboard": wandb.Image(pathMetrics, caption=
            f"Stage {layer_num} | Wasserstein: lower=better | "
            f"Autocorr: fake should match real | Solar-Wind corr: real≈-0.5 physically expected"),
        f"compare_stage{layer_num}": wandb.Image(pathVs, caption=
            f"Stage {layer_num} | Rows: Real Solar / Fake Solar / Real Wind / Fake Wind | "
            f"Each column is one test sample | colorscale shared per row per sample"),
        f"hist_stage{layer_num}": wandb.Image(pathHist, caption=
            f"Stage {layer_num} | Pixel value distributions in [-1,1] | "
            f"Fake should match Real shape — gaps indicate the generator misses certain value ranges"),
        f"copula_kde_full_stage{layer_num}": wandb.Image(str(pathCloud))
    })

    wandb.finish()


if __name__ == "__main__":
    evaluate_from_checkpoint(
        Path("Resources/checkpoints/stage6.pt"),
        layer_num=6
    )