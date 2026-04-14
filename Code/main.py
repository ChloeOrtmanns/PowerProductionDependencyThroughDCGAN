# ---RANDOM-NOTES--------------------------------------------------------------------------------------------------------------
# Most of the code is based off https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# ---IMPORTS-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import wandb

import matplotlib.pyplot as plt

from Models.generator import Generator
from Models.discriminator import Discriminator

# ---VARIABLES-----------------------------------------------------------------------------------------------------------------
statsPath = Path("Resources/stats.pt")
splitsPath = Path("Resources/splits.pt")
pathsPath = Path("Resources/paths.pt")
checkpointPath = ("Resources/checkpoints")

lat = 121
lon = 201

latentDim = 10      # latent dimension
numEpochs = 150       # number of training epochs. You may increase it to gain better results but it will take more time.
batchSize = 32      # -> that means it'll group them by 32 in one batch, so 11x32+13=365 for one year
ngpu = 0
nz = 100            # Size of z latent vector (underlying degrees of freedom)
# |---------When nz is too small => outputs look too similar, when too big => may learn noise instead of structure
ngf = 64            # 32 is trauning is unstable, 128 for more detail
# |---------ngf ↑ → more capacity → better detail → harder training
# |---------ngf ↓ → simpler model → more stable → less expressive
nc = 2
ndf = 32
lr_G = 0.0002
lr_D = 0.0001


print(f"Using Pytorch {torch.__version__}.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}.")

wandb.init(
    name="lr-schedulers,sw-corr-moni,ncritic:5",
    project="dcgan-energy",
    config={
        "nz": nz, "ngf": ngf, "ndf": ndf, "nc": nc,
        "batchSize": batchSize, "numEpochs": numEpochs,
        "lr_G": lr_G, "lr_D": lr_D,
        "beta1": 0.0, "beta2": 0.9
    }
)

# ---DATASET+LOADER------------------------------------------------------------------------------------------------------------
# ---NORMALIZATION-------------------------------------------------------------------------------------------------------------
# neural networks work best when inputs are roughly: mean ~ 0, std ~1. That's why it's important to normalize. If you wouldn't,
# the model could prioritize larger-scale features = unstable/slow training = poorly behaving gradients. 
# IMPORTANT to note is you should compute the stats on training data only and reuse same values for validation/test
class EnergyDataset(Dataset):
    def __init__(self, solarPaths, windPaths, stats=None):                      # Compute constants
        self.dsSolar = xr.open_mfdataset(solarPaths, combine="by_coords")
        self.dsWind  = xr.open_mfdataset(windPaths , combine="by_coords")

        # Align datasets
        self.dsSolar, self.dsWind = xr.align(self.dsSolar, self.dsWind)

        self.solar = self.dsSolar["Solar Energy Potential"]   
        self.wind = self.dsWind["Wind Energy Potential"]

        self.stats = stats

    def __len__(self):
        return self.solar.sizes["time"]

    def __getitem__(self, idx):                                     # Apply transformation per sample
        solarSample = self.solar.isel(time=idx).values
        windSample = self.wind.isel(time=idx).values

        # Convert to tensors
        solarTensor = torch.tensor(solarSample, dtype=torch.float32)
        windTensor = torch.tensor(windSample, dtype=torch.float32)

        if self.stats is not None:
            solarTensor = (solarTensor - self.stats["solarMin"]) / (self.stats["solarMax"] - self.stats["solarMin"])
            solarTensor = solarTensor * 2 - 1  # → [-1, 1]

            windTensor  = (windTensor  - self.stats["windMin"]) / (self.stats["windMax"] - self.stats["windMin"])
            windTensor  = windTensor * 2 - 1  # → [-1, 1]

        # Concatenate along channel dimension
        x = torch.stack([solarTensor, windTensor], dim=0)

        return x

# ---FUNCTIONS-----------------------------------------------------------------------------------------------------------------    
def weights_init(m):                                
    # custom weights initialization called on ``netG`` and ``netD``
    # the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02.
    # The weights_init function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose,
    # and batch normalization layers to meet this criteria. This function is applied to the models immediately after initialization.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.normal_(m.weight_orig if hasattr(m, 'weight_orig') else m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def savePathsLists():
    solarPaths = list(Path("../../../mnt/mvbc-gan/Data/Power/Solar").glob("*.nc"))
    windPaths  = list(Path("../../../mnt/mvbc-gan/Data/Power/Wind").glob("*.nc"))

    torch.save({
    "solarPaths": solarPaths,
    "windPaths": windPaths
}, pathsPath)
    
def pathsLists():
    paths = torch.load(pathsPath, weights_only=False)
    return paths["solarPaths"], paths["windPaths"]

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

def show_generated_samples(generator, stats, epoch, nz=100, device='cpu'):
    generator.eval()  # set to eval mode for inference
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)  # generate 1 sample
        fake = generator(noise)[0]  # remove batch dimension: [2, 121, 201]
        
        # If you normalized to [-1,1], rescale back to original range for plotting
        fake_solar = (fake[0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]
        fake_wind  = (fake[1] + 1) / 2 * (stats["windMax"] - stats["windMin"]) + stats["windMin"]
        
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title("Fake Solar")
        plt.imshow(fake_solar.cpu(), cmap='viridis')
        plt.colorbar()
        
        plt.subplot(1,2,2)
        plt.title("Fake Wind")
        plt.imshow(fake_wind.cpu(), cmap='viridis')
        plt.colorbar()
        
        plt.savefig(f"epoch_{epoch}_sample.png")
        plt.close()
    generator.train()  # back to training mode

def show_real_sample(data_loader, stats, device='cpu'):
    # Get one batch
    real_batch = next(iter(data_loader))
    real_sample = real_batch[0].to(device)  # pick first sample in batch, shape [2,121,201]

    # Rescale back to original range if you normalized to [-1,1]
    real_solar = (real_sample[0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]
    real_wind  = (real_sample[1] + 1) / 2 * (stats["windMax"] - stats["windMin"]) + stats["windMin"]

    # Plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Real Solar")
    plt.imshow(real_solar.cpu(), cmap='viridis')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Real Wind")
    plt.imshow(real_wind.cpu(), cmap='viridis')
    plt.colorbar()

    plt.savefig(f"real_sample.png")
    plt.close()

def save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "netG_state": netG.state_dict(),
        "netD_state": netD.state_dict(),
        "optimizerG_state": optimizerG.state_dict(),
        "optimizerD_state": optimizerD.state_dict(),
    }, filepath)
    print(f"Saved checkpoint → {filepath}")

def load_checkpoint(filepath, netG, netD, optimizerG, optimizerD, device):
    ckpt = torch.load(filepath, map_location=device)
    netG.load_state_dict(ckpt["netG_state"])
    netD.load_state_dict(ckpt["netD_state"])
    optimizerG.load_state_dict(ckpt["optimizerG_state"])
    optimizerD.load_state_dict(ckpt["optimizerD_state"])
    print(f"Loaded checkpoint from {filepath} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]

# for the Weisser GAN
def gradient_penalty(netD, real_data, fake_data, device):
    b_size = real_data.size(0)
    # Random interpolation between real and fake
    alpha = torch.rand(b_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_data + (1 - alpha) * fake_data.detach()).requires_grad_(True)

    score = netD(interpolated)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(b_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def sample_synthetic_days(generator, stats, n_samples, nz=100, device='cpu'):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, nz, 1, 1, device=device)
        fake = generator(noise)  # [n_samples, 2, 121, 201]

        # Denormalize
        solar = (fake[:, 0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]
        wind  = (fake[:, 1] + 1) / 2 * (stats["windMax"] - stats["windMin"]) + stats["windMin"]

    return solar.cpu().numpy(), wind.cpu().numpy()



def main():
    # only run this function once
    # savePathsLists()
    # calculateSaveStatsSplits()

    # ---NORMALIZED-DATASET--------------------------------------------------------------------------------------------------------
    stats = torch.load(statsPath)
    solarPaths, windPaths = pathsLists()
    normalizedDs = EnergyDataset(
        solarPaths,
        windPaths,
        torch.load(statsPath)
    )

    # ---DATASET-SPLITTING---------------------------------------------------------------------------------------------------------
    splits = torch.load(splitsPath)
    trainIndices = splits["train"]
    testIndices  = splits["test"]
    
    trainDataset = Subset(normalizedDs, trainIndices)
    testDataset = Subset(normalizedDs, testIndices)

    # ---LOADER--------------------------------------------------------------------------------------------------------------------
    
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)    
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)

    test_solar_all = []
    test_wind_all  = []
    for batch in testLoader:
        test_solar_all.append(batch[:, 0])
        test_wind_all.append(batch[:, 1])
    test_solar_all = torch.cat(test_solar_all, dim=0)  # [n_test, 121, 201]
    test_wind_all  = torch.cat(test_wind_all,  dim=0)

    total_solar_real = test_solar_all.mean(dim=(1,2)).numpy()
    total_wind_real  = test_wind_all.mean(dim=(1,2)).numpy()
    sw_corr_real     = np.corrcoef(total_solar_real, total_wind_real)[0,1]   # sw_corr_real is now fixed — compute it once, reuse every epoch

    # ---GENERATOR-----------------------------------------------------------------------------------------------------------------
    netG = Generator(ngpu, nz, ngf, nc, lat, lon).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # ---Testing---
    # z = torch.randn(1, nz, 1, 1)  # batch_size=1
    # out = netG(z, debug=True)
    # should be torch.Size([1, 2, 121, 201])

    # ---DISCRIMINATOR------------------------------------------------------------------------------------------------------------
    netD = Discriminator(ngpu, nc, ndf).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)

    # ---Testing---
    # x = torch.randn(1, 2, 121, 201).to(device)
    # out = netD(x, debug=True)

    # print(out.shape)

    # ---TRAININGLOOP-------------------------------------------------------------------------------------------------------------
    show_real_sample(trainLoader, torch.load(statsPath), device)
    
    # ----WGAN-GP----------
    lambda_gp = 10      # gradient penalty weight — standard value from the paper
    n_critic = 5        # critic steps per generator step
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_D, betas=(0.0, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_G, betas=(0.0, 0.9))
    schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    best_score = float('inf')
    patience = 20       # stop if no improvement for this many epochs
    min_delta = 0.01
    epochs_no_improve = 0

    for epoch in range(numEpochs):
        # Accumulators
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_wdist  = 0.0
        epoch_gp     = 0.0
        n_batches    = 0

        for i, real_data in enumerate(trainLoader):

            real_data = real_data.to(device)
            b_size = real_data.size(0)

            # CRITIC
            for _ in range(n_critic):
                netD.zero_grad()
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_data = netG(noise).detach()
                score_real = netD(real_data).mean()
                score_fake = netD(fake_data).mean()
                gp = gradient_penalty(netD, real_data, fake_data, device)
                loss_D = -score_real + score_fake + lambda_gp * gp
                loss_D.backward()
                optimizerD.step()


            # GENERATOR
            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_data = netG(noise)
            loss_G = -netD(fake_data).mean()
            loss_G.backward()
            optimizerG.step()

            # To keep track of the avg of the epoch
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_wdist  += (score_real - score_fake).item()
            epoch_gp     += gp.item()
            n_batches    += 1

        # --- END OF EPOCH EVALUATION ---
        avg_loss_D = epoch_loss_D / n_batches
        avg_loss_G = epoch_loss_G / n_batches
        avg_wdist  = epoch_wdist  / n_batches
        avg_gp     = epoch_gp     / n_batches

        print(
            f"[{epoch}/{numEpochs}] "
            f"Loss_D: {avg_loss_D:.4f}  "
            f"Loss_G: {avg_loss_G:.4f}  "
            f"W-dist: {avg_wdist:.4f}  "
            f"GP: {avg_gp:.4f}"
        )

        wandb.log({
            "loss_D": avg_loss_D,
            "loss_G": avg_loss_G,
            "wasserstein_distance": avg_wdist,
            "gp": avg_gp,
            "epoch": epoch,
        })

        show_generated_samples(netG, torch.load(statsPath), epoch, nz=nz, device=device)
        wandb.log({
            "generated_sample": wandb.Image(f"epoch_{epoch}_sample.png"),
            "epoch": epoch,
        })
        
        if sw_corr_error < best_score - min_delta:
            best_score = sw_corr_error
            epochs_no_improve = 0
            save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, filepath=Path("Resources/checkpoints/best.pt"))
            print(f"New best checkpoint at epoch {epoch} (score={wasserstein_dist:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {patience} epochs")
                break

        # Also save a "latest" every epoch so you can always resume
        save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, filepath=Path("Resources/checkpoints/latest.pt"))
        # wandb.log({"balance_score": balance_score, "epoch": epoch})
        wandb.log({"wasserstein_dist": wasserstein_dist, "epoch": epoch})

        # Quick monitoring metrics — cheap to compute
        solar_synth, wind_synth = sample_synthetic_days(netG, stats, n_samples=500, device=device)
        total_solar_synth = solar_synth.mean(axis=(1,2))
        total_wind_synth  = wind_synth.mean(axis=(1,2))
        sw_corr_synth = np.corrcoef(total_solar_synth, total_wind_synth)[0,1]
        sw_corr_error = abs(sw_corr_real - sw_corr_synth)

        wandb.log({"sw_correlation_error": sw_corr_error, "epoch": epoch})

        # ---Add learning rate schedulers
        schedulerD.step()
        schedulerG.step()
        wandb.log({"lr_G": schedulerG.get_last_lr()[0], "lr_D": schedulerD.get_last_lr()[0]})
        
    wandb.finish()


if __name__ == "__main__":
    main()


