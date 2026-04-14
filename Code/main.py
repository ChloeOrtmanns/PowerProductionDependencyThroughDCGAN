# ---RANDOM-NOTES--------------------------------------------------------------------------------------------------------------
# Most of the code is based off https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# ---IMPORTS-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
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


print(f"Using Pytorch {torch.__version__}.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}.")

wandb.init(
    project="dcgan-energy",
    config={
        "nz": nz, "ngf": ngf, "ndf": ndf, "nc": nc,
        "batchSize": batchSize, "numEpochs": numEpochs,
        "lr_G": 0.0002, "lr_D": 0.00005,
        "beta1": 0.5,
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
        nn.init.normal_(m.weight.data, 0.0, 0.02)
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



def main():
    # only run this function once
    # savePathsLists()
    # calculateSaveStatsSplits()

    # ---NORMALIZED-DATASET--------------------------------------------------------------------------------------------------------
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
    # out = netD(x)

    # print(out.shape)

    # ---TRAININGLOOP-------------------------------------------------------------------------------------------------------------
    show_real_sample(trainLoader, torch.load(statsPath), device)
    
    criterion = nn.BCELoss() # Binary Cross Entropy loss -> can be changed into WGAN

    #  set up two separate optimizers. As specified in the DCGAN paper, both are Adam optimizers with learning rate 0.0002 and Beta1 = 0.5
    beta1 = 0.5
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.00005, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(beta1, 0.999))

    best_score = float('inf')
    patience = 10       # stop if no improvement for this many epochs
    epochs_no_improve = 0

    for epoch in range(numEpochs):
        for i, real_data in enumerate(trainLoader):

            # --- FORMAT BATCH ---
            real_data = real_data.to(device)   # [B, 2, 121, 201]
            b_size = real_data.size(0)

            # --- LABEL SMOOTHING ---
            # Label smoothing is a regularization technique in deep learning that prevents models from becoming overconfident
            # by replacing strict "hard" one-hot encoded targets (e.g., [0,0,1]) with "soft" target probabilities (e.g., [0.5,0.5,0.9]).
            # It improves model generalization and reduces overfitting by encouraging smaller logit gaps, often enhancing test accuracy
            real_labels = torch.full((b_size, 1), 0.9, device=device)  # real=0.9 instead of 1
            fake_labels = torch.full((b_size, 1), 0.0, device=device)  # fake=0

            # --- DISCRIMINATOR ---
            netD.zero_grad()

            # --- Add noise (decays over time) ---
            noise_strength = max(0.1 * (1 - epoch / numEpochs), 0.01)

            real_data_noisy = real_data + noise_strength * torch.randn_like(real_data)
            output_real = netD(real_data_noisy).view(-1, 1)
            # Calculate D's loss on real batch
            loss_real = criterion(output_real, real_labels)

            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake data batch with G
            fake_data = netG(noise)

            fake_data_noisy = fake_data.detach() + noise_strength * torch.randn_like(fake_data)
            
            output_fake = netD(fake_data_noisy).view(-1, 1)
            # Calculate D's loss on fake batch
            loss_fake = criterion(output_fake, fake_labels)

            loss_D = loss_real + loss_fake
            # You can calculate gradients for D in backward pass
            loss_D.backward()
            # Update D
            optimizerD.step()

            # --- GENERATOR ---
            for _ in range(3):  # train G more frequently
                netG.zero_grad()

                # Generate fake again to get fresh gradients
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_data = netG(noise)

                # Perform a forward pass of all-fake batch through D
                output_fake = netD(fake_data).view(-1, 1)
                # Calculate G's loss
                loss_G = criterion(output_fake, real_labels)  # trick D to think fakes are real
                # Calculate gradients for G
                loss_G.backward()
                # Update G
                optimizerG.step()

            # -------------------------------
            if i % 50 == 0:
                # print(f"[{epoch}/{numEpochs}] [{i}/{len(trainLoader)}] "
                #     f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
                print(f"[{epoch}/{numEpochs}] [{i}/{len(trainLoader)}] "
                    f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
                wandb.log({
                    "loss_D": loss_D.item(),
                    "loss_G": loss_G.item(),
                    "epoch": epoch,
                    "step": epoch * len(trainLoader) + i,
                })
        # --- END OF EPOCH EVALUATION ---
        show_generated_samples(netG, torch.load(statsPath), epoch, nz=nz, device=device)
        wandb.log({
            "generated_sample": wandb.Image(f"epoch_{epoch}_sample.png"),
            "epoch": epoch,
        })
        # "Balance score": how far loss_D is from ideal equilibrium (0.7)
        # and penalize if generator is losing badly
        balance_score = abs(loss_D.item() - 0.7) + max(0, loss_G.item() - 4.0)

        if balance_score < best_score:
            best_score = balance_score
            epochs_no_improve = 0
            save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, path=Path("Resources/checkpoints/best/best.pt"))
            print(f"New best checkpoint at epoch {epoch} (score={balance_score:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {patience} epochs")
                break

        # Also save a "latest" every epoch so you can always resume
        save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, path=Path("Resources/checkpoints/latest/latest.pt"))
        wandb.log({"balance_score": balance_score, "epoch": epoch})
        
    wandb.finish()


if __name__ == "__main__":
    main()


