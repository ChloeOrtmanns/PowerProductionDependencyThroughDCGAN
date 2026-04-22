import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset

class EnergyDataset(Dataset):
    def __init__(self, solar_np, wind_np, stats=None, layer_num=6, stage_sizes=None):                      # Compute constants
        self.solar = solar_np
        self.wind  = wind_np
        self.stats = stats
        self.layer_num = layer_num
        self.stage_sizes = stage_sizes

    def __len__(self):
        return self.solar.shape[0]

    def __getitem__(self, idx):                                     # Apply transformation per sample
        solarTensor = torch.tensor(self.solar[idx], dtype=torch.float32)
        windTensor  = torch.tensor(self.wind[idx],  dtype=torch.float32)

        if self.stats is not None:
            solarTensor = (solarTensor - self.stats["solarMin"]) / (self.stats["solarMax"] - self.stats["solarMin"])
            solarTensor = solarTensor * 2 - 1  # → [-1, 1]

            windTensor  = (windTensor  - self.stats["windMin"]) / (self.stats["windMax"] - self.stats["windMin"])
            windTensor  = windTensor * 2 - 1  # → [-1, 1]

        # Concatenate along channel dimension
        x = torch.stack([solarTensor, windTensor], dim=0)

        # Resize to current progressive stage resolution
        target_h, target_w = self.stage_sizes[self.layer_num]
        if (target_h, target_w) != (121, 201):
            x = x.unsqueeze(0)  # (1, 2, 121, 201) — F.interpolate needs a batch dim
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            x = x.squeeze(0)    # (2, target_h, target_w)

        return x

def get_dataloader(solar_np, wind_np, stats, splits, layer_num,
                   STAGE_SIZES, batch_size=4, split="train"):
    dataset = EnergyDataset(solar_np, wind_np, stats=stats,
                            layer_num=layer_num, stage_sizes=STAGE_SIZES)  # ← lowercase
    indices = splits[split]
    subset  = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size,
                      shuffle=(split == "train"), drop_last=True)