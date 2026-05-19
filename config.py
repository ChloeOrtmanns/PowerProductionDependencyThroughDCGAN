from pathlib import Path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE_SIZES = {
    1: (4,   7),
    2: (8,   13),
    3: (16,  26),
    4: (32,  52),
    5: (64,  104),
    6: (121, 201),
}
BATCH_SIZES = {1: 64, 2: 64, 3: 32, 4: 32, 5: 16, 6: 16}
TOTAL_ITERS = {1: 8000, 2: 10000, 3: 15000, 4: 20000, 5: 40000, 6: 60000}
N_CRITIC = {1: 5, 2: 5, 3: 5, 4: 3, 5: 2, 6: 1}
GP_LAMBDA = {1: 10, 2: 10, 3: 10, 4: 10, 5: 5, 6: 5}
LR = {1: 0.001, 2: 0.001, 3: 0.0005, 4: 0.0005, 5: 0.0001, 6: 0.0001}

STATS_PATH   = Path("Resources/stats.pt")
SPLITS_PATH  = Path("Resources/splits.pt")
PATHS_PATH   = Path("Resources/paths.pt")
CHECKPOINT_DIR = Path("Resources/checkpoints")