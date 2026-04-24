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
BATCH_SIZES = {1: 64, 2: 64, 3: 32, 4: 16, 5: 8, 6: 4}
TOTAL_ITERS = {1: 10000, 2: 15000, 3: 25000, 4: 40000, 5: 50000, 6: 75000}

SEED        = 42

STATS_PATH   = Path("Resources/stats.pt")
SPLITS_PATH  = Path("Resources/splits.pt")
PATHS_PATH   = Path("Resources/paths.pt")
CHECKPOINT_DIR = Path("Resources/checkpoints")