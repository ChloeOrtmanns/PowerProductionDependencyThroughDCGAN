from pathlib import Path
import torch
from config import device

def save_checkpoint(g, d, g_optimizer, d_optimizer, g_running, layer_num, path):
    torch.save({
        "g":           g.state_dict(),
        "d":           d.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "g_running":   g_running.state_dict(),
        "layer_num":   layer_num,
    }, path)
    print(f"Saved checkpoint -> {path}")

def load_checkpoint(path, g, d, g_optimizer, d_optimizer, g_running=None):
    ckpt = torch.load(path, weights_only=False)
    g.load_state_dict(ckpt["g"])
    d.load_state_dict(ckpt["d"])
    g_optimizer.load_state_dict(ckpt["g_optimizer"])
    d_optimizer.load_state_dict(ckpt["d_optimizer"])
    if g_running is not None and "g_running" in ckpt:
        g_running.load_state_dict(ckpt["g_running"])