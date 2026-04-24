from pathlib import Path
import torch
from config import device

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