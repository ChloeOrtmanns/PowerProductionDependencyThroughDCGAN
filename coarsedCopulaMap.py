import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

COPULA_OUTPUT_DIR = Path("Resources/FakeData/20260506 - 90/CoarseCoopula5")

solar_synth = np.load(COPULA_OUTPUT_DIR / "solar_synthetic_coarse.npy")
wind_synth  = np.load(COPULA_OUTPUT_DIR / "wind_synthetic_coarse.npy")
solar_test  = np.load(COPULA_OUTPUT_DIR / "solar_test_coarse.npy")
wind_test   = np.load(COPULA_OUTPUT_DIR / "wind_test_coarse.npy")

mean_real_s = solar_test.mean(axis=0)
mean_real_w = wind_test.mean(axis=0)
mean_cop_s  = solar_synth.mean(axis=0)
mean_cop_w  = wind_synth.mean(axis=0)
diff_s      = mean_cop_s - mean_real_s
diff_w      = mean_cop_w - mean_real_w

rows = [
    ("Real (test set)",  mean_real_s, mean_real_w, False),
    ("Coarse Copula",    mean_cop_s,  mean_cop_w,  False),
    ("Copula − Real",    diff_s,      diff_w,      True),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("Spatial Mean Comparison (Coarse) — Real vs Copula", fontsize=14, fontweight="bold")

for ci, ct in enumerate(["Solar Energy Potential", "Wind Energy Potential"]):
    axes[0, ci].set_title(ct, fontsize=11, fontweight="bold")

d_s_lim = abs(diff_s).max()
d_w_lim = abs(diff_w).max()
s_vmin, s_vmax = mean_real_s.min(), mean_real_s.max()
w_vmin, w_vmax = mean_real_w.min(), mean_real_w.max()

for ri, (label, s_map, w_map, is_diff) in enumerate(rows):
    for ci, (data, d_lim, base_vmin, base_vmax) in enumerate([
        (s_map, d_s_lim, s_vmin, s_vmax),
        (w_map, d_w_lim, w_vmin, w_vmax),
    ]):
        ax = axes[ri, ci]
        if is_diff:
            im = ax.imshow(data, cmap="RdBu_r", origin="lower", vmin=-d_lim, vmax=d_lim)
        else:
            im = ax.imshow(data, cmap="viridis", origin="lower", vmin=base_vmin, vmax=base_vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if ci == 0:
            ax.set_ylabel(label, fontsize=9, fontweight="bold")
        ax.axis("off")

plt.tight_layout()
plt.savefig(COPULA_OUTPUT_DIR / "mean_map_coarse_copula.png", dpi=150, bbox_inches="tight")
plt.show()