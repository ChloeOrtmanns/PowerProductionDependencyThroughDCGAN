from pathlib import Path
import matplotlib.pyplot as plt

"""
x-axis: -1 means minimum energy potential, 0 is the midpoint, 1 means maximum
y-axis: how frequently values in that range appear
The area under the curve sums to 1
"""
def plot_histograms(real_imgs, fake_imgs, stats, layer_num):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ch, name in enumerate(["solar", "wind"]):
        real_vals = real_imgs[:, ch].cpu().numpy().flatten()
        fake_vals = fake_imgs[:, ch].cpu().numpy().flatten()

        axes[ch].hist(real_vals, bins=80, alpha=0.5, label="Real", density=True)
        axes[ch].hist(fake_vals, bins=80, alpha=0.5, label="Fake", density=True)
        axes[ch].set_title(name)
        axes[ch].legend()

    path = Path(f"Resources/samples/hist_stage{layer_num}.png")
    plt.savefig(path)
    plt.close()

    return path