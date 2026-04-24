import matplotlib.pyplot as plt
from pathlib import Path


"""
row1: plots the first 4 testdatasets from the testloader batch (solar channel)
row2: plots 4 randomly generated datasets (solar channel)
row3: plots the first 4 testdatasets from the testloader batch (wind channel)
row4: plots 4 randomly generated datasets (wind channel)
"""
def plot_real_vs_fake(real_imgs, fake_imgs, stats, layer_num):
    real_imgs = real_imgs.cpu()
    fake_imgs = fake_imgs.cpu()

    n = min(4, real_imgs.size(0))

    fig, axes = plt.subplots(4, n+1, figsize=(4*n+1, 10), gridspec_kw={"width_ratios": [4]*n + [0.3]})

    for i in range(n):

        # --- denormalize ---
        real_solar = (real_imgs[i, 0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]
        fake_solar = (fake_imgs[i, 0] + 1) / 2 * (stats["solarMax"] - stats["solarMin"]) + stats["solarMin"]

        real_wind = (real_imgs[i, 1] + 1) / 2 * (stats["windMax"] - stats["windMin"]) + stats["windMin"]
        fake_wind = (fake_imgs[i, 1] + 1) / 2 * (stats["windMax"] - stats["windMin"]) + stats["windMin"]

        # compute shared scale per sample per channel
        vmin_s = min(real_solar.min(), fake_solar.min())
        vmax_s = max(real_solar.max(), fake_solar.max())
        vmin_w = min(real_wind.min(), fake_wind.min())
        vmax_w = max(real_wind.max(), fake_wind.max())

        im = axes[0, i].imshow(real_solar, cmap="viridis", origin="lower", vmin=vmin_s, vmax=vmax_s)
        axes[1, i].imshow(fake_solar, cmap="viridis", origin="lower", vmin=vmin_s, vmax=vmax_s)
        im_w = axes[2, i].imshow(real_wind,  cmap="viridis", origin="lower", vmin=vmin_w, vmax=vmax_w)
        axes[3, i].imshow(fake_wind,  cmap="viridis", origin="lower", vmin=vmin_w, vmax=vmax_w)

        # --- solar row ---
        axes[0, i].set_title("Real" if i == 0 else "")
        axes[0, i].axis("off")

        axes[1, i].set_title("Fake" if i == 0 else "")
        axes[1, i].axis("off")

        # --- wind row ---
        axes[2, i].axis("off")
        axes[3, i].axis("off")
    
    # solar colorbar — use the top-half of the last column
    fig.colorbar(im,   cax=axes[0, -1])  # just uses row 0 of last col
    fig.colorbar(im_w, cax=axes[2, -1])  # just uses row 2 of last col
    # hide the unused last-column axes
    axes[1, -1].axis("off")
    axes[3, -1].axis("off")

    axes[0, 0].set_ylabel("Solar Real")
    axes[1, 0].set_ylabel("Solar Fake")
    axes[2, 0].set_ylabel("Wind Real")
    axes[3, 0].set_ylabel("Wind Fake")

    plt.tight_layout()

    path = Path(f"Resources/samples/real_vs_fake_plot_stage{layer_num}.png")
    plt.savefig(path)
    plt.close()

    return path