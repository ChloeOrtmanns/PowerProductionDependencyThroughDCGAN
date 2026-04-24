import numpy as np
from scipy.stats import gaussian_kde, norm
from scipy.stats import multivariate_normal
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

def plot_kde_full(real_imgs, fake_imgs, stats, layer_num):
    typeSize = "small"
    # typeSize = "all"

    """denorm(): Reverses the [-1, 1] normalization back to original physical units (e.g. W/m²).
	:param x:		value of a channel of the image tensor
	:param vmin:	total minimum value over the dataset
	:param vmax:	total maximun value over the dataset
	:returns:		the inverted x_norm = 2*(x - vmin)/(vmax - vmin) - 1
	"""
    def denorm(x, vmin, vmax):
        return (x + 1) / 2 * (vmax - vmin) + vmin

    ###Extracts channel 0 (solar) and channel 1 (wind) from the image tensors and denormalizes them. Shape is (B, H, W) per variable
    real_solar_map = denorm(real_imgs[:, 0], stats["solarMin"], stats["solarMax"])
    real_wind_map  = denorm(real_imgs[:, 1], stats["windMin"],  stats["windMax"])
    fake_solar_map = denorm(fake_imgs[:, 0], stats["solarMin"], stats["solarMax"])
    fake_wind_map  = denorm(fake_imgs[:, 1], stats["windMin"],  stats["windMax"])

    if (typeSize == "small"):
        ###Defines a center point on the spatial grid and a radius of 2 pixels around it
        cy, cx = 60, 100
        r = 2
        def patch_mean(maps, cy, cx, r):
            return maps[:, cy-r:cy+r+1, cx-r:cx+r+1].mean(dim=(1,2)).numpy()
        """cross variable dependence"""
        ###For each image in the batch, crops a 5×5 patch around (cy, cx) and averages it spatially
        real_s  = patch_mean(real_solar_map, cy, cx, r)
        real_w  = patch_mean(real_wind_map,  cy, cx, r)
        fake_s  = patch_mean(fake_solar_map, cy, cx, r)
        fake_w  = patch_mean(fake_wind_map,  cy, cx, r)
        """spatial dependence"""
        cx_west, cx_east = 75, 125
        real_sA = patch_mean(real_solar_map, cy, cx_west, r)
        real_sB = patch_mean(real_solar_map, cy, cx_east, r)
        fake_sA = patch_mean(fake_solar_map, cy, cx_west, r)
        fake_sB = patch_mean(fake_solar_map, cy, cx_east, r)
    else:
        """cross variable dependence"""
        ###Full map area mean
        real_s = real_solar_map.mean(dim=(1,2)).numpy()   
        real_w = real_wind_map.mean(dim=(1,2)).numpy()
        fake_s = fake_solar_map.mean(dim=(1,2)).numpy()
        fake_w = fake_wind_map.mean(dim=(1,2)).numpy()
        """spatial dependence"""
        real_sA = real_solar_map[:, :, :67].mean(dim=(1,2)).numpy()      # west lon
        real_sB = real_solar_map[:, :, 134:].mean(dim=(1,2)).numpy()     # east lon
        fake_sA = fake_solar_map[:, :, :67].mean(dim=(1,2)).numpy()
        fake_sB = fake_solar_map[:, :, 134:].mean(dim=(1,2)).numpy()

	### GAUSSIAN COPULA SAMPLER
    """sample_gaussian_copula():fits a copula to (xA, xB) and draws new samples from it, mapped back to the original marginal distributions via ECDF inversion
								The copula captures the dependency structure independently of the marginal distributions.
	:param xA:			parameter 1, eg solar energy
	:param xB:			parameter 2, eg wind energy
	:param n_samples:	the amount of samples that have to be generated
	:returns:			the final copula samples in the original physical units, with the original marginals but the fitted dependency structure.
	"""
    def sample_gaussian_copula(xA, xB, n_samples):
        n = len(xA)
        eps = 1e-6

        ###PIT: Probability Integral Transform - converts raw values to uniform [0,1] using their empirical ranks. Dividing by n+1 avoids exact 0 or 1
        uA = rankdata(xA) / (n + 1)
        uB = rankdata(xB) / (n + 1)

        ###uniform -> gaussian - Maps uniform values to standard normal space via the normal quantile function (inverse CDF)
        zA = norm.ppf(np.clip(uA, eps, 1 - eps))
        zB = norm.ppf(np.clip(uB, eps, 1 - eps))

        ###Estimates the single correlation parameter ρ of the bivariate normal in copula space.
        rho = np.corrcoef(zA, zB)[0, 1]
        cov = np.array([[1, rho], [rho, 1]])

        ###Sample from fitted bivariate normal - Draws new samples from the fitted bivariate normal, then maps them back to [0,1] uniform space using the normal CDF.
        z_samples = multivariate_normal(mean=[0, 0], cov=cov).rvs(n_samples)
        u_samples = norm.cdf(z_samples)  # back to uniform space

        ###invert empirical marginals (quantile mapping back to original scale) - for each sampled uniform value, find the closest real data quantile
        def invert_ecdf(u_new, x_ref):
            quantiles = np.sort(x_ref)
            indices = (u_new * len(quantiles)).astype(int).clip(0, len(quantiles) - 1)
            return quantiles[indices]
        sA_copula = invert_ecdf(u_samples[:, 0], xA)
        sB_copula = invert_ecdf(u_samples[:, 1], xB)

        return sA_copula, sB_copula

    n_samples = len(real_s)

    ###Panel 1: copula samples for (solar, wind)
    cop_s, cop_w = sample_gaussian_copula(real_s, real_w, n_samples)

    ###Panel 2: copula samples for (solar_west, solar_east)
    cop_sA, cop_sB = sample_gaussian_copula(real_sA, real_sB, n_samples)

	###PLOTTING TIME!!!
    """make_kde_grid_3(): Builds a 2D KDE density grid for three paired datasets over a shared spatial domain.
	:param xA:	channel1 of first dataset
	:param yA:	channel2 of first dataset
	:param xB:	ch1 if 2nd ds
	:param yB:	ch2 of 2nd ds
	:param xC:	ch1 of 3th ds
	:param yC:	ch2 of 3th ds
	:param n:	size of evaluation grid (standard on 80)
	:returns:	
	"""
    def make_kde_grid_3(xA, yA, xB, yB, xC, yC, n=80j):
        x_min = min(xA.min(), xB.min(), xC.min())
        x_max = max(xA.max(), xB.max(), xC.max())
        y_min = min(yA.min(), yB.min(), yC.min())
        y_max = max(yA.max(), yB.max(), yC.max())
		##creates the grid covering the range of all three datasets
        xx, yy = np.mgrid[x_min:x_max:n, y_min:y_max:n]
        pos = np.vstack([xx.ravel(), yy.ravel()])
		##Fits a 2D KDE to dataset A and evaluates it at every grid point. Repeated for B and C.
        zz_A = gaussian_kde(np.vstack([xA, yA]))(pos).reshape(xx.shape)
        zz_B = gaussian_kde(np.vstack([xB, yB]))(pos).reshape(xx.shape)
        zz_C = gaussian_kde(np.vstack([xC, yC]))(pos).reshape(xx.shape)
        return xx, yy, zz_A, zz_B, zz_C

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Copula Dependency Analysis — Stage {layer_num}", fontsize=13)

    """cross-variable dependence"""
    xx, yy, zz_real, zz_fake, zz_cop = make_kde_grid_3(
        real_s, real_w, fake_s, fake_w, cop_s, cop_w
    )
    rho_real = np.corrcoef(real_s,  real_w)[0, 1]
    rho_fake = np.corrcoef(fake_s,  fake_w)[0, 1]
    rho_cop  = np.corrcoef(cop_s,   cop_w)[0, 1]

    axes[0].contourf(xx, yy, zz_real, levels=6, cmap="Blues",   alpha=0.7)
    axes[0].contour( xx, yy, zz_fake, levels=6, colors="purple", linewidths=1.2)
    axes[0].contour( xx, yy, zz_cop,  levels=6, colors="green",  linewidths=1.2,
                     linestyles="dashed")
    axes[0].set_title(f"Cross-Variable Dependence\n"
                      f"Real ρ={rho_real:.2f} | StyleGAN ρ={rho_fake:.2f} | "
                      f"Copula ρ={rho_cop:.2f}")
    axes[0].set_xlabel("Solar Energy Potential (patch mean)")
    axes[0].set_ylabel("Wind Energy Potential (patch mean)")

    """spatial dependence"""
    xx, yy, zz_real, zz_fake, zz_cop = make_kde_grid_3(
        real_sA, real_sB, fake_sA, fake_sB, cop_sA, cop_sB
    )
    rho_real_sp = np.corrcoef(real_sA, real_sB)[0, 1]
    rho_fake_sp = np.corrcoef(fake_sA, fake_sB)[0, 1]
    rho_cop_sp  = np.corrcoef(cop_sA,  cop_sB)[0, 1]

    axes[1].contourf(xx, yy, zz_real, levels=6, cmap="Blues",   alpha=0.7)
    axes[1].contour( xx, yy, zz_fake, levels=6, colors="purple", linewidths=1.2)
    axes[1].contour( xx, yy, zz_cop,  levels=6, colors="green",  linewidths=1.2,
                     linestyles="dashed")
    axes[1].set_title(f"Spatial Dependence (West vs East Solar)\n"
                      f"Real ρ={rho_real_sp:.2f} | StyleGAN ρ={rho_fake_sp:.2f} | "
                      f"Copula ρ={rho_cop_sp:.2f}")
    axes[1].set_xlabel("Solar — West Belgium (patch mean)")
    axes[1].set_ylabel("Solar — East Belgium (patch mean)")

    ##Legend
    fig.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.7, label="Real Data"),
        Line2D([0],[0], color="purple", linewidth=1.5, label="StyleGAN Generated"),
        Line2D([0],[0], color="green",  linewidth=1.5, label="Gaussian Copula",
               linestyle="dashed"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    path = Path(f"Resources/samples/copula_kde_full_stage{layer_num}_{typeSize}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return path