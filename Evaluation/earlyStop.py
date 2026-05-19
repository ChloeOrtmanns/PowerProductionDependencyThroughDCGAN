import torch
import numpy
from scipy.stats import wasserstein_distance

def early_stop_metric(g_running, test_loader, layer_num, device, n_batches=2):
    """Cheap val metric for early stopping. Returns avg Wasserstein (lower = better)."""
    test_imgs = []
    test_iter = iter(test_loader)
    for _ in range(n_batches):
        try:
            test_imgs.append(next(test_iter))
        except StopIteration:
            break
    test_batch = torch.cat(test_imgs, dim=0).to(device)

    g_running.eval()
    with torch.no_grad():
        z         = torch.randn(test_batch.size(0), 512, device=device)
        fake_imgs = g_running(z, layer_num=layer_num, alpha=1.0).cpu()
    g_running.train(False)

    test_batch = test_batch.cpu()
    w_solar = wasserstein_distance(
        test_batch[:, 0].numpy().flatten(),
        fake_imgs[:, 0].numpy().flatten()
    )
    w_wind = wasserstein_distance(
        test_batch[:, 1].numpy().flatten(),
        fake_imgs[:, 1].numpy().flatten()
    )
    return (w_solar + w_wind) / 2