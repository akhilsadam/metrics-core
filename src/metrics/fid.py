import torch
from torchmetrics.image.fid import FrechetInceptionDistance


class FIDMetric:
    """Wrapper around FrechetInceptionDistance with utility functions."""

    def __init__(self, reset_real_features=True, normalize=True):
        self.tm_fid = FrechetInceptionDistance(reset_real_features=reset_real_features, normalize=normalize).to(
            "cuda"
        )

    def process(self, x, scale=1):
        """Convert grayscale to RGB for FID computation."""
        y = (x / scale) + 0.5
        y = torch.clamp(y, 0, 1)
        return torch.cat([y, y, y], dim=1)

    def fid(self, x_hat, x, scale):
        """Compute FID between generated and real images."""
        self.tm_fid.reset()
        self.tm_fid.update(self.process(x, scale), real=True)
        self.tm_fid.update(self.process(x_hat, scale), real=False)
        return self.tm_fid.compute()

    def cfid(self, x_hat, x):
        """Compute FID on color images (already RGB)."""
        self.tm_fid.reset()
        self.tm_fid.update(x, real=True)
        self.tm_fid.update(x_hat, real=False)
        return float(self.tm_fid.compute().item())
