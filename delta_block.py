
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaBottleneck3D(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.k_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels)
        )

        self.v_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels)
        )

        self.beta_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W, D = x.shape
        N = H * W * D

        X = x.view(B, C, N)

        k = self.k_proj(x)
        k = F.normalize(k, dim=1).unsqueeze(-1)

        v = self.v_proj(x).unsqueeze(-1)

        beta = 2.0 * self.beta_proj(x)
        beta = beta.view(B, 1, 1)

        kTX = torch.bmm(k.transpose(1, 2), X)
        delta = v.transpose(1, 2) - kTX
        rank1 = torch.bmm(k, delta)

        X_new = X + beta * rank1

        return X_new.view(B, C, H, W, D)
