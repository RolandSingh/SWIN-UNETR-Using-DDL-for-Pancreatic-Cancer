
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class DDLBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w, d = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        x_flat = x_flat + self.fc2(self.act(self.fc1(x_flat)))
        x = x_flat.permute(0, 2, 1).view(b, c, h, w, d)
        return x


class Encoder10WithDDL(nn.Module):
    def __init__(self, original_encoder10):
        super().__init__()
        self.encoder10 = original_encoder10
        self.ddl = None

    def forward(self, x):
        x = self.encoder10(x)

        if self.ddl is None:
            channels = x.shape[1]
            self.ddl = DDLBlock(channels).to(x.device)

        x = self.ddl(x)
        return x


class SwinUNETR_DDL(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        feature_size=48,
        use_checkpoint=True,
    ):
        super().__init__()

        self.model = SwinUNETR(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

        # Inject DDL at bottleneck (after encoder10)
        self.model.encoder10 = Encoder10WithDDL(self.model.encoder10)

    def forward(self, x):
        return self.model(x)
