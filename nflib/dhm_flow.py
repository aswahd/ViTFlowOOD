import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LipSwish(),
            nn.InducedNormLinear(in_channels, hidden_channels),
            nn.LipSwish(),
            nn.InducedNormLinear(hidden_channels, in_channels),
            nn.LipSwish(),
            nn.InducedNormLinear(in_channels, hidden_channels)
        )

    def forward(self, x):
        z = x + self.layers(x)
        log_det = torch.sum(torch.log(torch.abs(1 + self.layers(x).detach())), dim=tuple(range(1, len(z.shape))))
        return z, log_det


class FlowModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Logit transform
        self.logit = nn.Logit()

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(10):
            in_channels = input_size if i == 0 else hidden_channels
            hidden_channels = 256 if i < 6 else 128
            self.blocks.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.LipSwish(),
                nn.InducedNormLinear(in_channels, hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                ResidualBlock(hidden_channels, hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                nn.LipSwish(),
                nn.InducedNormLinear(hidden_channels, in_channels),
                nn.BatchNorm2d(in_channels),
                ResidualBlock(in_channels, hidden_channels),
                nn.BatchNorm2d(hidden_channels)
            ))

    def forward(self, x):
        x = self.logit(x)

        log_det = 0
        for block in self.blocks:
            x = nn.functional.activation_norm(x, activation_fn=nn.LipSwish())
            z, block_log_det = block(x)
            log_det += block_log_det
            x = nn.functional.activation_norm(z, activation_fn=nn.LipSwish())

        return z, log_det

    def reverse(self, z):
        x = z

        for block in reversed(self.blocks):
            x = nn.functional.activation_norm(x, activation_fn=nn.LipSwish())
            x = block.layers(x - block(x)[0])
            x = nn.functional.activation_norm(x, activation_fn=nn.LipSwish())

        x = nn.functional.sigmoid(x)

        log_det = -torch.sum(torch.log(x) + torch.log(1 - x), dim=tuple(range(1, len(z.shape))))

        return x, log_det
