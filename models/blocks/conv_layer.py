import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 out_channels: int,
                 downsampling: bool):
        super().__init__()

        self.stride = 1
        if downsampling:
            self.stride = 2
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, hid_channels, 1, self.stride),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            nn.Conv1d(hid_channels, hid_channels, 3, padding=1),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(),
            nn.Conv1d(hid_channels, out_channels, 1),
            nn.ReLU(),
        )

        if in_channels != out_channels:

            self.res = nn.Conv1d(in_channels, out_channels, 1, self.stride)

    def forward(self, x):
        y = self.layer(x)
        if self.res:
            r = self.res(x)
            y = y + r

        return y
