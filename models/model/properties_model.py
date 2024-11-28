import torch.nn as nn
import torch.nn.functional as F


class PropertiesModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 512)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(512, 128)
        self.r2 = nn.ReLU()
        self.out = nn.Linear(128, out_dim)

    def forward(self, x, y):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.out(x)
        return F.mse_loss(x, y), x
