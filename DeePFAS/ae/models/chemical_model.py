import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class PropModel(pl.LightningModule):
    def __init__(self,
                 prop_dim,
                 hidden_dim):
        super().__init__()
        # in_dim = properties dim + dim of 4096 bits morgan fps (3 radius)

        self.prop_dim = prop_dim
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.r1 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, prop_dim)

    def forward(self, x, y):
        x = self.r1(self.l1(x))
        x = self.out(x)
        prop_loss = F.mse_loss(x, y)

        return prop_loss
