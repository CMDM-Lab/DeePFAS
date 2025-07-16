import torch.nn.functional as F
import torch.nn as nn
class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, inputs, targets):
        return F.cosine_similarity(inputs, targets, dim=-1)
