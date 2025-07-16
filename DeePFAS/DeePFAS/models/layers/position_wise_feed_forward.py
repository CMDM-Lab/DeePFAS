import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 out_dim: int, 
                 hidden_dim: int, 
                 drop_prob=0.1):

        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
