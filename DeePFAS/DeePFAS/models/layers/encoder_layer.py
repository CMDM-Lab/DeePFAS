import torch.nn as nn

from DeePFAS.models.layers.layer_norm import LayerNorm
from DeePFAS.models.layers.multi_head_attention import MultiHeadAttention
from DeePFAS.models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 ffn_hidden: int,
                 n_head: int,
                 drop_prob: float):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(in_dim=in_dim, 
                                            out_dim=in_dim,
                                            n_head=n_head)

        self.norm1 = LayerNorm(d_model=in_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(in_dim=in_dim, 
                                           hidden_dim=ffn_hidden,
                                           out_dim=in_dim,
                                           drop_prob=drop_prob)

        self.norm2 = LayerNorm(d_model=out_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask, intensity=None):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask, intensity=intensity)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        # dimention reduction ver.
        # x = self.dropout2(x)
        # x = self.output(x + _x)
        # x = self.norm2(x)

        # no reduction ver.
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
