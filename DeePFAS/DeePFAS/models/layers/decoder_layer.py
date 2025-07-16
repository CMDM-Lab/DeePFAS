
import torch.nn as nn

from DeePFAS.models.layers.layer_norm import LayerNorm
from DeePFAS.models.layers.multi_head_attention import MultiHeadAttention
from DeePFAS.models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):

    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 ffn_hidden: int,
                 n_head: int,
                 drop_prob: float):

        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(in_dim=in_dim,
                                                 out_dim=in_dim, 
                                                 n_head=n_head)
        self.norm1 = LayerNorm(d_model=in_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.enc_dec_attention = MultiHeadAttention(in_dim=in_dim, 
                                                    out_dim=out_dim,
                                                    n_head=n_head)
        self.norm2 = LayerNorm(d_model=out_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(in_dim=out_dim,
                                           out_dim=out_dim,
                                           hidden_dim=ffn_hidden,
                                           drop_prob=drop_prob)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm3 = LayerNorm(d_model=out_dim)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            x = self.dropout2(x) 
            x = self.norm2(x)
            enc = self.linear(enc)

        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x, enc

