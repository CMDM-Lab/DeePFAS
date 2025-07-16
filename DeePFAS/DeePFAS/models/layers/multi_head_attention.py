from torch import nn

from DeePFAS.models.embedding.positional_encode import IntensityRotaryEncoding
from DeePFAS.models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, 
                 in_dim,
                 out_dim,
                 n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.pos_emb = IntensityRotaryEncoding(in_dim // (n_head << 1), use_xpos=True, cache_if_possible=False)
        self.w_q = nn.Linear(in_dim, in_dim)
        self.w_k = nn.Linear(in_dim, in_dim)
        self.w_v = nn.Linear(in_dim, in_dim)
        self.w_concat = nn.Linear(in_dim, out_dim)

    def forward(self, q, k, v, mask=None, intensity=None):

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)
        q, k = self.pos_emb.rotate_queries_and_keys(q, k, intensity.unsqueeze(1))
        out, attention = self.attention(q, k, v, mask=mask)
        
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):

        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor):

        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
