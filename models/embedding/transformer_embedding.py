
import torch
import torch.nn as nn

from ..embedding.positional_encode import PositionalEncoding
from ..embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):

    def __init__(self,
                 d_model: int,
                 max_len: int,
                 voc_size: int,
                 drop_prob: float,
                 padding_idx: int,
                 device: torch.device):

        super(TransformerEmbedding, self).__init__()

        self.device = device
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = TokenEmbedding(voc_size, d_model, padding_idx)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x, intensity=None):
        tok_emb = self.tok_emb(x)
        pos_emb = self.peak_embedding(intensity) if intensity else self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

    def peak_embedding(self, intensity):
        max_len = self.max_len
        d_model = self.d_model
        encoding = torch.zeros((max_len, d_model), device=self.device)
        pos = torch.arange(0, max_len, device=self.device)
        pos = pos.float().unsqueeze(1)
        p_2i = torch.arange(0, d_model, step=2, device=self.device)
        encoding[:, 0::2] = torch.sin(intensity * pos / (10000 ** (p_2i / d_model)))
        encoding[:, 1::2] = torch.cos(intensity * pos / (10000 ** (p_2i / d_model)))
        return encoding
