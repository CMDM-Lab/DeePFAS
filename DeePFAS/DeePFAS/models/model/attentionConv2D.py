import torch
from typing import List
import torch.nn as nn
from DeePFAS.models.layers.encoder_layer import EncoderLayer
from DeePFAS.config.models import DeePFASConfig
import torch.nn.functional as F
from DeePFAS.utils.smiles_process import batch_add_pad

class AttentionConv2D(nn.Module):
    def __init__(self, 
                 device: torch.device,
                 config: DeePFASConfig,
                 src_pad_idx: int = 0,
                 output_size: int = 512,
                 num_filters: int = 256,
                 kernel_size: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50],
                 drop_prob: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings=config.encoder.voc_size,
                                      embedding_dim=config.encoder.d_model,
                                      padding_idx=src_pad_idx)

        self.dropout1 = nn.Dropout(p=config.encoder.drop_prob)

        self.src_pad_idx = src_pad_idx
        self.max_len = config.spectrum.max_num_peaks
        self.device = device
        self.layers = nn.ModuleList([EncoderLayer(in_dim=config.encoder.d_model,
                                                  out_dim=config.encoder.d_model,
                                                  ffn_hidden=config.encoder.ffn_hidden,
                                                  n_head=config.encoder.n_head,
                                                  drop_prob=config.encoder.drop_prob)
                                     for i in range(config.encoder.n_layers)])
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, config.encoder.d_model), padding=(k - 2, 0))
            for k in kernel_size
        ])
        self.fc = nn.Linear(len(kernel_size) * num_filters, output_size)
        self.dropout2 = nn.Dropout(drop_prob)
        self.tanh = nn.Tanh()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        return F.max_pool1d(x, x.size(2)).squeeze(2)

    def mask_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, m_z, intensity):
        pad_len = min(max([len(mz) for mz in m_z]), self.max_len)
        src = torch.tensor(batch_add_pad(m_z, pad_len, self.src_pad_idx), device=self.device).long()
        src_mask = self.mask_src_mask(src).to(self.device)
        intensity = torch.tensor(batch_add_pad(intensity, pad_len, 0), device=self.device, dtype=torch.float32)
        x = self.dropout1(self.token_emb(src))
        for layer in self.layers:
            x = layer(x, src_mask, intensity)
        x = x.unsqueeze(1)
        conv_results = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(conv_results, 1)
        x = self.dropout2(x)
        x = self.fc(x)
        return self.tanh(x)
