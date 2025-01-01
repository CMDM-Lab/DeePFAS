from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.conv_layer import ConvLayer
from ..blocks.encoder_layer import EncoderLayer
from ..config.models import Ms2VecConfig
from ..utils.smiles_process import batch_add_pad


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0

class MS2VEC(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 trg_sos_idx,
                 device,
                 config: Ms2VecConfig):

        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.src_emb = nn.Embedding(num_embeddings=config.encoder.voc_size,
                                    embedding_dim=config.encoder.d_model,
                                    padding_idx=src_pad_idx)

        self.transformer = nn.Transformer(d_model=config.encoder.d_model,
                                          nhead=config.encoder.n_head,
                                          num_encoder_layers=config.encoder.n_layers,
                                          dim_feedforward=config.encoder.ffn_hidden,
                                          num_decoder_layers=config.decoder.n_layers,
                                          dropout=config.encoder.drop_prob,
                                          batch_first=True)
        self.drop_out = nn.Dropout(config.encoder.drop_prob)
        self.linear = nn.Linear(config.decoder.voc_size)
        self.device = device
        self.config = config

    def forward(self, src, trg, intensity):
        src_emb = self.src_emb(src)
        pos_emb = self.peak_embedding(intensity=intensity, 
                                      d_model=self.config.encoder.d_model,
                                      max_len=self.config.encoder.voc_size)
        src = self.drop_out(src_emb + pos_emb)
        # trg shape: (batch, seq_len)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[1])
        src_mask = self.mask_src_mask(src)
        output = self.transformer(
            src=src,
            tgt=trg,
            src_key_padding_mask=src_mask,
            tgt_mask=trg_mask
        )
        return self.linear(output)

    def mask_src_mask(self, src):
        return (src != self.src_pad_idx)

    def peak_embedding(self, 
                       intensity,
                       d_model: int,
                       max_len: int):

        encoding = torch.zeros((max_len, d_model), device=self.device)
        pos = torch.arange(0, max_len, device=self.device)
        pos = pos.float().unsqueeze(1)
        p_2i = torch.arange(0, d_model, step=2, device=self.device)
        encoding[:, 0::2] = torch.sin(intensity * pos / (10000 ** (p_2i / d_model)))
        encoding[:, 1::2] = torch.cos(intensity * pos / (10000 ** (p_2i / d_model)))
        return encoding

class AttentionConv(nn.Module):
    def __init__(self, 
                 device: torch.device,
                 config: Ms2VecConfig,
                 src_pad_idx: int):
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=config.encoder.voc_size,
                                      embedding_dim=config.encoder.d_model,
                                      padding_idx=src_pad_idx)

        self.dropout = nn.Dropout(p=config.encoder.drop_prob)

        self.src_pad_idx = src_pad_idx
        self.layers = nn.ModuleList([EncoderLayer(in_dim=config.encoder.d_model,
                                                  out_dim=config.encoder.d_model,
                                                  ffn_hidden=config.encoder.ffn_hidden,
                                                  n_head=config.encoder.n_head,
                                                  drop_prob=config.encoder.drop_prob)
                                     for i in range(config.encoder.n_layers)])
        self.convs = nn.ModuleList([ConvLayer(in_channels=config.encoder.d_model - 64 * i,
                                              hid_channels=config.encoder.d_model - 32 * (i + 1),
                                              out_channels=config.encoder.d_model - 64 * (i + 1),
                                              downsampling=True)
                                    for i in range(config.conv.n_layers)])
        print(f'embedding num params: {count_parameters(self.token_emb)}')
        print(f'self att num params: {count_parameters(self.layers)}')
        print(f'convs num params: {count_parameters(self.convs)}')
        self.out = nn.Linear(config.spectrum.max_num_peaks >> config.conv.n_layers, 1)
        self.act = nn.Tanh()
        self.max_len = config.spectrum.max_num_peaks
        self.device = device

    def forward(self, m_z, intensity):
        # x : (batch_size, seq_len, ids)
        # lengths = [len(i_x) for i_x in m_z]
        x = torch.tensor(batch_add_pad(m_z, self.max_len, self.src_pad_idx), device=self.device)
        # x = nn.utils.rnn.pad_sequence(sequences=m_z,
        #                               batch_first=True,
        #                               padding_value=self.src_pad_idx)
        src_mask = self.mask_src_mask(x)
        intensity = torch.tensor(batch_add_pad(intensity, self.max_len, 0), device=self.device)
        # print(intensity.shape)
        # intensity = nn.utils.rnn.pad_sequence(sequences=intensity,
        #                                       batch_first=True,
        #                                       padding_value=1)

        x = self.dropout(self.token_emb(x))
        for layer in self.layers:
            x = layer(x, src_mask, intensity)

        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        # x = x.transpose(1, 2)
        # return F.max_pool1d(x, x.size(-1), 1).squeeze(2)
        return self.act(self.out(x).squeeze(2))

    def mask_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

def conv_out_dim(l_in, kernel, stride, padding, dilation):
    l_out = (l_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    return l_out

class Net1D(nn.Module):
    def __init__(self, in_dim, config: Ms2VecConfig):
        super().__init__()
        out_1 = conv_out_dim(in_dim, 
                             config.conv.conv_kernel_dim_1, 
                             config.conv.conv_stride_1, 
                             config.conv.conv_padding_1, 
                             config.conv.conv_dilation)

        out_2 = conv_out_dim(out_1, 
                             config.conv.pool_kernel_dim_1, 
                             config.conv.pool_stride_1, 
                             config.conv.pool_padding_1, 
                             config.conv.pool_dilation)

        out_3 = conv_out_dim(out_2, 
                             config.conv.conv_kernel_dim_2, 
                             config.conv.conv_stride_2, 
                             config.conv.conv_padding_2, 
                             config.conv.conv_dilation)

        out_4 = conv_out_dim(out_3, 
                             config.conv.pool_kernel_dim_2, 
                             config.conv.pool_stride_2, 
                             config.conv.pool_padding_2,
                             config.conv.pool_dilation)

        self.cnn_out = config.conv.channels_out * out_4
        print(f'{out_1}, {out_2}, {out_3}, {out_4}')

        self.conv1 = nn.Conv1d(config.conv.channels_in, 
                               config.conv.channels_med_1, 
                               config.conv.conv_kernel_dim_1, 
                               stride=config.conv.conv_stride_1, 
                               padding=config.conv.conv_padding_1, 
                               dilation=config.conv.conv_dilation)

        self.norm1 = nn.BatchNorm1d(config.conv.channels_med_1)

        self.pool1 = nn.MaxPool1d(config.conv.pool_kernel_dim_1, 
                                  stride=config.conv.pool_stride_1, 
                                  padding=config.conv.pool_padding_1, 
                                  dilation=config.conv.pool_dilation)

        self.pool2 = nn.MaxPool1d(config.conv.pool_kernel_dim_2, 
                                  stride=config.conv.pool_stride_2, 
                                  padding=config.conv.pool_padding_2, 
                                  dilation=config.conv.pool_dilation)

        self.conv2 = nn.Conv1d(config.conv.channels_med_1, 
                               config.conv.channels_out, 
                               config.conv.conv_kernel_dim_2, 
                               stride=config.conv.conv_stride_2, 
                               padding=config.conv.conv_padding_2, 
                               dilation=config.conv.conv_dilation)

        self.norm2 = nn.BatchNorm1d(config.conv.channels_out)
        self.fc1 = nn.Linear(config.conv.channels_out * out_4, config.conv.fc_dim_1)
        self.fc2 = nn.Linear(config.conv.fc_dim_1, config.conv.emb_dim)
        self.norm3 = nn.BatchNorm1d(config.conv.fc_dim_1)

    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.view(-1 , self.cnn_out)
        x = F.relu(self.norm3(self.fc1(x)))
        x = torch.tanh(self.fc2(x))
        return x

class AttentionConv2D(nn.Module):
    def __init__(self, 
                 device: torch.device,
                 config: Ms2VecConfig,
                 src_pad_idx: int,
                 output_size: int = 512,
                 num_filters: int = 100, 
                 kernel_size: List[int] = [3, 4, 5], 
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
        src = torch.tensor(batch_add_pad(m_z, pad_len, self.src_pad_idx), device=self.device)
        src_mask = self.mask_src_mask(src).to(self.device)
        intensity = torch.tensor(batch_add_pad(intensity, pad_len, 0), device=self.device)
        x = self.dropout1(self.token_emb(src))
        for layer in self.layers:
            x = layer(x, src_mask, intensity)
        x = x.unsqueeze(1)
        conv_results = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(conv_results, 1)
        x = self.dropout2(x)
        x = self.fc(x)
        return self.tanh(x)

class Conv1D(nn.Module):
    def __init__(self, in_dim, config: Ms2VecConfig):
        super().__init__()
        out_1 = conv_out_dim(in_dim, 
                             config.conv.conv_kernel_dim_1, 
                             config.conv.conv_stride_1, 
                             config.conv.conv_padding_1, 
                             config.conv.conv_dilation)

        out_2 = conv_out_dim(out_1, 
                             config.conv.pool_kernel_dim_1, 
                             config.conv.pool_stride_1, 
                             config.conv.pool_padding_1, 
                             config.conv.pool_dilation)

        out_3 = conv_out_dim(out_2, 
                             config.conv.conv_kernel_dim_2, 
                             config.conv.conv_stride_2, 
                             config.conv.conv_padding_2, 
                             config.conv.conv_dilation)

        out_4 = conv_out_dim(out_3, 
                             config.conv.pool_kernel_dim_2, 
                             config.conv.pool_stride_2, 
                             config.conv.pool_padding_2,
                             config.conv.pool_dilation)

        self.cnn_out = config.conv.channels_out * out_4
        print(f'{out_1}, {out_2}, {out_3}, {out_4}')

        self.conv1 = nn.Conv1d(config.conv.channels_in, 
                               config.conv.channels_med_1, 
                               config.conv.conv_kernel_dim_1, 
                               stride=config.conv.conv_stride_1, 
                               padding=config.conv.conv_padding_1, 
                               dilation=config.conv.conv_dilation)

        self.norm1 = nn.BatchNorm1d(config.conv.channels_med_1)

        self.pool1 = nn.MaxPool1d(config.conv.pool_kernel_dim_1, 
                                  stride=config.conv.pool_stride_1, 
                                  padding=config.conv.pool_padding_1, 
                                  dilation=config.conv.pool_dilation)

        self.pool2 = nn.MaxPool1d(config.conv.pool_kernel_dim_2, 
                                  stride=config.conv.pool_stride_2, 
                                  padding=config.conv.pool_padding_2, 
                                  dilation=config.conv.pool_dilation)

        self.conv2 = nn.Conv1d(config.conv.channels_med_1, 
                               config.conv.channels_out, 
                               config.conv.conv_kernel_dim_2, 
                               stride=config.conv.conv_stride_2, 
                               padding=config.conv.conv_padding_2, 
                               dilation=config.conv.conv_dilation)

        self.norm2 = nn.BatchNorm1d(config.conv.channels_out)
        self.fc1 = nn.Linear(config.conv.channels_out * out_4, config.conv.fc_dim_1)
        self.fc2 = nn.Linear(config.conv.fc_dim_1, config.conv.emb_dim)
        self.norm3 = nn.BatchNorm1d(config.conv.fc_dim_1)

    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.view(-1 , self.cnn_out)
        x = F.relu(self.norm3(self.fc1(x)))
        x = torch.tanh(self.fc2(x))
        return x