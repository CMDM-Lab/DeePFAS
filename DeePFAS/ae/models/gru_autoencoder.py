import operator
from queue import PriorityQueue

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae.config.models import MolExtractorConfig
from ae.utils.smiles_process import (FINAL_CHAR, INITIAL_CHAR, PAD_CHAR,
                                    VOC_MAP, idx_to_smiles)
import pytorch_lightning as pl
# Reference
# Translation model modified from Spec2mol model here https://github.com/KavrakiLab/Spec2Mol

class TranslationModel(pl.LightningModule):
    def __init__(self, 
                 config: MolExtractorConfig):
        super().__init__()

        self.config = config
        voc_size = config.embedding.enc_voc_size
        e_dim = config.embedding.e_dim
        self.x_emb = nn.Embedding(voc_size, e_dim, padding_idx=VOC_MAP[PAD_CHAR])
        self.encoder_rnn = nn.GRU(input_size=e_dim,
                                  hidden_size=config.encoder.hid_dim,
                                  num_layers=config.encoder.n_layers,
                                  batch_first=True,
                                  dropout=config.encoder.drop_prob if config.encoder.n_layers > 0 else 0,
                                  bidirectional=config.encoder.bidir)
        last_dim = config.encoder.hid_dim * (2 if config.encoder.bidir else 1)
        self.q_mu = nn.Linear(last_dim, config.embedding.z_dim)
        self.q_logvar = nn.Linear(last_dim, config.embedding.z_dim)

        self.decoder_rnn = nn.GRU(e_dim + config.embedding.z_dim,
                                  config.decoder.hid_dim,
                                  num_layers=config.decoder.n_layers,
                                  batch_first=True,
                                  dropout=config.decoder.drop_prob if config.decoder.n_layers > 0 else 0)
        self.decoder_lat = nn.Linear(config.embedding.z_dim, config.decoder.hid_dim)
        self.decoder_fc = nn.Linear(config.decoder.hid_dim, config.embedding.dec_voc_size)
        self.act = nn.Tanh()
        self.encoder = nn.ModuleList([self.encoder_rnn, self.q_mu, self.act, self.q_logvar])
        self.decoder = nn.ModuleList([self.decoder_rnn, self.decoder_lat, self.decoder_fc])
        self.vae = nn.ModuleList([self.x_emb, self.encoder, self.decoder])

    def forward(self, randomized_smiles, canonical_smiles, h_count=None):

        mu, logvar, z, kl_loss = self.forward_encoder(randomized_smiles)

        if self.config.hparams.ignore_vae:
            loss, pred = self.forward_decoder(canonical_smiles, mu)
        else:
            loss, pred = self.forward_decoder(canonical_smiles, z)

        return mu, logvar, z, kl_loss, loss, pred


    def forward_encoder(self, x):

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)
        _, h = self.encoder_rnn(x, None)
        h = h[-(1 + int(self.config.encoder.bidir)) :]
        h = torch.cat(h.split(1), dim=2).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        mu = self.act(mu)

        if self.config.hparams.ignore_vae:
            z = mu
            kl_loss = 0
        else:
            eps = torch.randn_like(mu, device=h.device)
            z = mu + (logvar / 2).exp() * eps
            kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return mu, logvar, z, kl_loss

    def forward_decoder(self, x, z):

        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(sequences=x,
                                      batch_first=True,
                                      padding_value=VOC_MAP[PAD_CHAR])

        emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, emb.size(1), 1)
        x_input = torch.cat([emb, z_0], dim=2)
        x_input = nn.utils.rnn.pack_padded_sequence(input=x_input,
                                                    lengths=lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)
        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.config.decoder.n_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        y = self.decoder_fc(output)

        loss = F.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)),
                               x[:, 1:].contiguous().view(-1),
                               ignore_index=VOC_MAP[PAD_CHAR])
        return loss, y

    def sample_z_prior(self, n_batch, device):
        return torch.randn(n_batch, self.q_mu.out_features, device=device)

    def sample(self, n_batch, max_len=128, z=None, temp=1.0, topk=0, device=None):

        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch, device)
            z = z.to(device)
            z_0 = z.unsqueeze(1)

            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.config.decoder.n_layers, 1, 1)
            w = torch.tensor(VOC_MAP[INITIAL_CHAR], device=device).repeat(n_batch)
            x = torch.tensor([VOC_MAP[PAD_CHAR]], device=device).repeat(n_batch, max_len)

            x[:, 0] = VOC_MAP[INITIAL_CHAR]
            end_pads = torch.tensor([max_len], device=device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=device)

            for i in range(1, max_len):
                emb = self.x_emb(w).unsqueeze(1)
                # (B, 1, dim)
                x_input = torch.cat([emb, z_0], dim=2)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                # w = torch.multinomial(y, 1)[:, 0]
                w = y.argmax(dim=1)
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == VOC_MAP[FINAL_CHAR]).clone().detach()
                end_pads[i_eos_mask.bool()] = i + 1
                eos_mask = eos_mask | i_eos_mask

            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [idx_to_smiles(i_x) for i_x in new_x]
