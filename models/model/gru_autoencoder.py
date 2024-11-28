import operator
from queue import PriorityQueue

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.models import GruAutoEncoderConfig
from ..utils.smiles_process import (FINAL_CHAR, INITIAL_CHAR, PAD_CHAR,
                                    VOC_MAP, idx_to_smiles)
from ..utils.SmilesEnumerator import SmilesEnumerator
from .beam import BeamSearchNode

# Reference
# Translation model modified from Spec2mol model here https://github.com/KavrakiLab/Spec2Mol

class TranslationModel(nn.Module):
    def __init__(self, 
                 config: GruAutoEncoderConfig):
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

    def show_candidates(self, config: GruAutoEncoderConfig, canonical_smiles, z=None, device=None):
        batch_candidates = self.beam_search(config.hparams.batch_size, 
                                            config.hparams.max_len,
                                            z=z,
                                            beam_width=config.hparams.beam_size,
                                            topk=config.hparams.topk,
                                            device=device)
        real_smiles = [idx_to_smiles(ids) for ids in canonical_smiles]
        result = pd.DataFrame({'REAL_CAN': real_smiles})
        topk_candidates = [[] for _ in range(config.hparams.topk)]
        match = ['False'] * config.hparams.batch_size
        match_num = 0
        for i, batch in enumerate(batch_candidates):
            correct = False
            for j, candidate in enumerate(batch):
                smiles = idx_to_smiles(torch.tensor(candidate))
                topk_candidates[j].append(smiles)
                if smiles == real_smiles[i]:
                    correct = True
                    match[i] = f'C {j+1}'
            if correct:
                match_num += 1
        result['MATCH'] = match
        match_ratio = match_num / config.hparams.batch_size
        for i in range(len(topk_candidates) - 1, -1, -1):
            result[f'CANDIDATE {i+1}'] = topk_candidates[i]
        print(result.head())
        print(result['CANDIDATE 1'])
        print(result['CANDIDATE 2'])
        print(result['CANDIDATE 3'])
        print(f'Total: {config.hparams.batch_size} Matched: {match_num} Percent Matched {match_ratio}')

    def beam_search(self, batch_size, max_len, z=None, beam_width=4, topk=4, device=None):
        with torch.no_grad():
            decoded_batch = []
            if z is None:
                raise ValueError('z is not exists')
            z = z.to(device)
            # z_0 = z.unsqueeze(1)
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.config.decoder.n_layers, 1, 1)
            for idx in range(batch_size):
                decoder_hidden = h[:, idx, :].unsqueeze(1)
                decoder_input = torch.LongTensor([[VOC_MAP[INITIAL_CHAR]]], device=device)
                endnodes = []
                num_required = min((topk + 1), topk - len(endnodes))

                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
                nodes = PriorityQueue()
                nodes.put((-node.eval(), node))
                qsize = 1
                z_0 = z[idx, :].reshape(1, 1, -1)
                while True:
                    if qsize > 2000: break

                    score, n = nodes.get()
                    decoder_input = n.idx
                    decoder_hidden = n.h_s
                    if n.idx.item() == VOC_MAP[FINAL_CHAR] and n.p_node is not None:
                        endnodes.append((score, n))
                        if len(endnodes) >= num_required: break
                        else: continue

                    emb = self.x_emb(decoder_input)
                    # (1, 1, e_dim + z_dim)
                    decoder_input = torch.cat([emb, z_0], dim=2)
                    o, _ = self.decoder_rnn(decoder_input, decoder_hidden)
                    decoder_input = self.decoder_fc(o.squeeze(1))
                    decoder_input = F.log_softmax(decoder_input, dim=1)
                    log_prob, indices = torch.topk(decoder_input, beam_width)
                    nextnodes = []

                    for new_k in range(beam_width):
                        decoded_t = indices[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()
                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.log_p + log_p, n.l + 1)
                        score = -node.eval()
                        nextnodes.append((score, node))
                    
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))
                    qsize += len(nextnodes) - 1
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for _ in range(topk)]
                
                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.idx)
                    while n.p_node is not None:
                        n = n.p_node
                        utterance.append(n.idx)
                    utterance = utterance[::-1]
                    utterances.append(utterance)
                decoded_batch.append(utterances)
            return decoded_batch

    def decode_candidates(self, batch_candidates, topk):
        topk_candidates = [[] for _ in range(topk)]
        for i, batch in enumerate(batch_candidates):
            for j, candidate in enumerate(batch):
                smiles = idx_to_smiles(torch.tensor(candidate))
                topk_candidates[j].append(smiles)
        result = pd.DataFrame()
        for i in range(len(topk_candidates) - 1, -1, -1):
            result[f'CANDIDATE {i+1}'] = topk_candidates[i]
        print(result.head())
        print(result['CANDIDATE 1'])
        print(result['CANDIDATE 2'])
        print(result['CANDIDATE 3'])
        return topk_candidates