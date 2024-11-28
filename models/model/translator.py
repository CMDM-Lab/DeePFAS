import torch
import torch.nn as nn

from ..config.models import AutoEncoderConfig
from .ms2embedding import MS2VEC
from .transformer import Transformer


class Translator(nn.Module):

    def __init__(self,
                 auto_encoder: Transformer,
                 config: AutoEncoderConfig,
                 src_pad_idx,
                 trg_pad_idx,
                 trg_sos_idx,
                 trg_eos_idx,
                 model: MS2VEC = None):
        super(Translator, self).__init__()
        self.alpha = 0.7
        self.beam_size = config.hparams.beam_size
        self.auto_encoder = auto_encoder
        self.model = model
        self.max_seq_len = config.hparams.max_len
        self.src_pad_idx = src_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.trg_eos_idx = trg_eos_idx

        # self.model.eval()
        self.auto_encoder.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_sos_idx] + [trg_pad_idx] * (config.hparams.max_len - 1)]))
        self.register_buffer(
            'blank_seqs',
            torch.full((self.beam_size, self.max_seq_len), trg_pad_idx, dtype=torch.long)
        )
        self.blank_seqs[:, 0] = self.trg_sos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, self.max_seq_len + 1, dtype=torch.long).unsqueeze(0)
        )

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = self.auto_encoder.mask_trg_mask(trg_seq)
        dec_output = self.auto_encoder.decoder(trg_seq, enc_output, trg_mask, None)
        return dec_output

    def _get_init_state(self, src_seq, mode='train'):
        beam_size = self.beam_size

        dec_output, enc_output = self.auto_encoder(src_seq, self.init_seq, mode=mode)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        assert src_seq.size(0) == 1

        trg_eos_idx = self.src_pad_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            enc_output, gen_seq, scores = self._get_init_state(src_seq)

            ans_idx = 0
            for step in range(2, max_seq_len):
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, scores)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)


                eos_locs = gen_seq == trg_eos_idx
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)

                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    _, topk_idx = scores.div(seq_lens.float() ** alpha).topk(beam_size)
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return [gen_seq[idx][:seq_lens[idx]] for idx in topk_idx]
