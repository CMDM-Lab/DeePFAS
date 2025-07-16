import torch
from beartype import beartype
from einops import rearrange, repeat
from torch import Tensor, einsum, nn
from torch.amp import autocast

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):

        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros((max_len, d_model), device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        p_2i = torch.arange(0, d_model, step=2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (p_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (p_2i / d_model)))

    def forward(self, x):

        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]

class IntensityEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):

        super(IntensityEncoding, self).__init__()
        self.encoding = torch.zeros((max_len, d_model), device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        p_2i = torch.arange(0, d_model, step=2, device=device)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (p_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (p_2i / d_model)))

    def forward(self, x, intensity):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :].repeat(batch_size, 1, 1) * intensity.unsqueeze(2)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast(device_type='cuda', enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1., seq_dim=-2):

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)

class IntensityRotaryEncoding(nn.Module):
    @beartype
    def __init__(
        self,
        dim,
        theta=10000,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.,
        theta_rescale_factor=1.,
        seq_before_head_dim=False,
        cache_if_possible=True
    ):
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        self.cache_if_possible = cache_if_possible
        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq

        self.tmp_store('dummy', torch.tensor(0))
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1
        self.interpolate_factor = interpolate_factor

        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store('scale', None)
            return
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def rotate_queries_and_keys(self, q, k, intensity, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        assert self.use_xpos
        seq_len = q.shape[seq_dim]
        freqs = self.forward(intensity, seq_len=seq_len)
        scale = 1.

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    @autocast(device_type='cuda', enabled=False)
    def forward(
        self,
        t: Tensor,
        seq_len=None,
        offset=0
    ):
        should_cache = (
            self.cache_if_possible and \
            not self.learned_freq and \
            exists(seq_len)
        )
        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs
        # print(f'freq :{freqs.shape}')
        # print(f't :{t.shape}')
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        # print(f'freqs: {freqs.shape}')
        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs

if __name__ == '__main__':
    intensity_emb = IntensityRotaryEncoding(dim=8, use_xpos=True)
    # (batch, heads, seq len, dimension of head)
    q = torch.randn(2, 4, 64, 16)
    k = torch.randn(2, 4, 64, 16)
    intensity = torch.randn(2, 64).unsqueeze(1)
    q, k = intensity_emb.rotate_queries_and_keys(q, k, intensity)
    print(k.shape)
