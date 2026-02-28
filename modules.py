import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)              # (B, E, H/P, W/P)
        x = x.flatten(2)              # (B, E, N)
        x = x.transpose(1, 2)         # (B, N, E)
        return x
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_ratio=8/3, bias=False):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated_gate = self.act(gate)
        gated_up_value = activated_gate * up # Element-wise multiplication
        output = self.down_proj(gated_up_value) # Final projection back to the input dimension
        return output

class RoPEPositionalEmbedding(nn.Module):
    def __init__(self, dim, 
                 max_seq_len=256+1, 
                 base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for rotary embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len

            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))  # (T, dim/2)

            self._cos_cached = freqs.cos()  # (T, dim/2)
            self._sin_cached = freqs.sin()  # (T, dim/2)

    
    def forward(self, x, seq_len=None):
        """
        Apply rotary position embeddings to input tensor.
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return apply_rotary_pos_emb(x, self._cos_cached, self._sin_cached)


def apply_rotary_pos_emb(x, cos, sin):
    # x: (B, seq, H, Hd)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    return torch.stack(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
        dim=-1
    ).flatten(-2)