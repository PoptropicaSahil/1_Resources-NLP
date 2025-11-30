# YaRN Implementation

The complete YaRN implementation, as used in HuggingFace Transformers, carefully blends interpolation and extrapolation strategies:

The key parameters in YaRN:

- beta_fast (32): Controls high-frequency cutoff for interpolation
- beta_slow (1): Controls low-frequency cutoff for extrapolation
- mscale: Scaling factor for attention temperature (typically 1)
- attention_factor: Temperature scaling applied to embeddings

```py
import torch
import math

def compute_yarn_parameters(dim, max_position_embeddings, base=10000,
                           scale_factor=1, original_max_position_embeddings=2048,
                           beta_fast=32, beta_slow=1, mscale=1):
    """
    Compute YaRN inverse frequencies and attention factor.
    Based on HuggingFace Transformers implementation.
    """

    def get_mscale(scale, mscale=1):
        """Compute the attention temperature scaling."""
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Attention factor for temperature scaling
    attention_factor = get_mscale(scale_factor, mscale)

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Find dimension where a certain number of rotations occur."""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        """Find dimension range for smooth transition between methods."""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        low = max(math.floor(low), 0)
        high = min(math.ceil(high), dim - 1)
        return low, high

    def linear_ramp_factor(min_val, max_val, dim):
        """Create smooth transition mask."""
        if min_val == max_val:
            max_val += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Base frequencies
    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)

    # Two strategies: interpolation (compressed) vs extrapolation (original)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale_factor * pos_freqs)

    # Find transition range based on beta parameters
    # beta_fast=32, beta_slow=1 are the paper's recommended values
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )

    # Blend between interpolation and extrapolation
    inv_freq_mask = 1 - linear_ramp_factor(low, high, dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask) +
        inv_freq_extrapolation * inv_freq_mask
    )

    return inv_freq, attention_factor

class YaRNRope(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000,
                 scale_factor=1, original_max_position_embeddings=2048):
        super().__init__()

        # Compute YaRN parameters
        inv_freq, attention_factor = compute_yarn_parameters(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            scale_factor=scale_factor,
            original_max_position_embeddings=original_max_position_embeddings
        )

        self.register_buffer("inv_freq", inv_freq)
        self.attention_factor = attention_factor

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply attention scaling through RoPE embeddings
        cos = emb.cos() * self.attention_factor
        sin = emb.sin() * self.attention_factor

        return cos, sin


```
