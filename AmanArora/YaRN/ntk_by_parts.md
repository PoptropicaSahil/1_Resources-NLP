# Implementation: Combining Three Strategies

The corrected method intelligently blends three RoPE variants based on frequency

> "NTK-Aware Interpolation "by Parts" Correction." <https://github.com/jquesnelle/scaled-rope/pull/1>.

```py
import torch
import math

def find_correction_factor(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Find dimension threshold for a target number of rotations."""
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dimension range for smooth transition between methods."""
    low = math.floor(find_correction_factor(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_factor(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)

def linear_ramp_mask(min, max, dim):
    """Create smooth transition mask between scaling methods."""
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

class NTKByPartsRope(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1,
                 ntk_factor=1, extrapolation_factor=1, original_max_position_embeddings=2048):
        super().__init__()

        # Interpolation constants found experimentally for LLaMA
        beta_0 = 1.25   # Start transition to NTK
        beta_1 = 0.75   # End transition to NTK
        gamma_0 = 16    # Start transition to extrapolation
        gamma_1 = 2     # End transition to extrapolation

        # Three different RoPE scaling strategies
        inv_freq_base = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq_linear = 1.0 / (scale * (base ** (torch.arange(0, dim, 2).float() / dim)))

        # NTK scaling
        ntk_base = base * scale ** (dim / (dim-2))
        inv_freq_ntk = 1.0 / (ntk_base ** (torch.arange(0, dim, 2).float() / dim))

        # Blend Linear and NTK based on frequency
        low, high = find_correction_range(beta_0, beta_1, dim, base, original_max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, dim // 2)) * ntk_factor
        inv_freq = inv_freq_linear * (1 - inv_freq_mask) + inv_freq_ntk * inv_freq_mask

        # Blend with extrapolation for very low frequencies
        low, high = find_correction_range(gamma_0, gamma_1, dim, base, original_max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, dim // 2)) * extrapolation_factor
        inv_freq = inv_freq * (1 - inv_freq_mask) + inv_freq_base * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq)

```
