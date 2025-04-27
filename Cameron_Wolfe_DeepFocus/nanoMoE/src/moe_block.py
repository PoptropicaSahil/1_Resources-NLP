import torch
from causal_self_attention import CausalSelfAttention
from moe_layer import MOELayer
from torch import nn


class MoEBlock(nn.Module):
    def __init__(
        self,
        d: int,
        H: int,
        C: int,
        n_exp: int,
        top_k: int,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
        bias: bool = False,
        dropout: float = 0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        C: maximum length of input sequences (in tokens)
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()  # type: ignore
        self.ln_1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, H, T, bias, dropout)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = MOELayer(
            d,
            n_exp,
            top_k,
            use_noisy_top_k,
            capacity_factor,
            bias,
            dropout,
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
