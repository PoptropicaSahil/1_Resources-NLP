"""
Based upon ColossalAI OpenMoE: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
"""

import torch
from torch import nn


class MLPExperts(nn.Module):
    def __init__(
        self,
        d: int,
        n_exp: int = 8,
        bias: bool = False,
        dropout: float = 0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()  # type: ignore
        self.bias = bias
        self.c_fc = nn.Parameter(torch.empty(n_exp, d, 4 * d))
        self.c_proj = nn.Parameter(torch.empty(n_exp, 4 * d, d))
        self.fc_bias = nn.Parameter(torch.empty(n_exp, 1, 4 * d)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(n_exp, 1, d)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)

        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)

        return x
