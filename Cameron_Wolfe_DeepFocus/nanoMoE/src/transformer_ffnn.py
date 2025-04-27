"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, d: int, bias: bool = False, dropout: float = 0.2):
        """
        Arguments:
        d: size of embedding dimension
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()  # type: ignore
        self.c_fc = nn.Linear(d, 4 * d, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d, d, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
