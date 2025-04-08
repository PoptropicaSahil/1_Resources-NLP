import math
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, d):
        """
        Arguments:
        d: size of embedding dimension
        """
        super().__init__()
        self.d = d
        
        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3*d, bias=False)

    def forward(self, x):
        # compute query, key, and value vectors in batch
        # split the output into separate query, key, and value tensors
        q, k, v  = self.c_attn(x).split(self.d, dim=2)

        # compute the attention matrix and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        # compute output vectors for each token
        y = att @ v
        return y