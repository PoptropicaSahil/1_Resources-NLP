import math
import torch
from torch import nn
import torch.nn.functional as F

class CrossAttention(nn.Module):

    def __init__(self, d):
        """
        Arguments:
        d: size of embedding dimension
        """
        super().__init__()
        self.d = d
        
        # linear projection for producing query matrix
        self.w_q = nn.Linear(d, d, bias=False)
        
        # linear projection for producing key / value matrices
        self.w_kv = nn.Linear(d, 2*d, bias=False)

    def forward(self, x_1, x_2):
        # compute query, key, and value matrices
        q = self.w_q(x_1)
        k, v = self.w_kv(x_2).split(self.d, dim=2)

        # compute the attention matrix and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        # compute output vectors for each token in x_1
        y = att @ v
        return y