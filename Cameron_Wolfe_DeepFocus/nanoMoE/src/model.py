"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .decoder_only_block import Block


class GPT(nn.Module):
    def __init__(
        self,
        d: int,
        H: int,
        C: int,
        V: int,
        layers: int,
        bias: bool = False,
        dropout: float = 0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        C: maximum length of input sequences (in tokens)
        V: size of the token vocabulary
        layers: number of decoder-only blocks
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__()  # type: ignore
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(V, d),  # token embeddings
                wpe=nn.Embedding(C, d),  # position embeddings
                drop=nn.Dropout(dropout),
                blocks=nn.ModuleList(
                    [Block(d, H, C, bias, dropout) for _ in range(layers)]
                ),
                ln_f=nn.LayerNorm(d),
                head=nn.Linear(d, V, bias=bias),
            )
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx is a [B, C] matrix of token indices
        # targets is a [B, C] matrix of target (next) token indices
        device = idx.device
        _, C = idx.size()  # [B, C]
        pos = torch.arange(0, C, dtype=torch.long, device=device)

        # generate token and position embeddings
        tok_emb = self.transformer.wte(idx)  # [B, C, d]
        pos_emb = self.transformer.wpe(pos)  # [C, d]
        x = self.transformer.drop(tok_emb + pos_emb)

        # pass through all decoder-only blocks
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)  # final layer norm

        if targets is not None:
            # compute the loss if we are given targets
            logits = self.transformer.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # only look at last token if performing inference
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits, loss
