"""
Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
See equation (5) on page 7
"""

import torch

# constants
B = 16  # batch size
C = 256  # sequence length
n_exp = 8  # number of experts

# create input tensor for router z-loss
router_logits = torch.rand(B, C, n_exp)  # [B, C, n_exp]

# exponentiate logits, sum logits of each expert, take log, and square
# code below is equivalent to the following:
# z_loss = torch.exp(router_logits)
# z_loss = torch.sum(z_loss, dim=-1)
# z_loss = torch.log(z_loss) ** 2.0
router_z_loss = torch.logsumexp(router_logits, dim=-1) ** 2.0  # [B, C]

# sum over all tokens and divide by total number of tokens
router_z_loss = torch.mean(router_z_loss)
