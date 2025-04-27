"""
Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
See equations (4)-(6) on page 7
"""

import torch
import torch.nn.functional as F

# constants
B = 16  # batch size
C = 256  # sequence length
n_exp = 8  # number of experts
K = 2  # number of active expert

# define tensors needed to compute load balancing loss
indices = torch.randint(1, n_exp + 1, (B, C, K))  # top-K indices ([B, C, K])
expert_probs = F.softmax(
    torch.rand(B, C, n_exp), dim=2
)  # expert probabilities ([B, C, n_exp])

# equation (5): compute ratio of tokens allocated to each expert
# total number of tokens is defined as total tokens in batch * K
with torch.no_grad():
    one_hot_indices = F.one_hot(indices, num_classes=n_exp)  # [B, C, K, n_exp]
    one_hot_indices = torch.sum(
        one_hot_indices.float(), dim=2
    )  # [B, C, n_exp] (sum over K dimension)
    tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

# equation (6): compute ratio of router probability allocated to each expert
prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

# equation (4): take a scaled dot product between prob / token allocation vectors
# multiply the result by the number of experts
load_balance_loss = n_exp * torch.sum(prob_per_expert * tokens_per_expert)
