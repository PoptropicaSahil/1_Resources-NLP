import torch
from torch import nn
from torch.nn import functional as F


class BasicSoftmaxRouter(nn.Module):
    def __init__(
        self,
        d:int,
        n_exp:int=8,
        top_k:int=2,
        use_noisy_top_k:bool=True,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        """

        super().__init__() # type: ignore

        # router settings
        self.top_k = top_k
        assert self.top_k >= 1 and self.top_k <= n_exp
        self.use_noisy_top_k = use_noisy_top_k

        # linear projection for (noisy) softmax routing
        # no bias used, see page 4 eq (4) in https://arxiv.org/abs/1701.06538
        self.w_g = nn.Linear(d, n_exp, bias=False)
        self.w_noise = nn.Linear(d, n_exp, bias=False) if self.use_noisy_top_k else None

    def forward(self, x: torch.Tensor):
        # eq (4) in https://arxiv.org/abs/1701.06538
        logits = self.w_g(x)  # [B, C, d] -> [B, C, n_exp]
        if self.use_noisy_top_k:
            # (optionally) add noise into the router
            noise = F.softplus(self.w_noise(x)) # type: ignore
            noise *= torch.randn_like(noise)
            logits += noise
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [B, C, k]
        return top_k_logits, top_k_indices
