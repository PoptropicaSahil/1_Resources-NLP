"""
Based upon ColossalAI OpenMoE
"""
import torch
from torch import nn
from basic_softmax_router import BasicSoftmaxRouter as Router
from expert_layer import MLPExperts

class MOELayer(nn.Module):
    def __init__(
        self,
        d:int,
        n_exp:int=8,
        top_k:int=2,
        use_noisy_top_k:bool=True,
        capacity_factor:float=1.25,
        bias:bool=False,
        dropout:float=0.2,
    ):
        """
        Arguments:
        d: size of embedding dimension
        n_exp: the number of experts to create in the expert layer
        top_k: the number of active experts for each token
        use_noisy_top_k: whether to add noise when computing expert output
        capacity_factor: used to compute expert capacity
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """

        super().__init__() # type: ignore
        self.router = Router(  # (noisy) top k router
            d=d,
            n_exp=n_exp,
            top_k=top_k,
            use_noisy_top_k=use_noisy_top_k,
            capacity_factor=capacity_factor, # type: ignore # TODO CHECK
        )
        self.experts = MLPExperts(  # group of MLPs (experts)
            d=d,
            n_exp=n_exp,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        B, C, d = x.size()  # track original shape of input
        num_tokens = B * C

        # pass each token through the router
        exp_weight, exp_mask, exp_batches = self.router(x)

        # compute expert output
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, d]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        exp_weight = exp_weight.view(num_tokens, -1)  # [B * C, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, d)  # [n_exp * exp_capacity, d]
        output = exp_weight @ exp_out  # [B * C, d]

        # resize output before return
        return output.view(B, T, d)
