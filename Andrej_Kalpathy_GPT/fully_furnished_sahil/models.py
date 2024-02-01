import json

import torch
import torch.nn as nn
from config import get_device
from data_loader import vocab_size
from torch.nn import functional as F

from log_config import script_run_logger, var_chk_logger

# hyperparameters
with open("config.json", "r") as f:
    config = json.load(f)

batch_size = config[
    "batch_size"
]  # how many independent sequences will we process in parallel?
block_size = config["block_size"]  # what is the maximum context length for predictions?
max_iters = config["max_iters"]
eval_interval = config["eval_interval"]
learning_rate = config["learning_rate"]
eval_iters = config["eval_iters"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
dropout = config["dropout"]
device = get_device()

script_run_logger.info("read config variables in models file")
# var_chk_logger.debug(
#     f"n_embd = {n_embd}, n_head = {n_head}, n_layer = {n_layer}, block_size = {block_size}"
# )


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # var_chk_logger.debug(f"input is x = {x},  shape = {x.shape}")
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # var_chk_logger.debug(f"wei as matrix mul = {wei}")
        # var_chk_logger.debug(f"self.tril[:T, :T] == 0 is {self.tril[:T, :T] == 0}")
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        # var_chk_logger.debug(f"matrix multiplication done out shape = {out.shape}")

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        var_chk_logger.debug(f"num_heads = {num_heads}, head_size = {head_size}")
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # var_chk_logger.debug(f"self.heads = {self.heads}")

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # var_chk_logger.debug(f"input is x = {x},  shape = {x.shape}")
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # var_chk_logger.debug(
        #     f"output from all heads concat is out = {out},  shape = {out.shape}"
        # )

        out = self.dropout(self.proj(out))
        # var_chk_logger.debug(
        #     f"output from projection layer is out = {out},  shape = {out.shape}"
        # )

        return out


class FeedFoward(nn.Module):
    """
    a simple linear layer followed by a non-linearity
    making the linear layer go till 4*n_embd  is just by convention
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # var_chk_logger.debug(
        #     f"input to transformer block is x = {x},  shape = {x.shape}"
        # )

        # first layer normalization then self attention
        x = x + self.sa(self.ln1(x))
        # var_chk_logger.debug(f"input to ffwd block is x = {x},  shape = {x.shape}")
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        # var_chk_logger.debug(
        #     f"tok_emb shape = {tok_emb.shape}, pos_emb shape = {pos_emb.shape}"
        # )
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
