import json

import torch
from config import get_device
from data_loader import decode, get_batch
from models import BigramLanguageModel

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
max_new_tokens = config["max_new_tokens"]
device = get_device()
# ------------


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=50)[0].tolist()))
