import json

import torch
from config import get_device
from data_loader import decode, get_batch

from log_config import script_run_logger, var_chk_logger
from models import BigramLanguageModel

var_chk_logger.debug("start of script real good")
script_run_logger.info("This is yoyoyoyo message")
var_chk_logger.info("This is a model training message")


# hyperparameters
with open("config.json", "r") as f:
    config = json.load(f)

batch_size = config["batch_size"]  # num independent sequences processed in parallel?
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

script_run_logger.info("read the input config values")


@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss of the model for train and validation splits and return the average loss for each split.
    This function does not take any parameters and returns a dictionary containing the average loss for the train and validation splits.
    """
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
# is there a need to assign it to a variable? NO! But we access m.generate in the last line
# print the number of parameters in the model
script_run_logger.info(f"{sum(p.numel() for p in m.parameters()) / 1e6}, M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        script_run_logger.info(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")  # 16*32, 16*32
    # var_chk_logger.debug(f"xb = {xb} shape = {xb.shape}, yb = {yb} shape = {xb.shape}")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=50)[0].tolist()))
