import json
from urllib.request import urlretrieve

import torch
from config import get_device

torch.manual_seed(1337)

# os.chdir("./")
# print(f'os.getcwd() is {os.getcwd()}')

# Construct the full path to config.json
# current_directory = os.path.dirname(os.path.realpath(__file__))
# config_path = os.path.join(current_directory, "config.json")


# hyperparameters
with open("config.json", "r") as f:
    config = json.load(f)

batch_size = config[
    "batch_size"
]  # how many independent sequences will we process in parallel?
block_size = config["block_size"]  # what is the maximum context length for predictions?
device = get_device()

# ------------

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "./inputs/input.txt",  # type: ignore
)


with open("./inputs/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):  # noqa: E741
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    allowed_values = ["train", "val"]
    if split not in allowed_values:
        raise ValueError("Invalid split value. Allowed values: " + str(allowed_values))

    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
