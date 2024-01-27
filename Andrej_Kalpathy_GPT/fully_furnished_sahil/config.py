# import json
import torch

# def load_config(file_path):
#     with open(file_path, 'r') as file:
#         config = json.load(file)
#     return config

# def get_vocab_size():
#     config = load_config('config.json')
#     return config['vocab_size']


def get_device():
    # Check if CUDA is available and set the device accordingly
    return "cuda" if torch.cuda.is_available() else "cpu"
