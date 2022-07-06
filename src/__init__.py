import torch

SOS_TOKEN = 0
EOS_TOKEN = 1

MAX_LENGTH = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 0.01
EPOCHS = 10000