import torch

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

MAX_LENGTH = 10  # max = 57
HIDDEN_SIZE = 256

LR = 0.10
EPOCHS = 1

N_LAYERS = 2
N_HEADS = 4
# N_LAYERS_ENC
# N_LAYERS_DEC
# N_HEADS_ENC
# N_HEADS_DEC

BATCH_SIZE = 3

WANDB = False
SEED = 65536
OPTIMIZER = 'adam'
N_GPU = 1
# EARLY_STOPPING = False

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# POSITIONAL_ENCODING: bool
# ENCODER_TYPE
# DECODER_TYPE
# DATA_NORM
# LAYER_NORM
# BATCH_NORM
# WARMUP
# LOSS_FN

wandb_config = {
    "learning_rate": LR,
    "epochs": EPOCHS,
    "seed": SEED,
    "optimizer": OPTIMIZER,
    "batch_size": BATCH_SIZE,
    "layers": N_LAYERS,
    "heads": N_HEADS,
    "type": "TransformerEncDec",
    "max_length": MAX_LENGTH,
    "n_gpu": N_GPU,
    "hidden_size": HIDDEN_SIZE,
}