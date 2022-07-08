import torch

SOS_TOKEN = 0
EOS_TOKEN = 1

MAX_LENGTH = 57 # max = 57
HIDDEN_SIZE = 256


LR = 0.10
EPOCHS = 10

N_LAYERS = 2
N_HEADS = 4
# N_LAYERS_ENC
# N_LAYERS_DEC
# N_HEADS_ENC
# N_HEADS_DEC

BATCH_SIZE = 1

WANDB = True
SEED = 65536
OPTIMIZER = 'adam'
N_GPU = 1
# EARLY_STOPPING = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ENCODER_TYPE

# DECODER_TYPE


# DATA_NORM

# LAYER_NORM

# BATCH_NORM

# WARMUP
