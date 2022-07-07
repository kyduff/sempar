import torch

from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Embedding
from src import device


class TransformerEnc(nn.Module):

  def __init__(self, vocab_size, d_hidden, n_head, n_layers):
    super(TransformerEnc, self).__init__()
    self.n_head = n_head
    self.embedding = Embedding(vocab_size, d_hidden)

    self.transformer_layer = TransformerEncoderLayer(d_hidden, n_head)
    self.transformer = TransformerEncoder(self.transformer_layer, n_layers)

  def forward(self, src, mask=None):
    src = self.embedding(src)
    src = self.transformer(src, mask)
    return src