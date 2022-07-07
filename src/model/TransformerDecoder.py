import torch

from torch import nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder, Embedding
from src import device


class TransformerDec(nn.Module):

  def __init__(self, vocab_size, d_hidden, n_head, n_layers):
    super(TransformerDec, self).__init__()
    self.d_hidden = d_hidden
    self.n_head = n_head
    self.vocab_size = vocab_size
    self.embedding = Embedding(vocab_size, d_hidden)

    self.transformer_layer = TransformerDecoderLayer(d_hidden, n_head)
    self.transformer = TransformerDecoder(self.transformer_layer, n_layers)
    self.out = nn.Linear(d_hidden, vocab_size)
    self.softmax = nn.LogSoftmax(dim=2)

  def forward(self, tgt, memory, tgt_mask=None):
    tgt = self.embedding(tgt)
    tgt = self.transformer(tgt, memory, tgt_mask)
    tgt = self.softmax(self.out(tgt))
    return tgt

