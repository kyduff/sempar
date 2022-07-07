from src.model import TransformerEnc, TransformerDec

from torch import nn

class TransformerEncDec(nn.Module):
  
  def __init__(self, encoder, decoder):
    super(TransformerEncDec, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.out_vocab_size = decoder.vocab_size

  def forward(self, src, tgt, tgt_mask=None, src_mask=None, **kwargs):
    memory = self.encoder(src, src_mask, **kwargs)
    out = self.decoder(tgt, memory, tgt_mask, **kwargs)
    return out