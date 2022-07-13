from src.model import TransformerEnc, TransformerDec

from torch import nn


class TransformerEncDec(nn.Module):

  def __init__(self, encoder, decoder):
    super(TransformerEncDec, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.out_vocab_size = decoder.vocab_size

  def forward(self,
              src,
              tgt,
              tgt_mask=None,
              src_mask=None,
              src_key_padding_mask=None,
              tgt_key_padding_mask=None):

    memory = self.encoder(src,
                          src_mask,
                          src_key_padding_mask=src_key_padding_mask)
    out = self.decoder(tgt,
                       memory,
                       tgt_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask)

    return out