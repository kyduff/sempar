from src.model import tensorFromSentence
from src import EOS_TOKEN, MAX_LENGTH, SOS_TOKEN, device
from src.model.Lang import Lang
import torch

from torch import nn


class EncoderDecoder(nn.Module):

  def __init__(self,
               encoder,
               decoder,
               input_lang: Lang,
               output_lang: Lang,
               max_length: int = MAX_LENGTH):
    super(EncoderDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.input_lang = input_lang
    self.output_lang = output_lang
    self.max_length = max_length

  def predict(self, sentence: str):

    input_lang = self.input_lang
    output_lang = self.output_lang
    encoder = self.encoder
    decoder = self.decoder
    max_length = self.max_length

    with torch.no_grad():
      input_tensor = tensorFromSentence(input_lang, sentence)
      input_length = input_tensor.size()[0]
      encoder_hidden = encoder.initHidden()

      encoder_outputs = torch.zeros(max_length,
                                    encoder.hidden_size,
                                    device=device)

      for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

      decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # SOS

      decoder_hidden = encoder_hidden

      decoded_words = []
      decoder_attentions = torch.zeros(max_length, max_length)

      for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_TOKEN:
          break
        else:
          decoded_words.append(output_lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()

      return decoded_words