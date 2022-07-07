import torch

from src import device, EOS_TOKEN


def indexesFromSentence(lang, sentence):
  return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
  indexes = indexesFromSentence(lang, sentence)
  indexes.append(EOS_TOKEN)
  return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
  input_tensor = tensorFromSentence(input_lang, pair[0])
  output_tensor = tensorFromSentence(output_lang, pair[1])
  return input_tensor, output_tensor


# thanks to
# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
def triangular_mask(size) -> torch.tensor:
  mask = torch.tril(torch.ones(size, size) == 1)
  mask = mask.float()
  mask = mask.masked_fill(mask == 0, float('-inf'))
  mask = mask.masked_fill(mask == 1, float(0.))
  return mask
