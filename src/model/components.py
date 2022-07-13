import torch

from src import BATCH_SIZE, PAD_TOKEN, SOS_TOKEN, device, EOS_TOKEN
from torch.nn.utils.rnn import pad_sequence


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


def batchify(seqs, batch_size=BATCH_SIZE):

  batches = []
  pad_token = PAD_TOKEN

  for i in range(0, len(seqs), batch_size):

    batch = seqs[i:i + batch_size]
    x_batch, y_batch = zip(*batch)

    x_padded = pad_sequence(x_batch, padding_value=pad_token).to(device)
    y_padded = pad_sequence(y_batch, padding_value=pad_token).to(device)

    batches.append((x_padded, y_padded))

  return batches


def get_dataloader(seqs, batch_size=BATCH_SIZE):

  sos = torch.tensor([[SOS_TOKEN]], device=device)

  # Move data to device; add SOS and strip EOS tokens
  seqs = ((x.to(device), y.to(device)) for x, y in seqs)
  seqs = [(x, torch.cat((sos, y))) for x, y in seqs]
  
  return batchify(seqs, batch_size)
