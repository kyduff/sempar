import math
import random
import time
from src.model.AttnDecoderRNN import AttnDecoderRNN
from src.model.EncoderRNN import EncoderRNN
from src.model.Lang import prepare_data
from src.model import tensorsFromPair
import torch

from torch import nn, optim

from src import EOS_TOKEN, MAX_LENGTH, SOS_TOKEN, device, LR, EPOCHS
from tqdm import tqdm


import wandb
wandb.init(project="nl2bash", entity="kyduff")
wandb.config = {
    "learning_rate": LR,
    "epochs": EPOCHS,
}

def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    teacher_forcing_ratio=0.5,
    max_length=MAX_LENGTH,
):
  """
  Train Model
  """
  encoder_hidden = encoder.initHidden()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  input_length = input_tensor.size(0)
  target_length = target_tensor.size(0)

  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

  loss = 0.0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_outputs[0, 0]

  decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
  decoder_hidden = encoder_hidden

  use_teacher_forcing = True if random.random(
  ) < teacher_forcing_ratio else False

  if use_teacher_forcing:
    # Enforce target as next input
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
          decoder_input, decoder_hidden, encoder_outputs)
      loss += criterion(decoder_output, target_tensor[di])
      decoder_input = target_tensor[di]  # teacher forcing

  else:
    # use model outputs without teacher forcing
    for di in range(target_length):
      decoder_output, decoder_hidden, decoder_attention = decoder(
          decoder_input, decoder_hidden, encoder_outputs)
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()

      loss += criterion(decoder_output, target_tensor[di])
      if decoder_input.item() == EOS_TOKEN:
        break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.item() / target_length


def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(input_lang,
               output_lang,
               pairs,
               encoder,
               decoder,
               n_iters,
               learning_rate=LR):
  losses = []
  start = time.time()
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  training_pairs = [
      tensorsFromPair(input_lang, output_lang, random.choice(pairs))
      for i in range(n_iters)
  ]
  criterion = nn.NLLLoss()

  for iter in tqdm(range(1, n_iters + 1)):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(input_tensor, target_tensor, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss

    wandb.log({"loss": loss})

  print_loss_avg = print_loss_total / n_iters
  print('%s (%d %d%%) %.4f' % (timeSince(
      start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))


def main():
  print(device)
  input_lang, output_lang, pairs = prepare_data()
  hidden_size = 256
  encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
  attn_decoder1 = AttnDecoderRNN(hidden_size,
                                 output_lang.n_words,
                                 dropout_p=0.1).to(device)

  print('Starting training...')
  trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, EPOCHS)


if __name__ == "__main__":
  main()