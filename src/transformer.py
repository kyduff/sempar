import random
import torch

from src.model import tensorsFromPair, triangular_mask, TransformerEncDec
from src import EOS_TOKEN, HIDDEN_SIZE, LR, MAX_LENGTH, SEED, SOS_TOKEN, device
from src.model.TransformerEncoder import TransformerEnc
from src.model.TransformerDecoder import TransformerDec
from src.model.Lang import prepare_data

from torch import nn, optim

criterion = nn.NLLLoss()
random.seed(SEED)


def train(input_lang, output_lang, model, optimizer, tensor_pairs):

  model.train()
  loss = 0
  n_tokens = model.out_vocab_size
  sos = torch.tensor([[SOS_TOKEN]], device=device)

  for pair in tensor_pairs:

    optimizer.zero_grad()

    input_tensor, target_tensor = pair
    x, y = input_tensor.to(device), target_tensor.to(device)

    y = torch.cat((sos, y))
    y_input = y[:-1]
    y_tgt = y[1:]

    y_length = y_input.size(0)
    tgt_mask = triangular_mask(y_length).to(device)

    pred = model(x, y_input, tgt_mask)
    pred = pred.view(-1, y_length, n_tokens) # move batch dim first
    batch_loss = criterion(pred.squeeze(), y_tgt.squeeze())

    batch_loss.backward()
    optimizer.step()

    loss += batch_loss.item()

  return loss


def predict(model, input_tensor):

  model.eval()

  input_tensor = input_tensor.to(device)
  tgt_input = torch.tensor([[SOS_TOKEN]], device=device)

  decoded_indices = []

  for _ in range(MAX_LENGTH):
    tgt_mask = triangular_mask(tgt_input.size(0)).to(device)

    pred = model(input_tensor, tgt_input, tgt_mask=tgt_mask)

    next_item = pred.topk(1)[1].view(-1)[-1].item()
    next_item = torch.tensor([[next_item]], device=device)

    tgt_input = torch.cat((tgt_input, next_item), dim=0)

    if next_item.item() == EOS_TOKEN:
      break
    else:
      decoded_indices.append(next_item.item())

  return decoded_indices


def main():
  input_lang, output_lang, train_data, test_data = prepare_data()

  train_tensors = [
      tensorsFromPair(input_lang, output_lang, p) for p in train_data
  ]
  test_tensors = [
      tensorsFromPair(input_lang, output_lang, p) for p in test_data
  ]

  hidden_size = HIDDEN_SIZE

  encoder = TransformerEnc(input_lang.n_words, hidden_size, 4, 2).to(device)
  decoder = TransformerDec(output_lang.n_words, hidden_size, 4, 2).to(device)
  model = TransformerEncDec(encoder, decoder).to(device)

  optimizer = optim.SGD(model.parameters(), lr=LR)

  for _ in range(20):
    loss = train(input_lang, output_lang, model, optimizer, test_tensors)

  print(loss)


if __name__ == '__main__':
  main()