import random
import time
import torch
import wandb

from src.utils.clai.utils.metric import metric_utils
from src.model import tensorsFromPair, triangular_mask, TransformerEncDec, batchify
from src import BATCH_SIZE, EOS_TOKEN, EPOCHS, HIDDEN_SIZE, LR, MAX_LENGTH, N_GPU, N_HEADS, N_LAYERS, OPTIMIZER, PAD_TOKEN, SEED, SOS_TOKEN, device, WANDB, wandb_config
from src.model.TransformerEncoder import TransformerEnc
from src.model.TransformerDecoder import TransformerDec
from src.model.Lang import prepare_data, Lang
from src.helpers import asMinutes

from torch import nn, optim
from tqdm import tqdm

criterion = nn.NLLLoss()
random.seed(SEED)

if WANDB:
  wandb.init(project="nl2bash", entity="kyduff", config=wandb_config)


def train_loop(model, optimizer, tensor_pairs, verbose=True):

  model.train()
  loss = 0
  n_tokens = model.out_vocab_size

  iterations = tqdm(tensor_pairs, leave=False) if verbose else tensor_pairs
  for pair in iterations:

    optimizer.zero_grad()

    input_tensor, target_tensor = pair
    x, y = input_tensor.to(device), target_tensor.to(device)

    y_input = y[:-1]
    y_tgt = y[1:]

    # Prevent attention to padding
    src_key_mask = (x == PAD_TOKEN).squeeze().t()
    tgt_key_mask = torch.logical_or(y_tgt == PAD_TOKEN,
                                    y_tgt == EOS_TOKEN).squeeze().t()

    y_length = y_input.size(0)
    tgt_mask = triangular_mask(y_length).to(device)

    pred = model(x,
                 y_input,
                 tgt_mask,
                 src_key_padding_mask=src_key_mask,
                 tgt_key_padding_mask=tgt_key_mask)

    # expected dimensions for NLLLoss are N,C,s_1,...,s_n
    pred = pred.view(-1, n_tokens, y_length).squeeze()
    y_tgt = y_tgt.squeeze().view(-1, y_length)

    batch_loss = criterion(pred, y_tgt)

    batch_loss.backward()
    optimizer.step()

    if WANDB:
      wandb.log({"loss": batch_loss})

    loss += batch_loss.item()

  return loss


def train(model, optimizer, tensor_pairs, epochs=EPOCHS):

  try:

    loss = 0.

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
      pbar.set_description(f'Epoch {epoch}/{epochs}')
      loss = train_loop(model, optimizer, tensor_pairs)

  except KeyboardInterrupt:
    print('Terminating training...')


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


def eval(model, test_tensors):

  model.eval()
  metric_vals = []

  try:
    for nlc, cmd in tqdm(test_tensors):

      # nlc_tokens = nlc.squeeze().tolist()
      cmd_tokens = cmd.squeeze().tolist()
      cmd_words = [output_lang.index2word[tok] for tok in cmd_tokens[:-1]]
      cmd = ' '.join(cmd_words)

      pred_tokens = predict(model, nlc)
      pred_cmd = ' '.join(output_lang.index2word[tok] for tok in pred_tokens)

      metric_val = metric_utils.compute_metric(pred_cmd, 1.0, cmd)
      metric_vals.append(metric_val)

  except KeyboardInterrupt:
    print('Terminating eval...')

  return metric_vals


def main():

  global input_lang
  global output_lang

  input_lang, output_lang, train_data, test_data = prepare_data()

  train_tensors = [
      tensorsFromPair(input_lang, output_lang, p) for p in train_data
  ]
  test_tensors = [
      tensorsFromPair(input_lang, output_lang, p) for p in test_data
  ]

  train_tensors = batchify(train_tensors)
  test_tensors = batchify(test_tensors)

  hidden_size = HIDDEN_SIZE

  encoder = TransformerEnc(input_lang.n_words, hidden_size, N_HEADS,
                           N_LAYERS).to(device)
  decoder = TransformerDec(output_lang.n_words, hidden_size, N_HEADS,
                           N_LAYERS).to(device)
  model = TransformerEncDec(encoder, decoder).to(device)

  optimizer = optim.Adam(model.parameters(), lr=LR)

  start = time.time()
  train(model, optimizer, train_tensors)
  elapsed = time.time() - start
  print(f'time: {asMinutes(elapsed)}')

  metric_vals = eval(model, test_tensors)
  print(f'Score: {sum(metric_vals) / len(metric_vals)}')


if __name__ == '__main__':
  main()