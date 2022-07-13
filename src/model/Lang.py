import json
import unicodedata
import random

from pathlib import Path

from src import EOS_TOKEN, MAX_LENGTH, PAD_TOKEN, SEED, SOS_TOKEN

FILEDIR = Path(__file__).parent.resolve()

random.seed(SEED)


class Lang:

  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
    self.n_words = 3  # Count SOS, EOS, PAD and

  def add_sentence(self, sentence):
    for word in sentence.split(' '):
      self.add_word(word)

  def add_word(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1


def unicode_to_ascii(s: str):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def normalize_string(s):
  s = unicode_to_ascii(s.lower().strip())
  return s


def read_data(lang1: str = 'nlc',
              lang2: str = 'cmd',
              reverse: bool = False,
              filepath: str = 'data/nl2bash.json'):

  datapath = Path(FILEDIR, '..', filepath)
  print(f'Reading data...')

  # Read the file and split into lines
  with open(datapath, encoding='utf-8') as infile:
    data: dict[str, dict[str, str]] = json.load(infile)

  # Split every line into pairs and normalize
  pairs = [[normalize_string(s) for s in datum.values()]
           for datum in data.values()]

  # Reverse pairs, make lang instances
  if reverse:
    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)
  else:
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

  return input_lang, output_lang, pairs


def filter_pair(p):
  return (len(p[0].split(' ')) < MAX_LENGTH
          and len(p[1].split(' ')) < MAX_LENGTH)


def filter_pairs(pairs):
  return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(reverse: bool = False, filepath: str = 'data/nl2bash.json'):
  input_lang, output_lang, pairs = read_data(reverse=reverse,
                                             filepath=filepath)
  print(f'Read {len(pairs)} sentence pairs')
  pairs = filter_pairs(pairs)
  print(f'Trimmed to {len(pairs)} sentence pairs')
  print('Counting words...')

  # Prepare train/test split
  random.shuffle(pairs)
  n = int(0.8 * len(pairs))
  train_data = pairs[:n]
  test_data = pairs[n:]

  print(f'Prepared {len(train_data)} training data')
  print(f'Prepared {len(test_data)} test data')

  for pair in pairs:
    input_lang.add_sentence(pair[0])
    output_lang.add_sentence(pair[1])

  print('Counted words:')
  print(input_lang.name, input_lang.n_words)
  print(output_lang.name, output_lang.n_words)
  return input_lang, output_lang, train_data, test_data
