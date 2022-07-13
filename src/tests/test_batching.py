from src.model import batchify, tensorsFromPair, get_dataloader
from src.model.Lang import prepare_data


def run():
  i, o, _, test = prepare_data()
  test = [tensorsFromPair(i, o, p) for p in test]

  batched = batchify(test)

  pass


def run_prep():
  i, o, _, test = prepare_data()
  test = [tensorsFromPair(i, o, p) for p in test]

  prepped = get_dataloader(test)

  pass


if __name__ == "__main__":
  # run()
  run_prep()