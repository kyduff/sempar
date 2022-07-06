import random
from src.model.Lang import Lang, prepare_data

N = 10


def run():
  _, _, pairs = prepare_data()

  for _ in range(N):
    pair = random.choice(pairs)
    print('nlc >', pair[0])
    print('cmd =', pair[1])
    print()


if __name__ == '__main__':
  run()
