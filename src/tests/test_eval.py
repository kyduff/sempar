from src.utils.clai.utils.metric import metric_utils

import json

from pathlib import Path

from tqdm import tqdm

filedir = Path(__file__).parent.resolve()
data_file = f'{filedir}/../data/nl2bash.json'

with open(data_file, 'r') as f:
  data: dict = json.load(f)


def run(*, epsilon=0.01):

  metric_vals = []

  for i, datum in tqdm(enumerate(data.values())):
    cmd = datum['cmd']
    metric_val = metric_utils.compute_metric(cmd, 1.0, cmd)
    assert (1.0 - epsilon < metric_val < 1.0 + epsilon)
    metric_vals.append(metric_val)

  print(f'Average metric value: {sum(metric_vals) / len(metric_vals)}')
  print(f'Maximum metric value: {max(metric_vals)}')


if __name__ == "__main__":
  run()
