from src.utils.clai.utils.metric import metric_utils

import json

from pathlib import Path

from tqdm import tqdm


filedir = Path(__file__).parent.resolve()
data_file = f'{filedir}/../data/nl2bash.json'


with open(data_file, 'r') as f:
  data : dict = json.load(f)


for datum in tqdm(data.values()):
  cmd = datum['cmd']
  metric_val = metric_utils.compute_metric(cmd, 1.0, cmd)
  assert(0.99 < metric_val < 1.01)
