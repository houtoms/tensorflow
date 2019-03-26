set -o nounset
set -o errexit
set -o pipefail

cd /opt/tensorflow/nvidia-examples/gnmt_v2

# hack to work with pytorch dataset
sed -ie 's/    src_vocab_file = hparams.vocab_prefix + "." + hparams.src/    src_vocab_file = hparams.vocab_prefix/g' nmt.py
sed -ie 's/    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt/    tgt_vocab_file = hparams.vocab_prefix/g' nmt.py

( python nmt.py --data_dir=/data/pytorch/wmt16_de_en --output_dir=output_dir --batch_size=1024 --num_gpus=8 --learning_rate=2e-3 --use_amp=false --max_train_epochs=1 2>&1 ) | tee log.log
python scripts/parse_training_log.py log.log | tee log.json

python << END
import json
import numpy as np
from pathlib import Path

baseline = 83336.93

log = json.loads(Path('log.json').read_text())
speed = np.mean(log['training_tokens_per_sec'])

print('Training speed :', speed)
print('Baseline       :', baseline)

if speed < baseline * 0.9:
    print("FAILED: speed ({}) doesn't match the baseline ({})".format(speed, baseline))
    exit(1)
print('SUCCESS')
END
