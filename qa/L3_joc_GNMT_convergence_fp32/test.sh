set -o nounset
set -o errexit
set -o pipefail

cd /opt/tensorflow/nvidia-examples/gnmt_v2

# hack to work with pytorch dataset
sed -ie 's/    src_vocab_file = hparams.vocab_prefix + "." + hparams.src/    src_vocab_file = hparams.vocab_prefix/g' nmt.py
sed -ie 's/    tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt/    tgt_vocab_file = hparams.vocab_prefix/g' nmt.py

( python nmt.py --data_dir=/data/pytorch/wmt16_de_en --output_dir=output_dir --batch_size=1024 --use_amp=false --num_gpus=8 --learning_rate=2e-3 2>&1 ) | tee log.log
python scripts/parse_training_log.py log.log | tee log.json

python << END
import json
import numpy as np
from pathlib import Path

baseline = np.array([20.22, 21.78, 22.69, 22.99, 23.70, 24.33])

log = json.loads(Path('log.json').read_text())
bleu = np.array(log['bleu'])

print('Bleu     :', bleu)
print('Baseline :', baseline)

if (bleu < baseline * 0.985).any():
    print("FAILED: bleu ({}) doesn't match the baseline ({})".format(bleu, baseline))
    exit(1)
print('SUCCESS')
END
