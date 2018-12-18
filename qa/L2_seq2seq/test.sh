#!/bin/bash

set -e
set -o pipefail

PYMAJ=$(python -c 'import sys; print(sys.version_info[0])')
if [[ $PYMAJ -eq 2 ]]; then
  echo "Open Seq2Seq requires Python 3. Skipping test."
  exit 0
fi

cd /opt/tensorflow/nvidia-examples/OpenSeq2Seq

DATA_DIR="/data/wmt16_en_de/"
GPUS=$(nvidia-smi -L 2>/dev/null| wc -l || echo 1)
sed -i "s|^data_root *=.*\$|data_root = \"$DATA_DIR\"|" \
    example_configs/text2text/en-de/en-de-nmt-small.py
sed -i "s|\"num_gpus\":.*,|\"num_gpus\":$GPUS,|" \
    example_configs/text2text/en-de/en-de-nmt-small.py

python run.py --config_file=example_configs/text2text/en-de/en-de-nmt-small.py \
    --mode=train_eval --max_steps=500

echo PASS
exit 0

