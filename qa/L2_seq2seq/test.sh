#!/bin/bash

set -e
set -o pipefail

DATA_DIR="/dldata/wmt16_en_de/"
MAX_GPUS=$(nvidia-smi -L | wc -l)
sed -i "s|^data_root *=.*\$|data_root = \"$DATA_DIR\"|" \
    example_configs/text2text/en-de-nmt-small.py
sed -i "s|\"num_gpus\":.*,|\"num_gpus\":$MAX_GPUS,|" \
    example_configs/text2text/en-de-nmt-small.py

python run.py --config_file=example_configs/text2text/en-de-nmt-small.py \
    --mode=train_eval --max_steps=500

echo PASS
exit 0

