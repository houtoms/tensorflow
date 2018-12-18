#!/bin/bash

set -e

GPUS=$(nvidia-smi -L | wc -l)

if [[ $GPUS -ge 2 ]]; then
    python test_tf_nccl_ops.py 2
fi

if [[ $GPUS -ge 4 ]]; then
    python test_tf_nccl_ops.py 4
fi
