#!/bin/bash

set -e

CIFAR10=../third_party/tensorflow_models/tutorials/image/cifar10
GPUS=$(nvidia-smi -L 2>/dev/null| wc -l || echo 1)
[[ $GPUS -gt 4 ]] && GPUS=4
python $CIFAR10/cifar10_multi_gpu_train.py --num_gpus=$GPUS --max_steps=15000
python $CIFAR10/cifar10_eval.py --run_once | grep precision | ./test_result
