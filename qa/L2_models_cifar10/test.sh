#!/bin/bash

set -e

CIFAR10=../third_party/tensorflow_models/tutorials/image/cifar10
MAXGPUS=$(nvidia-smi -L | wc -l)
[[ 4 -lt $MAXGPUS ]] && GPUS=4 || GPUS=$MAXGPUS
python $CIFAR10/cifar10_multi_gpu_train.py --num_gpus=$GPUS --max_steps=15000
python $CIFAR10/cifar10_eval.py --run_once | grep precision | ./test_result
