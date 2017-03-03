#!/bin/bash

CIFAR10=../third_party/tensorflow_models/tutorials/image/cifar10

python $CIFAR10/cifar10_multi_gpu_train.py --num_gpus=4 --max_steps=15000
python $CIFAR10/cifar10_eval.py --run_once | grep precision | ./test_result
