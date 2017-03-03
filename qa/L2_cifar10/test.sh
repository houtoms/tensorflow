#!/bin/bash

cd /usr/local/lib/python2.7/dist-packages/tensorflow/models/image/cifar10

python cifar10_multi_gpu_train.py --num_gpus=4 --max_steps=15000
python cifar10_eval.py --run_once | grep precision | /opt/tensorflow/qa/L2_cifar10/test_result
