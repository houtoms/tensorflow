#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/resnet.py --layers=152" \
  32 60,16000MiB_MIN,320000MiB_MAX 64,32000MiB_MIN
