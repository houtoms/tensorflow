#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/resnet.py --layers=152" \
  32 64,16000MiB_MIN
