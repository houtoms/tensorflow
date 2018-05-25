#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/vgg.py --layers=16" 32 64,16000MiB_MAX 128,16000MiB_MIN
