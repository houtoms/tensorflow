#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/vgg.py --layers=11" 32 128,12000MiB_MIN
