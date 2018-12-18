#!/bin/bash

exec ./base.sh "/workspace/nvidia-examples/cnn/resnet.py --layers=101" 32 64,12000MiB_MIN
