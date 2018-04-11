#!/bin/bash

set -e
N=$(nvidia-smi -L | wc -l)
mpiexec --allow-run-as-root -np $N python hvd_fp16_test.py
