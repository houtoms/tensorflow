#!/bin/bash

apt-get update
apt-get install -y openmpi-bin libopenmpi-dev

export HOROVOD_NCCL_INCLUDE=/usr/include
export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu
export HOROVOD_GPU_ALLREDUCE=NCCL

pip2 install --no-cache-dir horovod
pip3 install --no-cache-dir horovod
