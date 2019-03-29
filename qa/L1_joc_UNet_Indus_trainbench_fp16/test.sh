#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

cd /opt/tensorflow/nvidia-examples/UNet_Industrial/

python /opt/tensorflow/qa/joc_qa/segmentation/unet_industrial/tests_trainbench.py \
    --n_gpus=1 \
    --use_tf_amp \
    --nouse_xla

python /opt/tensorflow/qa/joc_qa/segmentation/unet_industrial/tests_trainbench.py \
    --n_gpus=4 \
    --use_tf_amp \
    --nouse_xla

python /opt/tensorflow/qa/joc_qa/segmentation/unet_industrial/tests_trainbench.py \
    --n_gpus=8 \
    --use_tf_amp \
    --nouse_xla
