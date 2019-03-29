#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

cd /opt/tensorflow/nvidia-examples/UNet_Industrial/

python /opt/tensorflow/qa/joc_qa/segmentation/unet_industrial/tests_inferbench.py \
    --nouse_tf_amp \
    --nouse_xla
