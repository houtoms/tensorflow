#!/bin/bash

set -e
set -v

#################### TENSORFLOW INFERENCE TESTS ########################
OUTPUT_PATH=$PWD
pushd ../../nvidia-examples/tftrt_inference
bash requirement.sh
export PYTHONPATH=$PYTHONPATH:$PWD/models

models=( mobilenet_v1 mobilenet_v2 nasnet_large nasnet_mobile resnet_v1_50 resnet_v2_50 vgg_16 vgg_19 inception_v3 inception_v4 )
for i in "${models[@]}"
do
  python -u inference.py --model $i 2>&1 | tee $OUTPUT_PATH/output_$i
  python check_accuracy.py --input $OUTPUT_PATH/output_$i
done
popd
