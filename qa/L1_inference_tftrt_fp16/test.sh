#!/bin/bash

set -e
set -v

###################### TF_TRT FP16 INFERENCE TESTS #####################

OUTPUT_PATH=$PATH
pushd ../../nvidia-examples/tftrt/scripts
bash requirement.sh
export PYTHONPATH=$PYTHONPATH:$PWD/models

models=( mobilenet_v1 mobilenet_v2 nasnet_large nasnet_mobile resnet_v1_50 resnet_v2_50 vgg_16 vgg_19 inception_v3 inception_v4 )

for i in "${models[@]}"
do
  python -u inference.py --model $i --use_trt --precision fp16  2>&1 | tee $OUTPUT_PATH/output_tftrt_fp16_$i
  python check_accuracy.py --tolerance 0.01 --model $i --input $OUTPUT_PATH/output_tftrt_fp16_$i
done
popd
