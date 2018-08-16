#!/bin/bash

set -e
set -v

pip install requests
MODELS="/opt/tensorflow/qa/third_party/tensorflow_models/"
export PYTHONPATH="$PYTHONPATH:$MODELS"
pushd $MODELS/research/slim
python setup.py install
popd

OUTPUT_PATH=$PWD
pushd ../../nvidia-examples/tftrt/scripts

models=(
  mobilenet_v1
  mobilenet_v2
  nasnet_large
  nasnet_mobile
  resnet_v1_50
  resnet_v2_50
  vgg_16 vgg_19
  inception_v3
  inception_v4 )
for i in "${models[@]}"
do
  python -u inference.py --model $i 2>&1 | tee $OUTPUT_PATH/output_$i
  python -u check_accuracy.py --input $OUTPUT_PATH/output_$i
  echo "DONE testing $i"
done
popd
