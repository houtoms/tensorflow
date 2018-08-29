#!/bin/bash

set -e
set -v

pip install requests
MODELS="$PWD/../third_party/tensorflow_models/"
export PYTHONPATH="$PYTHONPATH:$MODELS"
pushd $MODELS/research/slim
python setup.py install
popd

OUTPUT_PATH=$PWD
pushd ../../nvidia-examples/tftrt/scripts

models=(
  vgg_16
  vgg_19
  inception_v3 )
for i in "${models[@]}"
do
  echo "Testing $i..."
  python -u inference.py --batch_size 64 --model $i --use_trt --precision int8 2>&1 | tee $OUTPUT_PATH/output_tftrt_int8_$i
  python -u check_accuracy.py --tolerance 1.0 --input $OUTPUT_PATH/output_tftrt_int8_$i
  echo "DONE testing $i"
done
popd
