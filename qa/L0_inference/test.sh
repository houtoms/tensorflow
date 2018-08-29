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

model="mobilenet_v1"
echo "Testing $model..."
python -u inference.py --model $model --use_trt --precision fp16  2>&1 | tee $OUTPUT_PATH/output_tftrt_fp16_$model
python -u check_accuracy.py --input $OUTPUT_PATH/output_tftrt_fp16_$model
echo "DONE testing $model"
popd
