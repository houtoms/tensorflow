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
for use_trt_dynamic_op in "" "--use_trt_dynamic_op"; do
    echo "Testing $model $use_trt_dynamic_op"
    OUTPUT_FILE=$OUTPUT_PATH/output_tftrt_fp16_${model}${use_trt_dynamic_op}
    python -u inference.py --model $model --use_trt $use_trt_dynamic_op --precision fp16 2>&1 | tee $OUTPUT_FILE
    python -u check_accuracy.py --input $OUTPUT_FILE
    echo "DONE testing $model $use_trt_dynamic_op"
done
popd
