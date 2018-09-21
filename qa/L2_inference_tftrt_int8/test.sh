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



set_models() {
  NATIVE_ARCH=`uname -m`
  models=(
    #mobilenet_v1
    #mobilenet_v2
    #nasnet_large
    #nasnet_mobile
    #resnet_v1_50
    #resnet_v2_50
    #vgg_16
    #vgg_19
    inception_v3
    #inception_v4
  )
  if [${NATIVE_ARCH} == 'x86_64']; then
    models+=(vgg_16)
    models+=(vgg_19)
  fi
}


set_allocator() {
  NATIVE_ARCH=`uname -m`
  if [${NATIVE_ARCH} == 'aarch64']; then
    export TF_GPU_ALLOCATOR="cuda_malloc"
  else
    unset TF_GPU_ALLOCATOR
  fi
}


set_models
set_allocator

for i in "${models[@]}"
do
  echo "Testing $i..."
  python -u inference.py --batch_size 64 --model $i --use_trt --precision int8 --download_dir /data/tensorflow/models 2>&1 | tee $OUTPUT_PATH/output_tftrt_int8_$i
  python -u check_accuracy.py --tolerance 1.0 --input $OUTPUT_PATH/output_tftrt_int8_$i
  echo "DONE testing $i"
done
popd
