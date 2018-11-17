#!/bin/bash

set -e

pip install requests
MODELS="$PWD/../third_party/tensorflow_models/"
export PYTHONPATH="$PYTHONPATH:$MODELS"
pushd $MODELS/research/slim
python setup.py install
popd

OUTPUT_PATH=$PWD
pushd ../../nvidia-examples/inference/image-classification/scripts



set_models() {
  NATIVE_ARCH=`uname -m`
  models=(
    #mobilenet_v1
    mobilenet_v2
    nasnet_large
    nasnet_mobile
    resnet_v1_50
    resnet_v2_50
    #vgg_16
    #vgg_19
    inception_v3
    #inception_v4
  )
  if [ ${NATIVE_ARCH} == 'x86_64' ]; then
    models+=(vgg_16)
    models+=(vgg_19)
  fi
}


set_allocator() {
  NATIVE_ARCH=`uname -m`
  if [ ${NATIVE_ARCH} == 'aarch64' ]; then
    export TF_GPU_ALLOCATOR="cuda_malloc"
  else
    unset TF_GPU_ALLOCATOR
  fi
}


# This function should be used by all of the inference tests and updated to
# access a hashtable based on {model,precision,available_memory/machine_name}.
# We can put it in a central location such as /opt/tensorflow/qa/inference.
set_batch_size() {
  BATCH_SIZE=32
  AVAIL_MiB=$(nvidia-smi -q -d MEMORY \
            | grep -A3 "FB Memory Usage" \
            | grep Total \
            | awk 'BEGIN {F=1; M=-1}
                   {
                       if ($4 != "MiB") {M=-1}
                       if (F==1 || $3<M) {M=$3; F=0}
                   }
                   END {print M}')
  if [[ "$AVAIL_MiB" == "-1" ]]; then
    echo "Failed to detect GPU memory sizes"
    exit 1
  fi
  if [[ "$AVAIL_MiB" -ge "12000" ]]; then
    BATCH_SIZE=64
  fi
}


set_models
set_allocator
set_batch_size

for model in "${models[@]}"
do
  echo "Testing $model..."
  python -u inference.py \
      --data_dir "/data/imagenet/train-val-tfrecord" \
      --calib_data_dir "/data/imagenet/train-val-tfrecord" \
      --download_dir "/data/tensorflow/models" \
      --model $model \
      --use_trt \
      --batch_size $BATCH_SIZE \
      --precision int8 \
      2>&1 | tee $OUTPUT_PATH/output_tftrt_int8_$model
  python -u check_accuracy.py --tolerance 1.0 --input $OUTPUT_PATH/output_tftrt_int8_$model
  echo "DONE testing $model"
done
popd
