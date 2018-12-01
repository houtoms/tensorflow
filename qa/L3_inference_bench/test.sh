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


JETSON=false
NATIVE_ARCH=`uname -m`
if [ ${NATIVE_ARCH} == 'aarch64' ]; then
  JETSON=true
fi


set_models() {
  models=(
    mobilenet_v1
    mobilenet_v2
    nasnet_mobile
    resnet_v1_50
    resnet_v2_50
    resnet_v2_152
    #vgg_16
    #vgg_19
    inception_v3
    inception_v4
  )
  if [ ${JETSON} == false ]; then
    models+=(vgg_16)
    models+=(vgg_19)
  fi
}

set_batch_sizes() {
  if $JETSON ; then
    batch_sizes=(1 8 64)
  else
    batch_sizes=(1 8 128)
  fi
}

set_allocator() {
  if $JETSON ; then
    export TF_GPU_ALLOCATOR="cuda_malloc"
  else
    unset TF_GPU_ALLOCATOR
  fi
}

run_inference() {
  models=$1
  batch_sizes=$2
  for i in ${models[@]};
  do
    for bs in ${batch_sizes[@]};
    do
      echo "Testing $i batch_size $bs..."
      common_args="
        --model $i
        --default_models_dir /data/tensorflow/models
        --data_dir /data/imagenet/train-val-tfrecord
        --calib_data_dir /data/imagenet/train-val-tfrecord
        --batch_size $bs
        --num_iterations 2000"
      unset TF_GPU_ALLOCATOR
            
      python -u inference.py $common_args           --precision fp32                        2>&1 | tee $OUTPUT_PATH/output_tf_bs${bs}_fp32_$i

      set_allocator

      python -u inference.py $common_args --use_trt --precision fp32                        2>&1 | tee $OUTPUT_PATH/output_tftrt_fp32_bs${bs}_$i

      python -u inference.py $common_args --use_trt --precision fp16                        2>&1 | tee $OUTPUT_PATH/output_tftrt_fp16_bs${bs}_$i
      
      python -u inference.py $common_args --use_trt --precision int8 --num_calib_inputs 128 2>&1 | tee $OUTPUT_PATH/output_tftrt_int8_bs${bs}_$i
      
      echo "DONE testing $i batch_size $bs"
    done
  done
}


set_models
set_batch_sizes

run_inference $models $batch_sizes

models=(
  nasnet_large
)
batch_sizes=( 1 8 32 )

run_inference $models $batch_sizes

