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
  mobilenet_v1
  mobilenet_v2
  nasnet_mobile
  resnet_v1_50
  resnet_v2_50
  vgg_16
  vgg_19
  inception_v3
  inception_v4
)

batch_sizes=( 1 8 128 )

for i in ${models[@]};
do
  for bs in ${batch_sizes[@]};
  do
    echo "Testing $i batch_size $bs..."
    common_args="--model $i --batch_size $bs --use_synthetic --num_iterations 2000 --download_dir /data/tensorflow/models"
    python -u inference.py $common_args           --precision fp32                        2>&1 | tee $OUTPUT_PATH/output_tf_bs${bs}_fp32_$i
    python -u inference.py $common_args --use_trt --precision fp32                        2>&1 | tee $OUTPUT_PATH/output_tftrt_${bs}_fp32_$i
    python -u inference.py $common_args --use_trt --precision fp16                        2>&1 | tee $OUTPUT_PATH/output_tftrt_${bs}_fp16_$i
    python -u inference.py $common_args --use_trt --precision int8 --num_calib_inputs 128 2>&1 | tee $OUTPUT_PATH/output_tftrt_${bs}_int8_$i
    echo "DONE testing $i batch_size $bs"
  done
done


models=(
  nasnet_large
)
batch_sizes=( 1 8 32 )

for i in ${models[@]};
do
  for bs in ${batch_sizes[@]};
  do
    echo "Testing $i batch_size $bs..."
    common_args="--model $i --batch_size $bs --use_synthetic --num_iterations 2000 --download_dir /data/tensorflow/models"
    python -u inference.py $common_args           --precision fp32                        2>&1 | tee $OUTPUT_PATH/output_tf_bs${bs}_fp32_$i
    python -u inference.py $common_args --use_trt --precision fp32                        2>&1 | tee $OUTPUT_PATH/output_tftrt_${bs}_fp32_$i
    python -u inference.py $common_args --use_trt --precision fp16                        2>&1 | tee $OUTPUT_PATH/output_tftrt_${bs}_fp16_$i
    python -u inference.py $common_args --use_trt --precision int8 --num_calib_inputs 128 2>&1 | tee $OUTPUT_PATH/output_tftrt_${bs}_int8_$i
    echo "DONE testing $i batch_size $bs"
  done
done

popd
