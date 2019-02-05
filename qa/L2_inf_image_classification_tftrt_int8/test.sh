#!/bin/bash

set +e

echo Setup tensorflow/tensorrt...
TRT_PATH="$PWD/../../nvidia-examples/tensorrt/"
pushd $TRT_PATH
python setup.py install
popd

OUTPUT_PATH=$PWD
EXAMPLE_PATH="$TRT_PATH/tftrt/examples/image-classification/"
TF_MODELS_PATH="$TRT_PATH/tftrt/examples/third_party/models/"
SCRIPTS_PATH="$PWD/../inference/image_classification/"

export PYTHONPATH="$PYTHONPATH:$TF_MODELS_PATH"

echo Install dependencies of image_classification...
pushd $EXAMPLE_PATH
./install_dependencies.sh
popd

JETSON=false
NATIVE_ARCH=`uname -m`
if [ ${NATIVE_ARCH} == "aarch64" ]; then
  JETSON=true
fi

set_models() {
  models=(
    #mobilenet_v1 disabled due to low accuracy: http://nvbugs/2369608
    mobilenet_v2
    #nasnet_large disabled due to calibration taking too long
    #nasnet_mobile disabled only on Jetson due to memory issues
    resnet_v1_50
    #resnet_v2_50 disabled due to calibration taking too long
    #vgg_16 disabled only on Jetson due to low perf
    #vgg_19 disabled due to calibration taking too long
      #(Jetson has additional perf problems for VGG)
    #inception_v3 disabled due to calibration taking too long
    inception_v4
  )
  if ! $JETSON ; then
    models+=(nasnet_mobile)
    models+=(vgg_16)
  fi
}


set_allocator() {
  if $JETSON ; then
    export TF_GPU_ALLOCATOR="cuda_malloc"
  else
    unset TF_GPU_ALLOCATOR
  fi
}

set_allocator
set_models

rv=0
for model in "${models[@]}"
do
  echo "Testing $model..."
  pushd $EXAMPLE_PATH
  python -u image_classification.py \
      --data_dir "/data/imagenet/train-val-tfrecord" \
      --calib_data_dir "/data/imagenet/train-val-tfrecord" \
      --default_models_dir "/data/tensorflow/models" \
      --model $model \
      --use_trt \
      --batch_size 8 \
      --num_calib_inputs 8 \
      --precision int8 \
      --num_calib_input 8 \
      2>&1 | tee $OUTPUT_PATH/output_tftrt_int8_bs8_${model}_dynamic_op=False
  popd
  pushd $SCRIPTS_PATH
  python -u check_accuracy.py --tolerance 1.0 --input_path $OUTPUT_PATH --precision tftrt_int8 --batch_size 8 --model $model ; rv=$(($rv+$?))
  python -u check_nodes.py --input_path $OUTPUT_PATH --precision tftrt_int8 --batch_size 8 --model $model ; rv=$(($rv+$?))
  if $JETSON ; then
    python -u check_performance.py --input_path $OUTPUT_PATH --model $model --batch_size 8 --precision tftrt_int8 ; rv=$(($rv+$?))
  fi
  popd

  echo "DONE testing $model"
done
exit $rv
