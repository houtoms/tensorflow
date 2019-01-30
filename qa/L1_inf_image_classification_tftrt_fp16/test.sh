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
    mobilenet_v1
    mobilenet_v2
    #nasnet_large
    #nasnet_mobile
    resnet_v1_50
    resnet_v2_50
    #vgg_16
    #vgg_19
    inception_v3
    inception_v4
  )
  if ! $JETSON ; then
    models+=(nasnet_large)
    models+=(vgg_16)
    models+=(vgg_19)
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
      --default_models_dir "/data/tensorflow/models" \
      --model $model \
      --use_trt \
      --precision fp16 \
      2>&1 | tee $OUTPUT_PATH/output_tftrt_fp16_bs8_${model}_dynamic_op=False
  popd
  pushd $SCRIPTS_PATH
  python -u check_accuracy.py --input $OUTPUT_PATH --precision tftrt_fp16 --batch_size 8 --model $model ; rv=$(($rv+$?))
  python -u check_nodes.py --input_path $OUTPUT_PATH --model $model --precision tftrt_fp16 --batch_size 8 ; rv=$(($rv+$?))
  if $JETSON ; then
    python -u check_performance.py --input_path $OUTPUT_PATH --model $model --precision tftrt_fp16 --batch_size 8 ; rv=$(($rv+$?))
  fi
  set +e
  popd

  echo "DONE testing $model"
done
exit $rv
