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
    nasnet_mobile
    nasnet_large
    resnet_v1_50
    resnet_v2_50
    resnet_v2_152
    #vgg_16
    #vgg_19
    inception_v3
    inception_v4
  )
  if ! $JETSON ; then
    models+=(vgg_16)
    models+=(vgg_19)
  fi
}

run_inference() {
  models=$1
  for i in ${models[@]};
  do
    echo "Testing $i batch_size 8..."
    common_args="
      --model $i
      --default_models_dir /data/tensorflow/models
      --data_dir /data/imagenet/train-val-tfrecord
      --calib_data_dir /data/imagenet/train-val-tfrecord
      --batch_size 8"
    unset TF_GPU_ALLOCATOR
    pushd $EXAMPLE_PATH
    python -u image_classification.py $common_args           --precision fp32                        2>&1 | tee $OUTPUT_PATH/output_tf_fp32_bs${bs}_$i_dynamic_op=False
    popd
    pushd $SCRIPTS_PATH
    python -u check_accuracy.py --input_path $OUTPUT_PATH --precision tf_fp32 --batch_size 8 --model $i
    popd
    echo "DONE testing $i batch_size $bs"
  done
}


set_models

pushd $EXAMPLE_PATH
sed -z -i -e "s/tf_config = tf.ConfigProto()/tf_config = tf.ConfigProto()\n    tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1/g" image_classification.py
popd

run_inference $models


pushd $EXAMPLE_PATH
sed -z -i -e "s/tf_config = tf.ConfigProto()\n    tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1/tf_config = tf.ConfigProto()/g" image_classification.py
popd
