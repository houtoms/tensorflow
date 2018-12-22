#!/bin/bash

set +e

# IMAGE CLASSIFICATION

pip install requests
MODELS="$PWD/../third_party/tensorflow_models/"
export PYTHONPATH="$PYTHONPATH:$MODELS"
pushd $MODELS/research/slim
python setup.py install
popd

OUTPUT_PATH=$PWD
EXAMPLE_PATH="../../nvidia-examples/tensorrt/tftrt/examples/image-classification/"
SCRIPTS_PATH="../inference/image_classification/"

set_allocator() {
  NATIVE_ARCH=`uname -m`
  if [ ${NATIVE_ARCH} == 'aarch64' ]; then
    export TF_GPU_ALLOCATOR="cuda_malloc"
  else
    unset TF_GPU_ALLOCATOR
  fi
}

set_allocator

model="mobilenet_v1"

dynamic_op=(
True
False
)

for use_trt_dynamic_op in ${dynamic_op[@]}; do
    echo "Testing $model $use_trt_dynamic_op"
    dynamic_op_params=""
    if [ ${use_trt_dynamic_op} == True ] ; then
        dynamic_op_params=--use_trt_dynamic_op
    fi;

    OUTPUT_FILE=$OUTPUT_PATH/output_tftrt_fp16_bs8_${model}_dynamic_op=${use_trt_dynamic_op}
    pushd $EXAMPLE_PATH
    python -u image_classification.py \
        --data_dir "/data/imagenet/train-val-tfrecord" \
        --model $model \
        --use_trt \
        --precision fp16 \
        $dynamic_op_params \
        2>&1 | tee $OUTPUT_FILE
    popd
    pushd $SCRIPTS_PATH
    python -u check_accuracy.py --input_path $OUTPUT_PATH --batch_size 8 --model $model --dynamic_op $use_trt_dynamic_op --precision tftrt_fp16
    python -u check_nodes.py --input_path $OUTPUT_PATH --batch_size 8 --model $model --dynamic_op $use_trt_dynamic_op --precision tftrt_fp16
    popd
    echo "DONE testing $model $use_trt_dynamic_op"
done

# OBJECT DETECTION

EXAMPLE_PATH="../../nvidia-examples/tensorrt/tftrt/examples/object_detection/"
SCRIPTS_PATH="../inference/object_detection/"

echo Install dependencies of object_detection...
pushd $EXAMPLE_PATH
./install_dependencies.sh
popd

echo Setup tensorflow/tensorrt...
pushd $PWD/../../nvidia-examples/tensorrt
python setup.py install
popd

test_case="$SCRIPTS_PATH/tests/generic_acc/ssd_mobilenet_v1_coco_trt_fp16.json"
echo "Testing $test_case..."
python -m tftrt.examples.object_detection.test ${test_case}
echo "DONE testing $test_case"
failure=$?
exit $failure
