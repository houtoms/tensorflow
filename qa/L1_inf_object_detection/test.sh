#!/bin/bash

set +e

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

echo Detecting platform...
is_aarch64=$(lscpu | grep ^Architecture:.*aarch64)
is_8cpu=$(lscpu | grep ^CPU\(s\):[^0-9]*8$)
is_6cpu=$(lscpu | grep ^CPU\(s\):[^0-9]*6$)
is_4cpu=$(lscpu | grep ^CPU\(s\):[^0-9]*4$)


if [[ $is_aarch64 && $is_8cpu ]]; then
  is_xavier=1
elif [[ $is_aarch64 && $is_6cpu ]]; then
  is_tx2=1
elif [[ $is_aarch64 && $is_4cpu ]]; then
  is_nano=1
fi

echo "Setting test_path..."
test_path="$SCRIPTS_PATH/tests/generic_acc/${test_case}"
# There is a different test case dir just for xavier.
# The only difference between this dir and generic is the performance
# data that is specific to xavier.
if [[ "$is_xavier" == 1 ]]; then
  test_path="$SCRIPTS_PATH/tests/xavier_acc_perf/${test_case}"
elif [[ "$is_nano" == 1 ]]; then
  test_path="$SCRIPTS_PATH/tests/nano_acc/${test_case}"
fi

set_test_cases() {
  # Name of test cases for xavier and other GPUs are the same, but
  # they are stored in different directories.
  if [[ $is_aarch64 ]]; then
    test_cases=(
      ssd_mobilenet_v1_coco_trt_fp16.json
      ssd_mobilenet_v2_coco_trt_fp16.json
      ssdlite_mobilenet_v2_coco_trt_fp16.json
    )
  else
    test_cases=(
      ssd_inception_v2_coco_tf.json
      ssd_inception_v2_coco_trt_fp16.json
      ssd_inception_v2_coco_trt_fp32.json
      ssd_mobilenet_v1_coco_tf.json
      ssd_mobilenet_v1_coco_trt_fp16.json
      ssd_mobilenet_v1_coco_trt_fp32.json
      ssd_mobilenet_v2_coco_tf.json
      ssd_mobilenet_v2_coco_trt_fp16.json
      ssd_mobilenet_v2_coco_trt_fp32.json
      ssdlite_mobilenet_v2_coco_tf.json
      ssdlite_mobilenet_v2_coco_trt_fp16.json
      ssdlite_mobilenet_v2_coco_trt_fp32.json
      faster_rcnn_resnet50_coco_tf.json
      mask_rcnn_resnet50_atrous_coco_tf.json
      #faster_rcnn_resnet50_coco_trt_fp16.json
      #faster_rcnn_resnet50_coco_trt_fp32.json
      #faster_rcnn_nas_tf.json
      #faster_rcnn_nas_trt_fp16.json
      #faster_rcnn_nas_trt_fp32.json
      #mask_rcnn_resnet50_atrous_coco_trt_fp16.json
      #mask_rcnn_resnet50_atrous_coco_trt_fp32.json
   )
 fi
}

set_test_cases

echo Run all tests...
rv=0
for test_case in "${test_cases[@]}"
do
  echo "Testing $test_case..."
  python -m tftrt.examples.object_detection.test "${test_path}/${test_case}" ; rv=$(($rv+$?))
  echo "DONE testing $test_case"
done
exit $rv
