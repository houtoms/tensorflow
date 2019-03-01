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

echo Detect arch...
lscpu | grep -q ^Architecture:.*aarch64
is_aarch64=$[!$?]
lscpu | grep -q ^CPU\(s\):.*8
is_8cpu=$[!$?]
lscpu | grep -q ^CPU\(s\):.*4
is_4cpu=$[!$?]
is_nano=$[$is_aarch64 && $is_4cpu]
is_xavier=$[$is_aarch64 && $is_8cpu]

echo Find all test cases...
if [[ "$is_xavier" == 1 ]]
then
  TEST_CASES=(
    faster_rcnn_resnet50_coco.json
    ssd_inception_v2_coco_trt_fp16.json
    ssdlite_mobilenet_v2_coco_trt_fp16.json
    ssd_mobilenet_v1_coco_trt_fp16.json
    ssd_mobilenet_v2_coco_trt_fp16.json
    mask_rcnn_resnet50_atrous_coco.json
    ssd_inception_v2_coco_trt_fp32.json
    ssdlite_mobilenet_v2_coco_trt_fp32.json
    ssd_mobilenet_v1_coco_trt_fp32.json
    ssd_mobilenet_v2_coco_trt_fp32.json
    ssd_inception_v2_coco_tf.json
    ssdlite_mobilenet_v2_coco_tf.json
    ssd_mobilenet_v1_coco_tf.json
    ssd_mobilenet_v2_coco_tf.json
    )
  for (( i=0; i<${#array[@]}; i++ ));
  do
    TEST_CASES[$i]="$SCRIPTS_PATH/tests/xavier_acc_perf/${TEST_CASES[$i]}"
  done
elif [[ "$is_nano" == 1]]
then
  TEST_CASES=(
    ssd_mobilenet_v2_coco_trt_fp16.json
    ssd_mobilenet_v2_coco_tf.json
    )
  for (( i=0; i<${#array[@]}; i++ ));
  do
    TEST_CASES[$i]="$SCRIPTS_PATH/tests/generic_acc/${TEST_CASES[$i]}"
  done
else
  TEST_CASES=(
    #faster_rcnn_nas_tf.json
    #faster_rcnn_nas_trt_fp16.json
    #faster_rcnn_nas_trt_fp32.json
    faster_rcnn_resnet50_coco_tf.json
    #faster_rcnn_resnet50_coco_trt_fp16.json
    #faster_rcnn_resnet50_coco_trt_fp32.json
    mask_rcnn_resnet50_atrous_coco_tf.json
    #mask_rcnn_resnet50_atrous_coco_trt_fp16.json
    #mask_rcnn_resnet50_atrous_coco_trt_fp32.json
    ssd_inception_v2_coco_tf.json
    ssd_inception_v2_coco_trt_fp16.json
    ssd_inception_v2_coco_trt_fp32.json
    ssd_mobilenet_v1_coco_tf.json
    ssd_mobilenet_v1_coco_trt_fp16.json
    ssd_mobilenet_v1_coco_trt_fp32.json
    ssd_mobilenet_v2_coco_tf.json
    ssd_mobilenet_v2_coco_trt_fp16.jso
    ssd_mobilenet_v2_coco_trt_fp32.json
    ssdlite_mobilenet_v2_coco_tf.json
    ssdlite_mobilenet_v2_coco_trt_fp16.json
    ssdlite_mobilenet_v2_coco_trt_fp32.json
    )
  for (( i=0; i<${#array[@]}; i++ ));
  do
    TEST_CASES[$i]="$SCRIPTS_PATH/tests/generic_acc/${TEST_CASES[$i]}"
  done
fi

echo Run all tests...
rv=0
for test_case in "${TEST_CASES[@]}"
do
  echo "Testing $test_case..."
  python -m tftrt.examples.object_detection.test ${test_case} ; rv=$(($rv+$?))
  echo "DONE testing $test_case"
done
exit $rv
