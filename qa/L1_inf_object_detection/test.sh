#!/bin/bash

set -e

MAP_ERROR_THRESHOLD=0.001

source $PWD/../inference/object_detection/setup_env.sh

pushd $PWD/../../nvidia-examples/inference/object-detection

./setup.sh

# TENSORFLOW TESTS

MODELS=(
  ssd_mobilenet_v1_coco
  ssd_mobilenet_v2_coco
  ssd_inception_v2_coco
  # ssd_resnet_50_fpn_coco # excluded because of py3 issue in tensorflow/models repository
  faster_rcnn_resnet50_coco
  mask_rcnn_resnet50_atrous_coco
)

for model in "${MODELS[@]}"
do

  MODEL_DIR=$DATA_DIR/${model}_tf

  python -u -m object_detection_benchmark.inference $model \
    --batch_size 1 \
    --force_nms_cpu \
    --coco_image_dir $COCO_IMAGE_DIR \
    --coco_annotation_path $COCO_ANNOTATION_PATH \
    --static_data_dir $STATIC_DATA_DIR \
    --model_dir $MODEL_DIR \
    --image_ids_path $IMAGE_IDS_PATH \
    --image_shape 600,600

  python -u -m object_detection_benchmark.check_accuracy $model $MODEL_DIR \
    --tolerance $MAP_ERROR_THRESHOLD

done

# TENSORRT TESTS

MODELS=(
  ssd_mobilenet_v1_coco
  ssd_mobilenet_v2_coco
  ssd_inception_v2_coco
  # ssd_resnet_50_fpn_coco # excluded because of py3 issue in tensorflow/models repository
  # faster_rcnn_resnet50_coco # excluded because of known issues
  # mask_rcnn_resnet50_atrous_coco # excluded because of known issues
)

PRECISION_MODES=(
  FP32
  FP16
)

for model in "${MODELS[@]}"
do
  for precision_mode in "${PRECISION_MODES[@]}"
  do

  MODEL_DIR=$DATA_DIR/${model}_trt_${precision_mode}

  python -u -m object_detection_benchmark.inference $model \
    --batch_size 1 \
    --use_trt \
    --precision_mode $precision_mode \
    --minimum_segment_size 50 \
    --force_nms_cpu \
    --remove_assert \
    --coco_image_dir $COCO_IMAGE_DIR \
    --coco_annotation_path $COCO_ANNOTATION_PATH \
    --static_data_dir $STATIC_DATA_DIR \
    --model_dir $MODEL_DIR \
    --image_ids_path $IMAGE_IDS_PATH \
    --image_shape 600,600

  python -u -m object_detection_benchmark.check_accuracy $model $MODEL_DIR \
    --tolerance $MAP_ERROR_THRESHOLD

  done
done

popd
