#!/bin/bash

set -v
set -e

# TENSORFLOW TESTS

MODELS=(
  ssd_mobilenet_v1_coco
  ssd_mobilenet_v2_coco
  ssd_inception_v2_coco
  faster_rcnn_resnet50_coco
  mask_rcnn_resnet50_atrous_coco
)

for model in "${MODELS[@]}"
do
  python -u -m object_detection_benchmark.test $model \
    --force_nms_cpu \
    --coco_dir $COCO_DIR \
    --coco_year $COCO_YEAR \
    --static_data_dir $STATIC_DATA_DIR \
    --data_dir $DATA_DIR \
    --image_ids_path $IMAGE_IDS_PATH \
    --reference_map_path $REFERENCE_MAP_PATH
done

# TENSORRT TESTS

MODELS=(
  ssd_mobilenet_v1_coco
  ssd_mobilenet_v2_coco
  ssd_inception_v2_coco
)

PRECISION_MODES=(
  FP32
  FP16
)

for model in "${MODELS[@]}"
do
  for precision_mode in "${PRECISION_MODES[@]}"
  do
  python -u -m object_detection_benchmark.test $model \
    --use_trt \
    --precision_mode $precision_mode \
    --minimum_segment_size 50 \
    --force_nms_cpu \
    --remove_assert \
    --coco_dir $COCO_DIR \
    --coco_year $COCO_YEAR \
    --static_data_dir $STATIC_DATA_DIR \
    --data_dir $DATA_DIR \
    --image_ids_path $IMAGE_IDS_PATH \
    --reference_map_path $REFERENCE_MAP_PATH
  done
done
