#!/bin/bash

export COCO_IMAGE_DIR=/data/coco/coco-2017/coco2017/val2017
export COCO_ANNOTATION_PATH=/data/coco/coco-2017/coco2017/annotations/instances_val2017.json
export STATIC_DATA_DIR=/data/tensorflow/object_detection
export DATA_DIR=$PWD/data
export COCO_API_DIR=$PWD/../third_party/cocoapi
export TF_MODELS_DIR=$PWD/../third_party/tensorflow_models

export IMAGE_IDS_PATH=$STATIC_DATA_DIR/image_ids.json
export REFERENCE_MAP_PATH=$STATIC_DATA_DIR/reference_map.json
