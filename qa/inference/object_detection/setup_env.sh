#!/bin/bash

export COCO_DIR=/data/coco/coco-2017/coco2017
export COCO_YEAR=2017
export STATIC_DATA_DIR=/data/tensorflow/object_detection
export DATA_DIR=$PWD/data
export COCO_API_DIR=$PWD/../third_party/cocoapi
export TF_MODELS_DIR=$PWD/../third_party/tensorflow_models

export IMAGE_IDS_PATH=$STATIC_DATA_DIR/image_ids.json
export REFERENCE_MAP_PATH=$STATIC_DATA_DIR/reference_map.json
