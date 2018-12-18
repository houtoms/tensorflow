#!/bin/bash

export COCO_IMAGE_DIR=coco/val2017
export COCO_ANNOTATION_PATH=coco/annotations/instances_val2017.json
export STATIC_DATA_DIR=static_data
export DATA_DIR=data
export COCO_API_DIR=third_party/cocoapi
export TF_MODELS_DIR=third_party/models

export IMAGE_IDS_PATH=$STATIC_DATA_DIR/image_ids.json
export REFERENCE_MAP_PATH=$STATIC_DATA_DIR/reference_map.json
