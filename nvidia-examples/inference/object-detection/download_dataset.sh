#!/bin/bash

COCO_YEAR=2017
OUTPUT_DIR=coco

ANNOTATIONS_ZIP_NAME=annotations_trainval${COCO_YEAR}.zip
IMAGES_ZIP_NAME=val${COCO_YEAR}.zip

mkdir -p $OUTPUT_DIR

# download annotations
wget http://images.cocodataset.org/annotations/${ANNOTATIONS_ZIP_NAME} -O $OUTPUT_DIR/${ANNOTATIONS_ZIP_NAME}
unzip ${OUTPUT_DIR}/${ANNOTATIONS_ZIP_NAME} -d ${OUTPUT_DIR}

# download images
wget http://images.cocodataset.org/zips/${IMAGES_ZIP_NAME} -O ${OUTPUT_DIR}/${IMAGES_ZIP_NAME}
unzip ${OUTPUT_DIR}/${IMAGES_ZIP_NAME} -d ${OUTPUT_DIR}
