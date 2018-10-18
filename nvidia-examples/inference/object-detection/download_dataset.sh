#!/bin/bash

mkdir -p $COCO_DIR

ANNOTATIONS_ZIP_NAME=annotations_trainval${COCO_YEAR}.zip

# download annotations
wget http://images.cocodataset.org/annotations/${ANNOTATIONS_ZIP_NAME} -O $COCO_DIR/${ANNOTATIONS_ZIP_NAME}
unzip $COCO_DIR/${ANNOTATIONS_ZIP_NAME} -d $COCO_DIR

# download images
wget http://images.cocodataset.org/zips/${IMAGES_ZIP_NAME} -O $COCO_DIR/${IMAGES_ZIP_NAME}
unzip $COCO_DIR/${IMAGES_ZIP_NAME} -d $COCO_DIR
