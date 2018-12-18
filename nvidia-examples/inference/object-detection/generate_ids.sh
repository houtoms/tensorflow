#!/bin/bash

mkdir -p $STATIC_DATA_DIR

python -u -m object_detection_benchmark.generate_image_ids $COCO_ANNOTATION_PATH --num_images 3000 \
  --image_ids_path $STATIC_DATA_DIR/image_ids.json
