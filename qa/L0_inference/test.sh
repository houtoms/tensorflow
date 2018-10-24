#!/bin/bash

set -e
set -v

# IMAGE CLASSIFICATION

pip install requests
MODELS="$PWD/../third_party/tensorflow_models/"
export PYTHONPATH="$PYTHONPATH:$MODELS"
pushd $MODELS/research/slim
python setup.py install
popd

OUTPUT_PATH=$PWD
pushd ../../nvidia-examples/inference/image-classification/scripts

set_allocator() {
  NATIVE_ARCH=`uname -m`
  if [ ${NATIVE_ARCH} == 'aarch64' ]; then
    export TF_GPU_ALLOCATOR="cuda_malloc"
  else
    unset TF_GPU_ALLOCATOR
  fi
}

set_allocator

model="mobilenet_v1"
for use_trt_dynamic_op in "" "--use_trt_dynamic_op"; do
    echo "Testing $model $use_trt_dynamic_op"
    OUTPUT_FILE=$OUTPUT_PATH/output_tftrt_fp16_${model}${use_trt_dynamic_op}
    python -u inference.py \
        --data_dir "/data/imagenet/train-val-tfrecord" \
        --model $model \
        --use_trt \
        --precision fp16 \
        $use_trt_dynamic_op \
        2>&1 | tee $OUTPUT_FILE
    python -u check_accuracy.py --input $OUTPUT_FILE
    echo "DONE testing $model $use_trt_dynamic_op"
done
popd

# OBJECT DETECTION

MAP_ERROR_THRESHOLD=0.001

source $PWD/../inference/object_detection/setup_env.sh

pushd $PWD/../../nvidia-examples/inference/object-detection

./setup.sh

model=ssd_mobilenet_v1_coco
precision_mode=FP16

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
  --reference_map_path $REFERENCE_MAP_PATH \
  --map_error_threshold $MAP_ERROR_THRESHOLD

popd
