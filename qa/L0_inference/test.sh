#!/bin/bash

set -e

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

dynamic_op=(
True
False
)

for use_trt_dynamic_op in ${dynamic_op[@]}; do
    echo "Testing $model $use_trt_dynamic_op"
    dynamic_op_params=""
    if [ ${use_trt_dynamic_op} == True ] ; then
        dynamic_op_params=--use_trt_dynamic_op
    fi;

    OUTPUT_FILE=$OUTPUT_PATH/output_tftrt_fp16_bs8_${model}_dynamic_op=${use_trt_dynamic_op}
    python -u inference.py \
        --data_dir "/data/imagenet/train-val-tfrecord" \
        --model $model \
        --use_trt \
        --precision fp16 \
        $dynamic_op_params \
        2>&1 | tee $OUTPUT_FILE
    python -u check_accuracy.py --input_path $OUTPUT_PATH --batch_size 8 --model $model --dynamic_op $use_trt_dynamic_op --precision tftrt_fp16
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

popd
