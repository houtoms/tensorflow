#!/bin/bash


cd tf_trt_models

git clone https://github.com/tensorflow/models.git

pushd models/research
sed '87s/^/\/\//' object_detection/protos/ssd.proto
protoc object_detection/protos/*.proto --python_out=.
sudo python setup.py install
pushd slim
sudo python setup.py install
popd
popd

set -e


python tf_trt_inference_test.py --model inception_v1 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v2 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v3 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v4 --use_trt 1
rm -r data



python tf_trt_inference_test.py --model resnet_v2_50 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v2_101 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v2_152 --num_classes 1001 --use_trt 1 
rm -r data




python tf_trt_inference_test.py --model resnet_v1_50 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_101 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_152 --num_classes 1001 --use_trt 1 
rm -r data




python tf_inference_test.py --model mobilenet_v1_0.25_128 --num_classes 1001 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model mobilenet_v1_0p5_160 --num_classes 1001 --use_trt 1
rm -r data



python tf_trt_inference_test.py --model ssd_mobilenet_v1_coco --num_classes 1001 --use_trt 1 --detection 1
rm -r data
