#!/bin/bash


#apt-get install libfreetype6-dev
#apt-get install pkg-config
#apt-get install libpng12-dev

cd tf_trt_models

git clone https://github.com/tensorflow/models.git

pushd models/research
sed -i '87s/^/\/\//' object_detection/protos/ssd.proto
protoc object_detection/protos/*.proto --python_out=.
sudo python setup.py install
pushd slim
sudo python setup.py install
popd
popd


set -e



python tf_trt_inference_test.py --model mobilenet_v1_0p25_128 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model mobilenet_v1_0p5_160 --num_classes 1001 --use_trt 1
rm -r data




python tf_trt_inference_test.py --model inception_v1 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v2 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v3 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model inception_v4 --use_trt 1
rm -r data




python tf_trt_inference_test.py --model resnet_v1_50 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_101 --num_classes 1001 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_152 --num_classes 1001 --use_trt 1 
rm -r data



python tf_trt_inference_test.py --model mobilenet_v1_0p25_128 --num_classes 1001 --use_trt 0 
rm -r data

python tf_trt_inference_test.py --model mobilenet_v1_0p5_160 --num_classes 1001 --use_trt 0
rm -r data




python tf_trt_inference_test.py --model inception_v1 --use_trt 0
rm -r data

python tf_trt_inference_test.py --model inception_v2 --use_trt 0
rm -r data

python tf_trt_inference_test.py --model inception_v3 --use_trt 0
rm -r data

python tf_trt_inference_test.py --model inception_v4 --use_trt 0
rm -r data




python tf_trt_inference_test.py --model resnet_v1_50 --num_classes 1001 --use_trt 0 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_101 --num_classes 1001 --use_trt 0 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_152 --num_classes 1001 --use_trt 0 
rm -r data

