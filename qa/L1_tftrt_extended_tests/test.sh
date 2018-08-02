#!/bin/bash

NATIVE_ARCH=`uname -m`
if [ ${NATIVE_ARCH} == 'x86_64' ]; then
  # if running on x86 rather than aarch64 then presumably we're in a container
  # need to install various deps here

  # Libraries
  apt-get update && apt-get install -y libfreetype6-dev libpng12-dev libjpeg8-dev

  # Protobuf compiler
  pushd /
  PROTOBUF_VERSION=3.4.0 && \
    curl -L https://github.com/google/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
    cd /protobuf-${PROTOBUF_VERSION} && \
    ./autogen.sh && \
    ./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared 2>&1 > /dev/null && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install 2>&1 > /dev/null && \
    rm -rf /protobuf-${PROTOBUF_VERSION}
  popd
fi


cd tf_trt_models
ln -sf ../../third_party/tensorflow_models models
pushd models/research
sed -i '87s/^/\/\//' object_detection/protos/ssd.proto
protoc object_detection/protos/*.proto --python_out=.
python setup.py install
pushd slim
python setup.py install
popd
popd


set -e



###################### TF_TRT INFERENCE TESTS #####################



python tf_trt_inference_test.py --model resnet_v1_50 --num_classes 1000 --use_trt 1 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_101 --num_classes 1000 --use_trt 1 
rm -r data



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



python tf_trt_inference_test.py --model vgg_16 --num_classes 1000 --use_trt 1
rm -r data

python tf_trt_inference_test.py --model vgg_19 --num_classes 1000 --use_trt 1
rm -r data



#################### TENSORFLOW INFERENCE TESTS ########################



python tf_trt_inference_test.py --model resnet_v1_50 --num_classes 1000 --use_trt 0 
rm -r data

python tf_trt_inference_test.py --model resnet_v1_101 --num_classes 1000 --use_trt 0 
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



python tf_trt_inference_test.py --model vgg_16 --num_classes 1000 --use_trt 0
rm -r data

python tf_trt_inference_test.py --model vgg_19 --num_classes 1000 --use_trt 0
rm -r data
