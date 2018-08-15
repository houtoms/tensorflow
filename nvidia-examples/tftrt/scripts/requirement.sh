#!/bin/bash

set -e
set -v

# Setup for TF research/slim
# https://github.com/tensorflow/models/tree/master/research/slim
ln -sf ../../../qa/third_party/tensorflow_models models
pushd models/research/slim
python setup.py install

cd ../../../
# Setup for TF models/official
# https://github.com/tensorflow/models/tree/master/official
CURRENT_PATH=$(pwd)
echo $CURRENT_PATH
export PYTHONPATH="$PYTHONPATH:$CURRENT_PATH/models"
pip install requests
popd
