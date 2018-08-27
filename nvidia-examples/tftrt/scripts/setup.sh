#!/bin/bash
# Get path to tensorflow/models repository (https://github.com/tensorflow/models)
MODELS_PATH_DEFAULT="$PWD/../../../qa/third_party/tensorflow_models/"
read -p "Please enter path to tensorflow/models [$MODELS_PATH_DEFAULT]: " MODELS
MODELS="${MODELS:-$MODELS_PATH_DEFAULT}"

pip install requests
export PYTHONPATH="$PYTHONPATH:$MODELS"
echo $PYTHONPATH
pushd $MODELS/research/slim
python setup.py install
popd