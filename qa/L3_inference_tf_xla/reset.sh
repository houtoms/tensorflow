#!/bin/bash

EXAMPLE_PATH="$PWD/../../nvidia-examples/tensorrt/tftrt/examples/image-classification/"
pushd $EXAMPLE_PATH
sed -z -i -e "s/tf_config = tf.ConfigProto()\n    tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1/tf_config = tf.ConfigProto()/g" image_classification.py
popd
