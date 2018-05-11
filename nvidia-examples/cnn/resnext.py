#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
from builtins import range
import nvutils
import tensorflow as tf
import argparse

nvutils.init()

default_args = {
    'image_width' : 224,
    'image_height' : 224,
    'image_format' : 'channels_first',
    'batch_size' : 256,
    'data_dir' : nvutils.RequireInCmdline,
    'log_dir' : None,
    'precision' : 'fp16',
    'momentum' : 0.9,
    'learning_rate_init' : 2.0,
    'learning_rate_power' : 2.0,
    'weight_decay' : 1e-4,
    'loss_scale' : 128.0,
    'larc_eta' : 0.003,
    'larc_mode' : 'clip',
    'num_iter' : 90,
    'iter_unit' : 'epoch',
    'checkpoint_secs' : None,
    'display_every' : 10,
    'deterministic' : False,
}

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('--layers', default=50, type=int, required=True,
                    choices=[50, 101, 152],
                    help="""Number of resnext layers.""")

args, flags = nvutils.parse_cmdline(default_args, parser)


def resnext_split_branch(builder, inputs, stride):
    x = inputs
    with tf.name_scope('resnext_split_branch'):
        x = builder.conv2d(x, builder.bottleneck_width, 1, stride, 'SAME')
        x = builder.conv2d(x, builder.bottleneck_width, 3, 1, 'SAME')
    return x

def resnext_shortcut(builder, inputs, stride, input_size, output_size):
    x = inputs
    useConv = builder.shortcut_type == 'C' or (builder.shortcut_type == 'B' and input_size != output_size)
    with tf.name_scope('resnext_shortcut'):
        if useConv:
            x = builder.conv2d(x, output_size, 1, stride, 'SAME')
        elif output_size == input_size:
            if stride == 1:
                x = inputs
            else:
                x = builder.mpool2d(x, 1, stride, 'VALID')
        else:
            x = inputs
    return x

def resnext_bottleneck_v1(builder, inputs, depth, depth_bottleneck, stride):
    num_inputs = inputs.get_shape().as_list()[1]
    x = inputs
    with tf.name_scope('resnext_bottleneck_v1'):
        shortcut = resnext_shortcut(builder, x, stride, num_inputs, depth)
        branches_list = []
        for i in range(builder.cardinality):
            branch = resnext_split_branch(builder, x, stride)
            branches_list.append(branch)
        concatenated_branches = tf.concat(values=branches_list, axis=1, name='concat')
        bottleneck_depth = concatenated_branches.get_shape().as_list()[1]
        x = builder.conv2d_linear(concatenated_branches, depth, 1, 1, 'SAME')
        x = tf.nn.relu(x + shortcut)
    return x

def inference_residual(builder, inputs, layer_counts, bottleneck_callback):
    x = inputs
    x = builder.conv2d(       x, 64, 7, 2, 'VALID')
    x = builder.max_pooling2d(x,     3, 2, 'SAME')
    for i in range(layer_counts[0]):
        x = bottleneck_callback(builder, x,  256,  64, 1)
    for i in range(layer_counts[1]):
        x = bottleneck_callback(builder, x, 512, 128, 2 if i==0 else 1)
    for i in range(layer_counts[2]):
        x = bottleneck_callback(builder, x, 1024, 256, 2 if i==0 else 1)
    for i in range(layer_counts[3]):
        x = bottleneck_callback(builder, x, 2048, 512, 2 if i==0 else 1)
    x = builder.spatial_average2d(x)
    return x

def inference_resnext_v1_impl(builder, inputs, layer_counts):
    return inference_residual(builder, inputs, layer_counts, resnext_bottleneck_v1)

def resnext_v1(inputs, training):
    """Aggregated  Residual Networks family of models
    https://arxiv.org/abs/1611.05431
    """
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training, use_batch_norm=True)
    cardinality_to_bottleneck_width = { 1:64, 2:40, 4:24, 8:14, 32:4 }
    builder.cardinality = 32
    builder.shortcut_type = 'B'
    assert builder.cardinality in cardinality_to_bottleneck_width.keys(), \
    "Invalid  cardinality (%i); must be one of: 1,2,4,8,32" % builder.cardinality
    builder.bottleneck_width = cardinality_to_bottleneck_width[builder.cardinality]
    if   flags.layers ==  50: return inference_resnext_v1_impl(builder, inputs, [3,4, 6,3])
    elif flags.layers == 101: return inference_resnext_v1_impl(builder, inputs, [3,4,23,3])
    elif flags.layers == 152: return inference_resnext_v1_impl(builder, inputs, [3,8,36,3])
    else: raise ValueError("Invalid nlayer (%i); must be one of: 50,101,152" %
                           nlayer)

nvutils.train(resnext_v1, args)

if args['log_dir'] is not None:
    nvutils.validate(resnext_v1, args)

