# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
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
# =============================================================================

import os
import glob
import shutil
import subprocess
import tensorflow as tf
import nets.nets_factory
import tensorflow.contrib.slim as slim
import official.resnet.imagenet_main
from preprocessing import inception_preprocessing, vgg_preprocessing

class NetDef(object):
    """Contains definition of a model
    
    name: Name of model
    url: (optional) Where to download archive containing checkpoint
    model_dir_in_archive: (optional) Subdirectory in archive containing
        checkpoint files.
    preprocess: Which preprocessing method to use for inputs.
    input_size: Input dimensions.
    slim: If True, use tensorflow/research/slim/nets to build graph. Else, use
        model_fn to build graph.
    postprocess: Postprocessing function on predictions.
    model_fn: Function to build graph if slim=False
    num_classes: Number of output classes in model. Background class will be
        automatically adjusted for if num_classes is 1001.
    """
    def __init__(self, name, url=None, model_dir_in_archive=None,
                checkpoint_name=None, preprocess='inception',
            input_size=224, slim=True, postprocess=tf.nn.softmax, model_fn=None, num_classes=1001):
        self.name = name
        self.url = url
        self.model_dir_in_archive = model_dir_in_archive
        self.checkpoint_name = checkpoint_name
        if preprocess == 'inception':
            self.preprocess = inception_preprocessing.preprocess_image
        elif preprocess == 'vgg':
            self.preprocess = vgg_preprocessing.preprocess_image
        self.input_width = input_size
        self.input_height = input_size
        self.slim = slim
        self.postprocess = postprocess
        self.model_fn = model_fn
        self.num_classes = num_classes

    def get_input_dims(self):
        return self.input_width, self.input_height

    def get_num_classes(self):
        return self.num_classes

def get_netdef(model):
    """
    Creates the dictionary NETS with model names as keys and NetDef as values.
    Returns the NetDef corresponding to the model specified in the parameter.

    model: string, the model name (see NETS table)
    """
    NETS = {
        'mobilenet_v1': NetDef(
            name='mobilenet_v1',
            url='http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz'),

        'mobilenet_v2': NetDef(
            name='mobilenet_v2_140',
            url='https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz'),

        'nasnet_mobile': NetDef(
            name='nasnet_mobile',
            url='https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz'),

        'nasnet_large': NetDef(
            name='nasnet_large',
            url='https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz',
            input_size=331),

        'resnet_v1_50': NetDef(
            name='resnet_v1_50',
            url='http://download.tensorflow.org/models/official/20180601_resnet_v1_imagenet_checkpoint.tar.gz',
            model_dir_in_archive='20180601_resnet_v1_imagenet_checkpoint',
            slim=False,
            preprocess='vgg',
            model_fn=official.resnet.imagenet_main.ImagenetModel(resnet_size=50, resnet_version=1)),

        'resnet_v2_50': NetDef(
            name='resnet_v2_50',
            url='http://download.tensorflow.org/models/official/20180601_resnet_v2_imagenet_checkpoint.tar.gz',
            model_dir_in_archive='20180601_resnet_v2_imagenet_checkpoint',
            slim=False,
            preprocess='vgg',
            model_fn=official.resnet.imagenet_main.ImagenetModel(resnet_size=50, resnet_version=2)),

        'resnet_v2_152': NetDef(
            name='resnet_v2_152',
            slim=False,
            preprocess='vgg',
            model_fn=official.resnet.imagenet_main.ImagenetModel(resnet_size=152, resnet_version=2)),

        'vgg_16': NetDef(
            name='vgg_16',
            url='http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
            preprocess='vgg',
            num_classes=1000),

        'vgg_19': NetDef(
            name='vgg_19',
            url='http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
            preprocess='vgg',
            num_classes=1000),

        'inception_v3': NetDef(
            name='inception_v3',
            url='http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
            input_size=299),

        'inception_v4': NetDef(
            name='inception_v4',
            url='http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
            input_size=299),
    }
    return NETS[model]

def _deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text

def get_preprocess_fn(model, mode='classification'):
    """Creates a function to parse and process a TFRecord using the model's parameters

    model: string, the model name (see NETS table)
    mode: string, whether the model is for classification or detection
    returns: function, the preprocessing function for a record
    """
    def process(record):
        # Parse TFRecord
        imgdata, label, bbox, text = _deserialize_image_record(record)
        label -= 1 # Change to 0-based (don't use background class)
        try:    image = tf.image.decode_jpeg(imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        except: image = tf.image.decode_png(imgdata, channels=3)
        # Use model's preprocessing function
        netdef = get_netdef(model)
        image = netdef.preprocess(image, netdef.input_height, netdef.input_width, is_training=False)
        return image, label

    return process

def build_classification_graph(model, model_dir=None, default_models_dir='./data'):
    """Builds an image classification model by name

    This function builds an image classification model given a model
    name, parameter checkpoint file path, and number of classes.  This
    function performs some graph processing to produce a graph that is
    well optimized by the TensorRT package in TensorFlow 1.7+.

    model: string, the model name (see NETS table)
    model_dir: string, optional user provided checkpoint location
    default_models_dir: string, directory to store downloaded model checkpoints
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    netdef = get_netdef(model)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            tf_input = tf.placeholder(tf.float32, [None, netdef.input_height, netdef.input_width, 3], name='input')
            if netdef.slim:
                # TF Slim Model: get model function from nets_factory
                network_fn = nets.nets_factory.get_network_fn(netdef.name, netdef.num_classes,
                        is_training=False)
                tf_net, tf_end_points = network_fn(tf_input)
            else:
                # TF Official Model: get model function from NETS
                tf_net = netdef.model_fn(tf_input, training=False)

            tf_output = tf.identity(tf_net, name='logits')
            num_classes = tf_output.get_shape().as_list()[1]
            if num_classes == 1001:
                # Shift class down by 1 if background class was included
                tf_output_classes = tf.add(tf.argmax(tf_output, axis=1), -1, name='classes')
            else:
                tf_output_classes = tf.argmax(tf_output, axis=1, name='classes')

            # Get checkpoint.
            checkpoint_path = get_checkpoint(model, model_dir, default_models_dir)
            print('Using checkpoint:', checkpoint_path)
            # load checkpoint
            tf_saver = tf.train.Saver()
            tf_saver.restore(save_path=checkpoint_path, sess=tf_sess)

            # freeze graph
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=['logits', 'classes']
            )

    return frozen_graph

def get_checkpoint(model, model_dir=None, default_models_dir='.'):
    """Get the checkpoint. User may provide their own checkpoint via model_dir.
    If model_dir is None, attempts to download the checkpoint using url property
    from model definition (see get_netdef()). default_models_dir/model is first
    checked to see if the checkpoint was already downloaded. If not, the
    checkpoint will be downloaded from the url.

    model: string, the model name (see NETS table)
    model_dir: string, optional user provided checkpoint location
    default_models_dir: string, the directory where files are downloaded to
    returns: string, path to the checkpoint file containing trained model params
    """
    # User has provided a checkpoint
    if model_dir:
        checkpoint_path = find_checkpoint_in_dir(model_dir)
        if not checkpoint_path:
            print('No checkpoint was found in', model_dir)
            exit(1)
        return checkpoint_path

    # User has not provided a checkpoint. We need to download one. First check
    # if checkpoint was already downloaded and stored in default_models_dir.
    model_dir = os.path.join(default_models_dir, model)
    checkpoint_path = find_checkpoint_in_dir(model_dir)
    if checkpoint_path:
        return checkpoint_path

    # Checkpoint has not yet been downloaded. Download checkpoint if model has
    # defined a URL.
    if get_netdef(model).url:
        download_checkpoint(model, model_dir)
        return find_checkpoint_in_dir(model_dir)
    
    print('No model_dir was provided and the model does not define a download' \
          ' URL.')
    exit(1)

def find_checkpoint_in_dir(model_dir):
    # tf.train.latest_checkpoint will find checkpoints if a 'checkpoint' file is
    # present in the directory.
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    if checkpoint_path:
        return checkpoint_path

    # tf.train.latest_checkpoint did not find anything. Find .ckpt file
    # manually.
    files = glob.glob(os.path.join(model_dir, '*.ckpt*'))
    if len(files) == 0:
        return None
    # Use last file for consistency if more than one (may not actually be
    # "latest").
    checkpoint_path = sorted(files)[-1]
    # Trim after .ckpt-* segment. For example:
    # model.ckpt-257706.data-00000-of-00002 -> model.ckpt-257706
    parts = checkpoint_path.split('.')
    ckpt_index = [i for i in range(len(parts)) if 'ckpt' in parts[i]][0]
    checkpoint_path = '.'.join(parts[:ckpt_index+1])
    return checkpoint_path

def download_checkpoint(model, destination_path):
    # Make directories if they don't exist.
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    # Download archive.
    archive_path = os.path.join(destination_path,
                                os.path.basename(get_netdef(model).url))
    if not os.path.isfile(archive_path):
        subprocess.call(['wget', '--no-check-certificate',
                         get_netdef(model).url, '-O', archive_path])
    # Extract.
    subprocess.call(['tar', '-xzf', archive_path, '-C', destination_path])
    # Move checkpoints out of archive sub directories into destination_path
    if get_netdef(model).model_dir_in_archive:
        source_files = os.path.join(destination_path,
                                    get_netdef(model).model_dir_in_archive,
                                    '*')
        for f in glob.glob(source_files):
            shutil.copy2(f, destination_path)
