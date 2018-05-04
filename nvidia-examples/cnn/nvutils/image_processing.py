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

import tensorflow as tf
import sys
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import batching

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

def _decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')

def _crop_and_resize_image(image, original_bbox, height, width, rank=0, distort=False):
    with tf.name_scope('crop_and_resize'):
        # Evaluation is done on a center-crop of this ratio
        eval_crop_ratio = 0.8
        if distort:
            # Note: Only the aspect ratio of this really matters, because we
            #       only use the normalized bbox returned by
            #       tf.image.sample_distorted_bounding_box.
            initial_shape = [int(round(height / eval_crop_ratio)),
                             int(round(width  / eval_crop_ratio)),
                             3]
            bbox_begin, bbox_size, bbox = \
                tf.image.sample_distorted_bounding_box(
                    initial_shape,
                    bounding_boxes=tf.zeros(shape=[1,0,4]), # No bounding boxes
                    min_object_covered=0.25,
                    aspect_ratio_range=[0.8, 1.25],
                    area_range=[0.25, 1.0],
                    max_attempts=100,
                    seed=11 * rank, # Need to set for deterministic results
                    use_image_if_no_bounding_boxes=True)
            bbox = bbox[0,0] # Remove batch, box_idx dims
        else:
            # Central crop
            ratio_y = ratio_x = eval_crop_ratio
            bbox = tf.constant([0.5 * (1 - ratio_y), 0.5 * (1 - ratio_x),
                                0.5 * (1 + ratio_y), 0.5 * (1 + ratio_x)])
        image = tf.image.crop_and_resize(
            image[None,:,:,:], bbox[None,:], [0], [height, width])[0]
        image = tf.clip_by_value(image, 0., 255.)
        image = tf.cast(image, tf.uint8)
        return image

def _distort_image_color(image, order=0):
    with tf.name_scope('distort_color'):
        image = tf.multiply(image, 1. / 255.)
        brightness = lambda img: tf.image.random_brightness(img, max_delta=32. / 255.)
        saturation = lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5)
        hue        = lambda img: tf.image.random_hue(img, max_delta=0.2)
        contrast   = lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5)
        if order == 0: ops = [brightness, saturation, hue, contrast]
        else:          ops = [brightness, contrast, saturation, hue]
        for op in ops:
            image = op(image)
        # The random_* ops do not necessarily clamp the output range
        image = tf.clip_by_value(image, 0.0, 1.0)
        # Restore the original scaling
        image = tf.multiply(image, 255.)
        return image

def _parse_and_preprocess_image_record(record, counter, height, width, seed,
                                      distort=False, nsummary=10):
    imgdata, label, bbox, text = _deserialize_image_record(record)
    label -= 1 # Change to 0-based (don't use background class)
    with tf.name_scope('preprocess_train'):
        try:    image = _decode_jpeg(imgdata, channels=3)
        except: image = tf.image.decode_png(imgdata, channels=3)
        # TODO: Work out a not-awful way to do this with counter being a Tensor
        #if counter < nsummary:
        #    image_with_bbox = tf.image.draw_bounding_boxes(
        #        tf.expand_dims(tf.to_float(image), 0), bbox)
        #    tf.summary.image('original_image_and_bbox', image_with_bbox)
        image = _crop_and_resize_image(image, bbox, height, width, seed, distort)
        #if counter < nsummary:
        #    tf.summary.image('cropped_resized_image', tf.expand_dims(image, 0))
        if distort:
            image = tf.image.random_flip_left_right(image)
        #if counter < nsummary:
        #    tf.summary.image('flipped_image', tf.expand_dims(image, 0))
        return image, label

def image_set(filenames, batch_size, height, width,
                 training=False, rank=0, nranks=1, num_threads=10, nsummary=10):
    shuffle_buffer_size = 10000
    num_readers = 1
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if training:
        #ds = ds.shard(nranks, rank)
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size, seed=5 * (1 + rank))
    ds = ds.interleave(
        tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))
    preproc_func = lambda record, counter_: _parse_and_preprocess_image_record(
        record, counter_, height, width, rank,
        distort=training, nsummary=nsummary if training else 0)
    ds = ds.map(preproc_func, num_parallel_calls=num_threads)
    if training:
        ds = ds.shuffle(shuffle_buffer_size, seed=7 * (1 + rank))
    ds = ds.batch(batch_size)
    return ds

def image_set_new(filenames, batch_size, height, width,
                     training=False, rank=0, nranks=1,
                     num_threads=10, nsummary=10,
                     cache_data=False, num_splits=1):
    ds = tf.data.TFRecordDataset.list_files(filenames)
    #ds = ds.shard(nranks, rank) # HACK TESTING
    ds = ds.shuffle(buffer_size=10000, seed=5 * (1 + rank))
    ds = ds.apply(interleave_ops.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))
    if cache_data:
        ds = ds.take(1).cache().repeat()
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if training:
        ds = ds.shuffle(buffer_size=10000, seed=7 * (1 + rank)) # TODO: Need to specify seed=?
    ds = ds.repeat()
    preproc_func = lambda record, counter_: _parse_and_preprocess_image_record(
        record, counter_, height, width, rank,
        distort=training, nsummary=nsummary if training else 0)
    assert(batch_size % num_splits == 0)
    ds = ds.apply(
        batching.map_and_batch(
            map_func=preproc_func,
            batch_size=batch_size // num_splits,
            num_parallel_batches=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)
    return ds
