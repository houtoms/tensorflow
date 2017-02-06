#!/usr/bin/env python

# ***WARNING***
# This code is based on several (public) third-party sources and has not been
# cleared for public distribution.
# ***WARNING***

#  Copyright (c) 2016, NVIDIA Corporation
#  All rights reserved.

"""
 This is a mashup and extension of Soumith's convnet benchmark scripts,
   TensorFlow's Inception v3 training scripts, the lm_benchmark.py script,
   and TFSlim's ResNet implementation.
 It is intended for use as a benchmarking tool that provides complete
   control over how TF is used (as opposed to say relying on Keras or
   TFSlim, which may not be GPU-optimal).
 Contact Ben Barsdell for further information.

TODO: Fix queue error when using crop with non-resized images
      Consider adding random shift before crop in --resize_method=crop mode
      Add validation results (loss and top-1/5 classification error)

---------
Changelog
---------
v0.11
Fixed incorrect batchnorm in NCHW mode (NOTE: With TF <= 0.11, NHWC is faster than NCHW for batchnorm models; for TF >= 0.12, NCHW is now fused and is much faster)
Fixed slightly incorrect ResNet definition
Fixed results mismatch when using NCHW vs. NHWC (issue was with random seeds and final data order)
Fixed initialization factor for ReLU layers (2/Nin -> 1/Nin), which greatly improves initial training loss
Fixed error on exit when num_batches <= 10
Fixed all input modes to use uint8 data type
Added cifar10, vgg13+16, resnet18+34 and xception models
Added support for LRN and separable convolution layers
Added graceful exit when Ctrl-C is pressed
Added learning rate decay
Added lr_decay_factor, lr_decay_steps, lr_decay_staircase and nesterov parameters
Added summaries_interval parameter (seconds between summary dumps)
Added checkpoint loading
Added exponential averaging to reported loss (NOTE: With TF <= 0.11, the average starts out heavily biased to loss=0; this is fixed in TF 0.12)
Added cast from fp16 to fp32 before softmax
Changed training log output to include loss and effective accuracy instead of exp(loss)
Changed default learning rate from 0.005 to 0.003
Updated several deprecated API calls

v0.10
Added --resize_method=crop/bilinear/trilinear/area, with fast crop support (requires resized dataset)
Added --device=cpu/gpu to specify computation device
Added --cpu as a shortcut for the flags required to run purely on the CPU
Found that crop_to_bounding_box is much faster than resize_with_crop_or_pad
Changed number of output classes to always be 1000 (so model is now independent of dataset)
Changed jpeg decoding to fancy_upscaling=False (no obvious perf difference)
Changed default learning rate from 0.002 to 0.005
Changed default weight decay from 1e-4 to 1e-5

v0.9
Added vgg19 model and renamed old model to vgg11
Added lenet model
Added --display_every option
Reduced CPU bottleneck by moving float conversion+scaling to the GPU (only in --nodistortions mode)
Changed --num_intra_threads to default to 0 (i.e., system-defined)
  Note: This made a significant difference while experimenting with 8-bit PCIe
          transfers, but no difference with fp32 (which was faster overall).
Changed --nodistortions mode to include random bbox cropping
Changed distortions to default to False
Added show_memory=True to Chrome trace format output
Fixed some compatibility issues for TF r0.11
Fixed shape bug in --gpu_prefetch mode when using multiple GPUs

v0.8
Enabled displaying of default values in cmd line help
Added --inference mode
Added --summaries_dir for graph/training visualisation with TensorBoard
Added --num_inter_threads for controlling TF inter-op thread pool
Added --include_h2d_in_synthetic for optionally enabling h2d memcopy in synthetic mode
Added --gpu_prefetch for async H2D PCIe data copying (HIGHLY EXPERIMENTAL)
Changed --num_readers to default to 1 instead of 4
Removed some old unused code

v0.7
Fixed --distortions not defaulting to True
Added num_intra_threads option with default of 1 (slightly faster than default of 0 => TF decides)
Added printing of final performance statistics
"""

import tensorflow as tf

def tensorflow_version_tuple():
	v = tf.__version__
	major, minor, patch = v.split('.')
	return (int(major), int(minor), patch)
def tensorflow_version():
	vt = tensorflow_version_tuple()
	return vt[0]*100 + vt[1]

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.client import timeline
import numpy as np
import time
from collections import defaultdict
import os
import json
import argparse
import sys
from ctypes import cdll
import threading

libcudart = cdll.LoadLibrary('libcudart.so')
def cudaProfilerStart():
	libcudart.cudaProfilerStart()
def cudaProfilerStop():
	libcudart.cudaProfilerStop()

class ConvNetBuilder(object):
	def __init__(self, input_op, input_nchan, phase_train,
	             data_format='NCHW', data_type=tf.float32):
		self.top_layer   = input_op
		self.top_size    = input_nchan
		self.phase_train = phase_train
		self.data_format = data_format
		self.data_type   = data_type
		self.counts      = defaultdict(lambda: 0)
		self.use_batch_norm = False
		self.batch_norm_config = {}#'decay': 0.997, 'scale': True}
		self.top_layer = self.from_nhwc(self.top_layer)
	def conv(self, nOut, kH, kW, dH=1, dW=1, mode='SAME', input_layer=None, nIn=None,
	         batch_norm=None, activation='relu'):
		if input_layer is None:
			input_layer = self.top_layer
		if nIn is None:
			nIn = self.top_size
		name = 'conv' + str(self.counts['conv'])
		self.counts['conv'] += 1
		#with tf.name_scope(name) as scope, tf.variable_scope(name) as varscope:
		with tf.variable_scope(name) as scope:
			seed = self.counts['conv']
			init_factor = 1. if activation == 'relu' else 1.
			kernel = tf.get_variable('weights', [kH, kW, nIn, nOut], self.data_type,
			                         tf.random_normal_initializer(stddev=np.sqrt(init_factor/(nIn*kH*kW)),
			                                                      seed=seed))
			strides = [1, dH, dW, 1]
			if self.data_format == 'NCHW':
				strides = [strides[0], strides[3], strides[1], strides[2]]
			if mode != 'SAME_RESNET':
				conv = tf.nn.conv2d(input_layer, kernel, strides, padding=mode,
				                    data_format=self.data_format)
			else: # Special padding mode for ResNet models
				if dH == 1 and dW == 1:
					conv = tf.nn.conv2d(input_layer, kernel, strides, padding='SAME',
					                    data_format=self.data_format)
				else:
					rate = 1 # Unused (for 'a trous' convolutions)
					kernel_size_effective = kH + (kW - 1) * (rate - 1)
					pad_total = kernel_size_effective - 1
					pad_beg = pad_total // 2
					pad_end = pad_total - pad_beg
					padding = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
					if self.data_format == 'NCHW':
						padding = [padding[0], padding[3], padding[1], padding[2]]
					input_layer = tf.pad(input_layer, padding)
					conv = tf.nn.conv2d(input_layer, kernel, strides, padding='VALID',
					                    data_format=self.data_format)
			if batch_norm is None:
				batch_norm = self.use_batch_norm
			if not batch_norm:
				biases = tf.get_variable('biases', [nOut], self.data_type,
				                         tf.constant_initializer(0.0))
				#biased = tf.reshape(tf.nn.bias_add(conv, biases, data_format=self.data_format), conv.get_shape())
				biased = tf.nn.bias_add(conv, biases, data_format=self.data_format)
			else:
				self.top_layer = conv
				self.top_size  = nOut
				biased = self.batch_norm(**self.batch_norm_config)
			if activation == 'relu':
				conv1 = tf.nn.relu(biased)#, name=scope)
			elif activation == 'linear' or activation is None:
				conv1 = biased
			else:
				raise KeyError("Invalid activation type '%s'" % activation)
			self.top_layer = conv1
			self.top_size  = nOut
			return conv1
	def separable_conv(self, nOut, kH, kW, dH=1, dW=1, depth_mult=1,
	                   mode='SAME',
	                   input_layer=None, nIn=None,
	                   batch_norm=None, activation='relu'):
		if input_layer is None:
			input_layer = self.top_layer
		if nIn is None:
			nIn = self.top_size
		name = 'sepconv' + str(self.counts['sepconv'])
		self.counts['sepconv'] += 1
		with tf.variable_scope(name) as scope:
			seed = self.counts['sepconv']
			init_factor = 1. if activation == 'relu' else 1.
			depth_filter = tf.get_variable('spatial_weights',
			                               [kH, kW, nIn, depth_mult],
			                               self.data_type,
			                               tf.random_normal_initializer(stddev=np.sqrt(init_factor/(nIn*kH*kW)),
			                                                            seed=seed))
			point_filter = tf.get_variable('channel_weights',
			                               [1, 1, nIn*depth_mult, nOut],
			                               self.data_type,
			                               tf.random_normal_initializer(stddev=np.sqrt(init_factor/(nIn*depth_mult)),
			                                                            seed=seed))
			strides = [1, dH, dW, 1]
			# TODO: Need to add support for data_format='NCHW' to this function
			input_layer = self.to_nhwc(input_layer)
			conv = tf.nn.separable_conv2d(input_layer,
			                              depth_filter,
			                              point_filter,
			                              strides,
			                              padding=mode)
			conv = self.from_nhwc(conv)
			if batch_norm is None:
				batch_norm = self.use_batch_norm
			if not batch_norm:
				biases = tf.get_variable('biases', [nOut], self.data_type,
				                         tf.constant_initializer(0.0))
				biased = tf.reshape(tf.nn.bias_add(conv, biases,
				                                   data_format=self.data_format),
				                    conv.get_shape())
			else:
				self.top_layer = conv
				self.top_size  = nOut
				biased = self.batch_norm(**self.batch_norm_config)
			if activation == 'relu':
				conv1 = tf.nn.relu(biased)
			elif activation == 'linear' or activation is None:
				conv1 = biased
			else:
				raise KeyError("Invalid activation type '%s'" % activation)
			self.top_layer = conv1
			self.top_size  = nOut
			return conv1
	def mpool(self, kH, kW, dH=2, dW=2, mode='VALID', input_layer=None, nIn=None):
		if input_layer is None:
			input_layer = self.top_layer
		else:
			#self.top_size = None # Reset because we no longer know what it is
			self.top_size = nIn
		name = 'mpool' + str(self.counts['mpool'])
		self.counts['mpool'] += 1
		#with tf.name_scope(name) as scope:
		#with tf.variable_scope(name) as scope:
		if self.data_format == 'NHWC':
			pool = tf.nn.max_pool(input_layer,
			                      ksize=[1, kH, kW, 1],
			                      strides=[1, dH, dW, 1],
			                      padding=mode,
			                      name=name)
		else:
			pool = tf.nn.max_pool(input_layer,
			                      ksize=[1, 1, kH, kW],
			                      strides=[1, 1, dH, dW],
			                      padding=mode,
			                      data_format='NCHW',
			                      name=name)
		self.top_layer = pool
		return pool
	def apool(self, kH, kW, dH=2, dW=2, mode='VALID', input_layer=None, nIn=None):
		if input_layer is None:
			input_layer = self.top_layer
		else:
			#self.top_size = None # Reset because we no longer know what it is
			self.top_size = nIn
		name = 'apool' + str(self.counts['apool'])
		self.counts['apool'] += 1
		#with tf.name_scope(name) as scope:
		#with tf.variable_scope(name) as scope:
		if self.data_format == 'NHWC':
			pool = tf.nn.avg_pool(input_layer,
			                      ksize=[1, kH, kW, 1],
			                      strides=[1, dH, dW, 1],
			                      padding=mode,
			                      name=name)
		else:
			pool = tf.nn.avg_pool(input_layer,
			                      ksize=[1, 1, kH, kW],
			                      strides=[1, 1, dH, dW],
			                      padding=mode,
			                      data_format='NCHW',
			                      name=name)
		self.top_layer = pool
		return pool
	def lrn(self, depth_radius=5, input_layer=None, **kwargs):
		if input_layer is None:
			input_layer = self.top_layer
		else:
			self.top_size = None # Reset because we no longer know what it is
		name = 'lrn' + str(self.counts['lrn'])
		self.counts['lrn'] += 1
		# TODO: Need to implement native support for NCHW into tf.nn.local_response_normalization
		# TODO: Also need to implement support for fp16 LRN on GPU!
		input_layer = self.to_nhwc(input_layer)
		input_layer = self.to_fp32(input_layer)
		self.top_layer = tf.nn.local_response_normalization(input_layer,
		                                                    depth_radius,
		                                                    name=name,
		                                                    **kwargs)
		self.top_layer = self.from_fp32(self.top_layer)
		self.top_layer = self.from_nhwc(self.top_layer)
		return self.top_layer
	def reshape(self, shape, input_layer=None):
		if input_layer is None:
			input_layer = self.top_layer
		
		# Note: This ensures that the results for NCHW match those for NHWC
		input_layer = self.to_nhwc(input_layer)
		
		self.top_layer = tf.reshape(input_layer, shape)
		self.top_size  = int(self.top_layer.get_shape()[-1]) # HACK This may not always work
		return self.top_layer
	def flatten(self):
		self.reshape([int(self.top_layer.get_shape()[0]), -1])
	def affine(self, nOut, input_layer=None, nIn=None, activation='relu'):
		if input_layer is None:
			input_layer = self.top_layer
		if nIn is None:
			nIn = self.top_size
		name = 'affine' + str(self.counts['affine'])
		self.counts['affine'] += 1
		with tf.variable_scope(name) as scope:
			seed = self.counts['affine']
			init_factor = 1. if activation == 'relu' else 1.
			kernel = tf.get_variable('weights', [nIn, nOut], self.data_type,
			                         tf.random_normal_initializer(stddev=np.sqrt(init_factor/(nIn)),
			                                                      seed=seed))
			biases = tf.get_variable('biases', [nOut], self.data_type,
			                         tf.constant_initializer(0.0))
			logits = tf.matmul(input_layer, kernel) + biases
			if activation == 'relu':
				#affine1 = tf.nn.relu_layer(input_layer, kernel, biases, name=name)
				affine1 = tf.nn.relu(logits)#, name=name)
			elif activation == 'linear' or activation is None:
				affine1 = logits
			else:
				raise KeyError("Invalid activation type '%s'" % activation)
			self.top_layer = affine1
			self.top_size  = nOut
			return affine1
	def resnet_bottleneck_v1(self, depth, depth_bottleneck, stride, basic=False, input_layer=None, inSize=None):
		if input_layer is None:
			input_layer = self.top_layer
		if inSize is None:
			inSize = self.top_size
		name = 'resnet_v1' + str(self.counts['resnet_v1'])
		self.counts['resnet_v1'] += 1
		with tf.variable_scope(name) as scope:
			if depth == inSize:
				if stride == 1:
					shortcut = input_layer
				else:
					shortcut = self.mpool(1, 1, stride, stride, input_layer=input_layer, nIn=inSize)
			else:
				shortcut = self.conv(depth, 1, 1, stride, stride, activation=None, input_layer=input_layer, nIn=inSize)
			if basic:
				res_ = self.conv(depth_bottleneck, 3, 3, stride, stride, mode='SAME_RESNET', input_layer=input_layer, nIn=inSize)
				res  = self.conv(depth,            3, 3, 1, 1, activation=None)
			else:
				res_ = self.conv(depth_bottleneck, 1, 1, 1, 1, input_layer=input_layer, nIn=inSize)
				res_ = self.conv(depth_bottleneck, 3, 3, stride, stride, mode='SAME_RESNET')
				res  = self.conv(depth,            1, 1, 1, 1, activation=None)
			output = tf.nn.relu(shortcut + res)
			self.top_layer = output
			self.top_size  = depth
			return output
	def inception_module(self, name, cols, input_layer=None, inSize=None):
		if input_layer is None:
			input_layer = self.top_layer
		if inSize is None:
			inSize = self.top_size
		name += str(self.counts[name])
		self.counts[name] += 1
		with tf.variable_scope(name) as scope:
			col_layers      = []
			col_layer_sizes = []
			for c, col in enumerate(cols):
				col_layers.append([])
				col_layer_sizes.append([])
				for l, layer in enumerate(col):
					ltype, args = layer[0], layer[1:]
					kwargs = {'input_layer': input_layer, 'nIn': inSize} if l==0 else {}
					if   ltype == 'conv':  self.conv (*args, **kwargs)
					elif ltype == 'mpool': self.mpool(*args, **kwargs)
					elif ltype == 'apool': self.apool(*args, **kwargs)
					elif ltype == 'share': # Share matching layer from previous column
						self.top_layer = col_layers[c-1][l]
						self.top_size  = col_layer_sizes[c-1][l]
					else: raise KeyError("Invalid layer type for inception module: '%s'" % ltype)
					col_layers[c].append(self.top_layer)
					col_layer_sizes[c].append(self.top_size)
			catdim = 3 if self.data_format == 'NHWC' else 1
			self.top_layer = array_ops.concat(catdim, [layers[-1] for layers in col_layers])
			self.top_size  = sum([sizes[-1] for sizes in col_layer_sizes])
			return self.top_layer
	def residual(self, net, nout=None, dH=1, dW=1, scale=1.0, activation='relu'):
		inlayer = self.top_layer
		insize  = self.top_size
		reslayer = inlayer
		if nout is not None:
			reslayer = self.conv(nout, 1, 1, dH, dW, activation=None)
		self.top_layer = inlayer
		self.top_size  = insize
		net(self)
		self.top_layer = reslayer + scale*self.top_layer
		return self.activate(activation)
	def activate(self, activation='relu'):
		if activation == 'relu':
			self.top_layer = tf.nn.relu(self.top_layer)
		elif activation != 'linear' and activation is not None:
			raise ValueError("Invalid activation '%s'" % activation)
		return self.top_layer
	def spatial_mean(self, keep_dims=False):
		name = 'spatial_mean' + str(self.counts['spatial_mean'])
		self.counts['spatial_mean'] += 1
		axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
		#with tf.name_scope(name) as scope:
		#with tf.variable_scope(name) as scope:
		self.top_layer = tf.reduce_mean(self.top_layer, axes, keep_dims=keep_dims, name=name)
		return self.top_layer
	def dropout(self, keep_prob=0.5, input_layer=None):
		if input_layer is None:
			input_layer = self.top_layer
		else:
			self.top_size = None
		name = 'dropout' + str(self.counts['dropout'])
		self.counts['dropout'] += 1
		with tf.variable_scope(name) as scope:
			seed = self.counts['dropout']
			keep_prob_tensor = tf.constant(keep_prob, dtype=self.data_type)
			one_tensor       = tf.constant(1.0,       dtype=self.data_type)
			keep_prob_op = control_flow_ops.cond(self.phase_train,
			                                     lambda: keep_prob_tensor,
			                                     lambda: one_tensor)
			dropout = tf.nn.dropout(input_layer, keep_prob_op, seed=seed)
			self.top_layer = dropout
			return dropout
	def batch_norm(self, input_layer=None, **kwargs):
		if input_layer is None:
			input_layer = self.top_layer
		else:
			self.top_size = None
		name = 'batchnorm' + str(self.counts['batchnorm'])
		self.counts['batchnorm'] += 1
		with tf.variable_scope(name) as scope:
			if tensorflow_version() <= 11:
				input_layer = self.to_nhwc(input_layer)
				bn = tf.contrib.layers.batch_norm(input_layer,
				                                  is_training=self.phase_train,
				                                  scope=scope,
				                                  **kwargs)
				bn = self.from_nhwc(bn)
			else:
				bn = tf.contrib.layers.batch_norm(input_layer,
				                                  is_training=self.phase_train,
				                                  scope=scope,
				                                  data_format=self.data_format,
				                                  fused=(self.data_format=='NCHW'),
				                                  **kwargs)
		self.top_layer = bn
		return bn
	def to_nhwc(self, x):
		return tf.transpose(x, [0,2,3,1]) if self.data_format == 'NCHW' else x
	def from_nhwc(self, x):
		return tf.transpose(x, [0,3,1,2]) if self.data_format == 'NCHW' else x
	def to_fp32(self, x):
		return tf.cast(x, tf.float32) if x.dtype != tf.float32 else x
	def from_fp32(self, x):
		return tf.cast(x, self.data_type) if x.dtype != self.data_type else x

def inference_overfeat(cnn):
	# Note: VALID requires padding the images by 3 in width and height
	cnn.conv (96, 11, 11, 4, 4, mode='VALID')
	cnn.mpool(2, 2)
	cnn.conv (256, 5, 5, 1, 1, mode='VALID')
	cnn.mpool(2, 2)
	cnn.conv ( 512, 3, 3)
	cnn.conv (1024, 3, 3)
	cnn.conv (1024, 3, 3)
	cnn.mpool(2, 2)
	cnn.reshape([-1, 1024 * 6 * 6])
	cnn.affine(3072)
	cnn.affine(4096)
	return cnn

def inference_alexnet(cnn):
	# Note: VALID requires padding the images by 3 in width and height
	cnn.conv (64, 11, 11, 4, 4, 'VALID')
	cnn.mpool(3, 3, 2, 2)
	cnn.conv (192, 5, 5)
	cnn.mpool(3, 3, 2, 2)
	cnn.conv (384, 3, 3)
	cnn.conv (256, 3, 3)
	cnn.conv (256, 3, 3)
	cnn.mpool(3, 3, 2, 2)
	cnn.reshape([-1, 256 * 6 * 6])
	cnn.affine(4096)
	cnn.dropout()
	cnn.affine(4096)
	cnn.dropout()
	return cnn

def inference_vgg_impl(cnn, layer_counts):
	for _ in xrange(layer_counts[0]):
		cnn.conv (64, 3, 3)
	cnn.mpool(2, 2)
	for _ in xrange(layer_counts[1]):
		cnn.conv (128, 3, 3)
	cnn.mpool(2, 2)
	for _ in xrange(layer_counts[2]):
		cnn.conv (256, 3, 3)
	cnn.mpool(2, 2)
	for _ in xrange(layer_counts[3]):
		cnn.conv (512, 3, 3)
	cnn.mpool(2, 2)
	for _ in xrange(layer_counts[4]):
		cnn.conv (512, 3, 3)
	cnn.mpool(2, 2)
	cnn.reshape([-1, 512 * 7 * 7])
	cnn.affine(4096)
	cnn.affine(4096)
	return cnn

#def inference_generic(cnn, nlayer, nstack, nfilter, filter_size,
#                      naffine, affine_size):
#	for _ in xrange(nlayer):
#		for _ in xrange(nstack):
#			cnn.conv(nfilter, filter_size, filter_size)
#		cnn.mpool(2, 2)
#	cnn.flatten()
#	for _ in xrange(naffine):
#		cnn.affine(affine_size)
#	return cnn

def inference_vgg11(cnn): # VGG model 'A'
	return inference_vgg_impl(cnn, [1,1,2,2,2])

def inference_vgg13(cnn): # VGG model 'B'
	return inference_vgg_impl(cnn, [2,2,2,2,2])

def inference_vgg16(cnn): # VGG model 'D'
	return inference_vgg_impl(cnn, [2,2,3,3,3])

def inference_vgg19(cnn): # VGG model 'E'
	return inference_vgg_impl(cnn, [2,2,4,4,4])

def inference_lenet5(cnn):
	# Note: This matches TF's MNIST tutorial model
	cnn.conv (32, 5, 5)
	cnn.mpool(2, 2)
	cnn.conv (64, 5, 5)
	cnn.mpool(2, 2)
	cnn.reshape([-1, 64 * 7 * 7])
	cnn.affine(512)
	return cnn

def inference_cifar10(cnn):
	# Note: This matches TF's CIFAR10 tutorial model
	cnn.conv (64, 5, 5)
	cnn.mpool(3, 3, mode='SAME')
	cnn.lrn(4, bias=1.0, alpha=0.001/9.0, beta=0.75)
	cnn.conv (64, 5, 5)
	cnn.lrn(4, bias=1.0, alpha=0.001/9.0, beta=0.75)
	cnn.mpool(3, 3, mode='SAME')
	cnn.reshape([-1, 64 * 8 * 8])
	cnn.affine(384)
	cnn.affine(192)
	return cnn

def inference_resnet_v1(cnn, layer_counts, basic=False):
	cnn.use_batch_norm = True
	cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}
	cnn.conv (64, 7, 7, 2, 2, mode='SAME_RESNET')
	cnn.mpool(3, 3, 2, 2, mode='SAME')
	for i in xrange(layer_counts[0]):
		cnn.resnet_bottleneck_v1( 256,  64, 1, basic=basic)
	for i in xrange(layer_counts[1]):
		cnn.resnet_bottleneck_v1( 512, 128, 2 if i==0 else 1, basic=basic)
	for i in xrange(layer_counts[2]):
		cnn.resnet_bottleneck_v1(1024, 256, 2 if i==0 else 1, basic=basic)
	for i in xrange(layer_counts[3]):
		cnn.resnet_bottleneck_v1(2048, 512, 2 if i==0 else 1, basic=basic)
	cnn.spatial_mean()
	return cnn

def inference_googlenet(cnn):
	def inception_v1(cnn, k, l, m, n, p, q):
		cols = [[('conv', k, 1, 1)],
		        [('conv', l, 1, 1), ('conv', m, 3, 3)],
		        [('conv', n, 1, 1), ('conv', p, 5, 5)],
		        [('mpool', 3, 3, 1, 1, 'SAME'), ('conv', q, 1, 1)]]
		return cnn.inception_module('incept_v1', cols)
	cnn.conv ( 64, 7, 7, 2, 2)
	cnn.mpool(3,   3,  2, 2, mode='SAME')
	cnn.conv ( 64, 1, 1)
	cnn.conv (192, 3, 3)
	cnn.mpool(3, 3, 2, 2, mode='SAME')
	inception_v1(cnn,  64,  96, 128, 16,  32,  32)
	inception_v1(cnn, 128, 128, 192, 32,  96,  64)
	cnn.mpool(3, 3, 2, 2, mode='SAME')
	inception_v1(cnn, 192,  96, 208, 16,  48,  64)
	inception_v1(cnn, 160, 112, 224, 24,  64,  64)
	inception_v1(cnn, 128, 128, 256, 24,  64,  64)
	inception_v1(cnn, 112, 144, 288, 32,  64,  64)
	inception_v1(cnn, 256, 160, 320, 32, 128, 128)
	cnn.mpool(3, 3, 2, 2, mode='SAME')
	inception_v1(cnn, 256, 160, 320, 32, 128, 128)
	inception_v1(cnn, 384, 192, 384, 48, 128, 128)
	cnn.apool(7, 7, 1, 1, mode='VALID')
	cnn.reshape([-1, 1024])
	return cnn

def inference_inception_v3(cnn):
	def inception_v3_a(cnn, n):
		cols = [[('conv', 64, 1, 1)],
		        [('conv', 48, 1, 1), ('conv', 64, 5, 5)],
		        [('conv', 64, 1, 1), ('conv', 96, 3, 3), ('conv', 96, 3, 3)],
		        [('apool', 3, 3, 1, 1, 'SAME'), ('conv', n, 1, 1)]]
		return cnn.inception_module('incept_v3_a', cols)
	def inception_v3_b(cnn):
		cols = [[('conv',  64, 1, 1), ('conv', 96, 3, 3), ('conv', 96, 3, 3, 2, 2, 'VALID')],
		        [('conv', 384, 3, 3, 2, 2, 'VALID')],
		        [('mpool', 3, 3, 2, 2, 'VALID')]]
		return cnn.inception_module('incept_v3_b', cols)
	def inception_v3_c(cnn, n):
		cols = [[('conv', 192, 1, 1)],
		        [('conv',   n, 1, 1), ('conv', n, 1, 7), ('conv', 192, 7, 1)],
		        [('conv',   n, 1, 1), ('conv', n, 7, 1), ('conv', n, 1, 7), ('conv', n, 7, 1), ('conv', 192, 1, 7)],
		        [('apool', 3, 3, 1, 1, 'SAME'), ('conv', 192, 1, 1)]]
		return cnn.inception_module('incept_v3_c', cols)
	def inception_v3_d(cnn):
		cols = [[('conv', 192, 1, 1), ('conv', 320, 3, 3, 2, 2, 'VALID')],
		        [('conv', 192, 1, 1), ('conv', 192, 1, 7), ('conv', 192, 7, 1), ('conv', 192, 3, 3, 2, 2, 'VALID')],
		        [('mpool', 3, 3, 2, 2, 'VALID')]]
		return cnn.inception_module('incept_v3_d',cols)
	def inception_v3_e(cnn, pooltype):
		cols = [[('conv', 320, 1, 1)],
		        [('conv', 384, 1, 1), ('conv', 384, 1, 3)],
		        [('share',),          ('conv', 384, 3, 1)],
		        [('conv', 448, 1, 1), ('conv', 384, 3, 3), ('conv', 384, 1, 3)],
		        [('share',),          ('share',),          ('conv', 384, 3, 1)],
		        [('mpool' if pooltype == 'max' else 'apool', 3, 3, 1, 1, 'SAME'), ('conv', 192, 1, 1)]]
		return cnn.inception_module('incept_v3_e', cols)
	
	# TODO: This does not include the extra 'arm' that forks off
	#         from before the 3rd-last module (the arm is designed
	#         to speed up training in the early stages).
	cnn.use_batch_norm = True
	cnn.conv (32, 3, 3, 2, 2, mode='VALID')
	cnn.conv (32, 3, 3, 1, 1, mode='VALID')
	cnn.conv (64, 3, 3, 1, 1, mode='SAME')
	cnn.mpool(3, 3, 2, 2, mode='VALID')
	cnn.conv ( 80, 1, 1, 1, 1, mode='VALID')
	cnn.conv (192, 3, 3, 1, 1, mode='VALID')
	cnn.mpool(3, 3, 2, 2, 'VALID')
	inception_v3_a(cnn, 32)
	inception_v3_a(cnn, 64)
	inception_v3_a(cnn, 64)
	inception_v3_b(cnn)
	inception_v3_c(cnn, 128)
	inception_v3_c(cnn, 160)
	inception_v3_c(cnn, 160)
	inception_v3_c(cnn, 192)
	inception_v3_d(cnn)
	inception_v3_e(cnn, 'avg')
	inception_v3_e(cnn, 'max')
	cnn.apool(8, 8, 1, 1, 'VALID')
	cnn.reshape([-1, 2048])
	return cnn

# Stem functions
def inception_v4_sa(cnn):
	cols = [[('mpool', 3, 3, 2, 2, 'VALID')],
	        [('conv', 96, 3, 3, 2, 2, 'VALID')]]
	return cnn.inception_module('incept_v4_sa', cols)
def inception_v4_sb(cnn):
	cols = [[('conv', 64, 1, 1), ('conv', 96, 3, 3, 1, 1, 'VALID')],
	        [('conv', 64, 1, 1), ('conv', 64, 7, 1), ('conv', 64, 1, 7), ('conv', 96, 3, 3, 1, 1, 'VALID')]]
	return cnn.inception_module('incept_v4_sb', cols)
def inception_v4_sc(cnn):
	cols = [[('conv', 192, 3, 3, 2, 2, 'VALID')],
	        [('mpool', 3, 3, 2, 2, 'VALID')]]
	return cnn.inception_module('incept_v4_sc', cols)
# Reduction functions
def inception_v4_ra(cnn, k, l, m, n):
	cols = [[('mpool',   3, 3, 2, 2, 'VALID')],
	        [('conv', n, 3, 3, 2, 2, 'VALID')],
	        [('conv', k, 1, 1), ('conv', l, 3, 3), ('conv', m, 3, 3, 2, 2, 'VALID')]]
	return cnn.inception_module('incept_v4_ra', cols)
def inception_v4_rb(cnn):
	cols = [[('mpool',   3, 3, 2, 2, 'VALID')],
	        [('conv', 192, 1, 1), ('conv', 192, 3, 3, 2, 2, 'VALID')],
	        [('conv', 256, 1, 1), ('conv', 256, 1, 7), ('conv', 320, 7, 1), ('conv', 320, 3, 3, 2, 2, 'VALID')]]
	return cnn.inception_module('incept_v4_rb', cols)
def inception_resnet_v2_rb(cnn):
	cols = [[('mpool',   3, 3, 2, 2, 'VALID')],
	        # TODO: These match the paper but don't match up with the following layer
	        #[('conv', 256, 1, 1), ('conv', 384, 3, 3, 2, 2, 'VALID')],
	        #[('conv', 256, 1, 1), ('conv', 288, 3, 3, 2, 2, 'VALID')],
	        #[('conv', 256, 1, 1), ('conv', 288, 3, 3), ('conv', 320, 3, 3, 2, 2, 'VALID')]]
	        # TODO: These match Facebook's Torch implem
	        [('conv', 256, 1, 1), ('conv', 384, 3, 3, 2, 2, 'VALID')],
	        [('conv', 256, 1, 1), ('conv', 256, 3, 3, 2, 2, 'VALID')],
	        [('conv', 256, 1, 1), ('conv', 256, 3, 3), ('conv', 256, 3, 3, 2, 2, 'VALID')]]
	return cnn.inception_module('incept_resnet_v2_rb', cols)

def inference_inception_v4(cnn):
	def inception_v4_a(cnn):
		cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv',  96, 1, 1)],
		        [('conv',  96, 1, 1)],
		        [('conv',  64, 1, 1), ('conv', 96, 3, 3)],
		        [('conv',  64, 1, 1), ('conv', 96, 3, 3), ('conv', 96, 3, 3)]]
		return cnn.inception_module('incept_v4_a', cols)
	def inception_v4_b(cnn):
		cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 128, 1, 1)],
		        [('conv', 384, 1, 1)],
		        [('conv', 192, 1, 1), ('conv', 224, 1, 7), ('conv', 256, 7, 1)],
		        [('conv', 192, 1, 1), ('conv', 192, 1, 7), ('conv', 224, 7, 1), ('conv', 224, 1, 7), ('conv', 256, 7, 1)]]
		return cnn.inception_module('incept_v4_b', cols)
	def inception_v4_c(cnn):
		cols = [[('apool', 3, 3, 1, 1, 'SAME'), ('conv', 256, 1, 1)],
		        [('conv', 256, 1, 1)],
		        [('conv', 384, 1, 1), ('conv', 256, 1, 3)],
		        [('share',),          ('conv', 256, 3, 1)],
		        [('conv', 384, 1, 1), ('conv', 448, 1, 3), ('conv', 512, 3, 1), ('conv', 256, 3, 1)],
		        [('share',),          ('share',),          ('share',),          ('conv', 256, 1, 3)]]
		return cnn.inception_module('incept_v4_c', cols)
	
	cnn.use_batch_norm = True
	cnn.conv (32, 3, 3, 2, 2, mode='VALID')
	cnn.conv (32, 3, 3, 1, 1, mode='VALID')
	cnn.conv (64, 3, 3)
	inception_v4_sa(cnn)
	inception_v4_sb(cnn)
	inception_v4_sc(cnn)
	for _ in xrange(4):
		inception_v4_a(cnn)
	inception_v4_ra(cnn, 192, 224, 256, 384)
	for _ in xrange(7):
		inception_v4_b(cnn)
	inception_v4_rb(cnn)
	for _ in xrange(3):
		inception_v4_c(cnn)
	cnn.spatial_mean()
	cnn.dropout(0.8)
	return cnn

def inference_inception_resnet_v2(cnn):
	def inception_resnet_v2_a(cnn):
		cols = [[('conv', 32, 1, 1)],
		        [('conv', 32, 1, 1), ('conv', 32, 3, 3)],
		        [('conv', 32, 1, 1), ('conv', 48, 3, 3), ('conv', 64, 3, 3)]]
		cnn.inception_module('incept_resnet_v2_a', cols)
		cnn.conv(384, 1, 1, activation=None)
	def inception_resnet_v2_b(cnn):
		cols = [[('conv', 192, 1, 1)],
		        [('conv', 128, 1, 1), ('conv', 160, 1, 7), ('conv', 192, 7, 1)]]
		cnn.inception_module('incept_resnet_v2_b', cols)
		cnn.conv(1152, 1, 1, activation=None)
	def inception_resnet_v2_c(cnn):
		cols = [[('conv', 192, 1, 1)],
		        [('conv', 192, 1, 1), ('conv', 224, 1, 3), ('conv', 256, 3, 1)]]
		cnn.inception_module('incept_resnet_v2_c', cols)
		cnn.conv(2048, 1, 1, activation=None)
	
	cnn.use_batch_norm = True
	residual_scale = 0.2
	cnn.conv (32, 3, 3, 2, 2, mode='VALID')
	cnn.conv (32, 3, 3, 1, 1, mode='VALID')
	cnn.conv (64, 3, 3)
	inception_v4_sa(cnn)
	inception_v4_sb(cnn)
	inception_v4_sc(cnn)
	for _ in xrange(5):
		cnn.residual(inception_resnet_v2_a, scale=residual_scale)
	inception_v4_ra(cnn, 256, 256, 384, 384)
	for _ in xrange(10):
		# TODO: This was 1154 in the paper, but then the layers don't match up
		#         One Caffe model online appears to use 1088
		#         Facebook's Torch implem uses 1152
		cnn.residual(inception_resnet_v2_b, scale=residual_scale)
	inception_resnet_v2_rb(cnn)
	for _ in xrange(5):
		# TODO: This was 2048 in the paper, but then the layers don't match up
		#         One Caffe model online appears to use 2080
		#         Facebook's Torch implem uses 2048 but modifies the preceding reduction net so that it matches
		#cnn.residual(inception_resnet_v2_c, 2144, scale=residual_scale)
		cnn.residual(inception_resnet_v2_c, scale=residual_scale)
	cnn.spatial_mean()
	cnn.dropout(0.8)
	return cnn

def inference_xception(cnn):
	def make_xception_entry(nout, activate_first=True):
		def xception_entry(cnn):
			if activate_first:
				cnn.activate('relu')
			cnn.separable_conv(nout, 3, 3)
			cnn.separable_conv(nout, 3, 3, activation=None)
			cnn.mpool(3, 3, mode='SAME')
		return xception_entry
	def xception_middle(cnn):
		cnn.activate('relu')
		cnn.separable_conv(728, 3, 3)
		cnn.separable_conv(728, 3, 3)
		cnn.separable_conv(728, 3, 3, activation=None)
	def xception_exit(cnn):
		cnn.activate('relu')
		cnn.separable_conv( 728, 3, 3)
		cnn.separable_conv(1024, 3, 3, activation=None)
		cnn.mpool(3, 3, mode='SAME')
	
	cnn.use_batch_norm = True
	cnn.batch_norm_config = {'decay': 0.99, 'epsilon': 1e-5, 'scale': True}
	cnn.conv(32, 3, 3, 2, 2, mode='VALID')
	cnn.conv(64, 3, 3, 1, 1, mode='VALID')
	cnn.residual(make_xception_entry(128, False), 128, 2, 2, activation=None)
	cnn.residual(make_xception_entry(256),        256, 2, 2, activation=None)
	cnn.residual(make_xception_entry(728),        728, 2, 2, activation=None)
	for _ in xrange(8):
		cnn.residual(xception_middle, activation=None)
	cnn.residual(xception_exit, 1024, 2, 2, activation=None)
	cnn.separable_conv(1536, 3, 3)
	cnn.separable_conv(2048, 3, 3)
	cnn.spatial_mean()
	# Note: Optional FC layer not included
	cnn.dropout(0.5)
	return cnn

def all_average_gradients(local_grads, devices):
	# TODO: This function can probably be written better
	n = len(local_grads)
	# Create mutable copies of grads
	all_grads = []
	for grads, device in zip(local_grads, devices):
		g = []
		for grad,var in grads:
			with tf.device(device):
				with tf.control_dependencies([grad]):
					g.append([tf.identity(grad),var])
		all_grads.append(g)
	for i, device in zip(xrange(n), devices):
		# Attempts to do a ring-like all-reduce
		for j in xrange(1, n):
			for g in xrange(len(all_grads[0])):
				with tf.device(device):
					with tf.control_dependencies([local_grads[(i+j)%n][g][0]]):  # TODO: Needed?
						all_grads[i][g][0] += local_grads[(i+j)%n][g][0]
		for g in xrange(len(local_grads[0])):
			all_grads[i][g][0] *= 1/float(n)
	return all_grads

def all_average_gradients2(local_grads, devices):
	# This version updates the gradients in-place
	n = len(local_grads)
	all_grads = [[[grad,val] for grad,val in grads] for grads in local_grads]
	for i, device in zip(xrange(n), devices):
		with tf.device(device):
			# Attempts to do a ring-like all-reduce
			for j in xrange(1, n):
				for g in xrange(len(all_grads[0])):
					with tf.control_dependencies([local_grads[(i+j)%n][g][0]]):  # TODO: Needed?
						all_grads[i][g][0] += local_grads[(i+j)%n][g][0]
			for g in xrange(len(local_grads[0])):
				all_grads[i][g][0] *= 1/float(n)
	return all_grads

def all_average_gradients3(local_grads, devices):
	# This version does the reduction on the first GPU
	n     = len(local_grads)
	ngrad = len(local_grads[0])
	all_grads = [[[grad,val] for grad,val in grads] for grads in local_grads]
	with tf.device(devices[0]):
		for j in xrange(1, n):
			for g in xrange(ngrad):
				with tf.control_dependencies([local_grads[j][g][0]]):
					all_grads[0][g][0] += local_grads[j][g][0]
		for g in xrange(ngrad):
			all_grads[0][g][0] *= 1/float(n)
	for j in xrange(1, n):
		with tf.device(devices[j]):
			for g in xrange(ngrad):
				with tf.control_dependencies([all_grads[0][g][0]]):
					all_grads[j][g][0] = tf.identity(all_grads[0][g][0])
	return all_grads

def average_gradients(tower_gradvars):
	# tower_gradvars contains (fastest->slowest): [tower,variable,(grad,var)]
	if len(tower_gradvars) == 1:
		return tower_gradvars[0]
	avg_variable_gradvars = []
	for gradvars in zip(*tower_gradvars): # Loop over each variable
		avg_grad = gradvars[0][0] # First tower gradient
		for grad, _ in gradvars[1:]: # Remaining towers
			avg_grad += grad
		avg_grad *= 1./len(gradvars)
		shared_var = gradvars[0][1] # First tower variable
		avg_variable_gradvars.append( (avg_grad, shared_var) )
		#avg_variable_gradvars.append([(avg_grad,var) for _,var in gradvars])
	#return zip(*avg_variable_gradvars)
	return avg_variable_gradvars

def average_gradients_inception(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def loss_function(logits, labels):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels,
	                                                               name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	return loss

def loss_function_old(logits, labels):
	#with tf.device("/cpu:0"):
	#nclass = tf.shape(logits)[-1]
	nclass = logits[0].get_shape()[-1].value
	batch_size = tf.size(labels)
	labels = tf.expand_dims(labels, 1)
	indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
	concated = tf.concat(1, [indices, labels])
	onehot_labels = tf.sparse_to_dense(
		concated, tf.pack([batch_size, nclass]), 1.0, 0.0)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
	                                                        onehot_labels,
	                                                        name='xentropy')
	#loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	n = tf.size(cross_entropy)
	loss = tf.reduce_sum(cross_entropy, name='xentropy_mean') / tf.to_float(n)
	return loss

from abc import ABCMeta, abstractmethod

def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat(0, [ymin, xmin, ymax, xmax])

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']

def decode_jpeg(image_buffer, scope=None):#, dtype=tf.float32):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  #with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
  #with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
  with tf.name_scope(scope or 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 fancy_upscaling=False)

    #image = tf.Print(image, [tf.shape(image)], "Image shape: ")

    return image

def eval_image(image, height, width, bbox, thread_id, scope=None):
	with tf.name_scope(scope or 'eval_image'):
		if not thread_id:
			tf.image_summary('original_image',
			                 tf.expand_dims(image, 0))
		if FLAGS.resize_method == 'crop':
			# Note: This is much slower than crop_to_bounding_box
			#         It seems that the redundant pad step has huge overhead
			#distorted_image = tf.image.resize_image_with_crop_or_pad(image,
			#                                                         height, width)
			shape = tf.shape(image)
			y0 = (shape[0] - height) // 2
			x0 = (shape[1] - width)  // 2
			#distorted_image = tf.slice(image, [y0,x0,0], [height,width,3])
			distorted_image = tf.image.crop_to_bounding_box(image, y0, x0,
			                                                height, width)
		else:
			sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
				tf.shape(image),
				bounding_boxes=bbox,
				min_object_covered=0.1,
				aspect_ratio_range=[0.75, 1.33],
				area_range=[0.05, 1.0],
				max_attempts=100,
				use_image_if_no_bounding_boxes=True)
			bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
			# Crop the image to the specified bounding box.
			distorted_image = tf.slice(image, bbox_begin, bbox_size)
			resize_method = {
				'nearest':  tf.image.ResizeMethod.NEAREST_NEIGHBOR,
				'bilinear': tf.image.ResizeMethod.BILINEAR,
				'bicubic':  tf.image.ResizeMethod.BICUBIC,
				'area':     tf.image.ResizeMethod.AREA
			}[FLAGS.resize_method]
			# This resizing operation may distort the images because the aspect
			# ratio is not respected.
			if tensorflow_version() >= 11:
				distorted_image = tf.image.resize_images(distorted_image, [height, width],
				                                         resize_method, align_corners=False)
			else:
				distorted_image = tf.image.resize_images(distorted_image, height, width,
				                                         resize_method, align_corners=False)
		distorted_image.set_shape([height, width, 3])
		if not thread_id:
			tf.image_summary('cropped_resized_image',
			                 tf.expand_dims(distorted_image, 0))
		image = distorted_image
	return image

def distort_image(image, height, width, bbox, thread_id=0, scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  #with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
  #with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
  with tf.name_scope(scope or 'distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Display the bounding box in the first thread only.
    if not thread_id:
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox)
      tf.image_summary('image_with_bounding_boxes', image_with_box)

  # A large fraction of image datasets contain a human-annotated bounding
  # box delineating the region of the image containing the object of interest.
  # We choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an allowed
  # range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    if not thread_id:
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.image_summary('images_with_distorted_bounding_box',
                       image_with_distorted_box)

    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    resize_method = thread_id % 4
    if tensorflow_version() >= 11:
      distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                                 resize_method, align_corners=False)
    else:
      distorted_image = tf.image.resize_images(distorted_image, height, width,
                                                 resize_method, align_corners=False)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if not thread_id:
      tf.image_summary('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, thread_id)

    # Note: This ensures the scaling matches the output of eval_image
    distorted_image *= 255

    if not thread_id:
      tf.image_summary('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    return distorted_image

def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
  #with tf.op_scope([image], scope, 'distort_color'):
  #with tf.name_scope(scope, 'distort_color', [image]):
  with tf.name_scope(scope or 'distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

class ImagePreprocessor(object):
	def __init__(self, height, width, batch_size,
	             dtype=tf.float32,
	             train=True,
	             distortions=None,
	             num_preprocess_threads=None,
	             num_readers=None,
	             input_queue_memory_factor=None):
		self.height = height
		self.width  = width
		self.batch_size = batch_size
		self.dtype  = dtype
		self.train  = train
		if num_preprocess_threads is None:
			num_preprocess_threads = FLAGS.num_preprocess_threads
		if num_readers is None:
			num_readers = FLAGS.num_readers
		if input_queue_memory_factor is None:
			input_queue_memory_factor = FLAGS.input_queue_memory_factor
		if distortions is None:
			distortions = FLAGS.distortions
		if distortions:
			# Round up to a multiple of 4 due to distortions implementation
			num_preprocess_threads = ((num_preprocess_threads-1)//4+1)*4
		self.num_preprocess_threads = num_preprocess_threads
		self.num_readers = num_readers
		self.input_queue_memory_factor = input_queue_memory_factor
		self.distortions = distortions
	
	def preprocess(self, image_buffer, bbox, thread_id):
		# Note: Width and height of returned image is known only at runtime
		image = tf.image.decode_jpeg(image_buffer, channels=3)
		if self.train and self.distortions:
			image = distort_image(image, self.height, self.width, bbox, thread_id)
		else:
			image = eval_image(image, self.height, self.width, bbox, thread_id)
		# Note: image is now float32 [height,width,3] with range [0, 255]
		return image
	
	def minibatch(self, dataset, subset):
		with tf.name_scope('batch_processing'):
			data_files = dataset.data_files(subset)
			shuffle  = self.train
			capacity = 16 if self.train else 1
			#print data_files
			filename_queue = tf.train.string_input_producer(data_files,
			                                                shuffle=shuffle,
			                                                capacity=capacity,
			                                                seed=1)
			# Approximate number of examples per shard.
			examples_per_shard = 1024
			# Size the random shuffle queue to balance between good global
			# mixing (more examples) and memory use (fewer examples).
			# 1 image uses 299*299*3*4 bytes = 1MB
			# The default input_queue_memory_factor is 16 implying a shuffling queue
			# size: examples_per_shard * 16 * 1MB = 17.6GB
			min_queue_examples = examples_per_shard * self.input_queue_memory_factor
			if self.train:
				examples_queue = tf.RandomShuffleQueue(
					capacity=min_queue_examples + 3 * self.batch_size,
					min_after_dequeue=min_queue_examples,
					dtypes=[tf.string],
					seed=2)
			else:
				examples_queue = tf.FIFOQueue(
					capacity=examples_per_shard + 3 * self.batch_size,
					dtypes=[tf.string])
			if self.num_readers == 0: # Special case to use one reader per preproc thread
				_, example_serialized = dataset.reader().read(filename_queue)
			else:
				enqueue_ops = []
				for _ in xrange(self.num_readers):
					_2, value = dataset.reader().read(filename_queue)
					enqueue_ops.append(examples_queue.enqueue([value]))
				tf.train.queue_runner.add_queue_runner(
				  tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops,
				                                    queue_closed_exception_types=(tf.errors.OutOfRangeError,
				                                                                  tf.errors.InvalidArgumentError)))
				example_serialized = examples_queue.dequeue()
			images_and_labels = []
			for thread_id in xrange(self.num_preprocess_threads):
				# Parse a serialized Example proto to extract the image and metadata.
				image_buffer, label_index, bbox, _ = parse_example_proto(example_serialized)
				image = self.preprocess(image_buffer, bbox, thread_id)
				images_and_labels.append([image, label_index])
			images, label_index_batch = tf.train.batch_join(
				images_and_labels,
				batch_size=self.batch_size,
				capacity=2 * self.num_preprocess_threads * self.batch_size)#,
				#dynamic_pad=True) # HACK TESTING dynamic_pad=True
			images = tf.cast(images, self.dtype)
			depth = 3
			images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, depth])
			label_index_batch = tf.reshape(label_index_batch, [self.batch_size])
			# Display the training images in the visualizer.
			tf.image_summary('images', images)
			
			#image_shape = (self.batch_size, self.height, self.width, depth)
			#image_buf = tf.Variable(tf.zeros(input_shape, input_data_type, trainable=False))
			
			return images, label_index_batch

class Dataset(object):
	__metaclass__ = ABCMeta
	def __init__(self, name, data_dir=None):
		self.name = name
		if data_dir is None:
			data_dir = FLAGS.data_dir
		self.data_dir = data_dir
	def data_files(self, subset):
		tf_record_pattern = os.path.join(self.data_dir, '%s-*' % subset)
		data_files = tf.gfile.Glob(tf_record_pattern)
		if not data_files:
			raise RuntimeError('No files found for %s dataset at %s' %
			                   (subset, self.data_dir))
		return data_files
	def reader(self):
		return tf.TFRecordReader()
	@abstractmethod
	def num_classes(self):
		pass
	@abstractmethod
	def num_examples_per_epoch(self, subset):
		pass
	def __str__(self):
		return self.name

class FlowersData(Dataset):
	def __init__(self, data_dir=None):
		super(FlowersData, self).__init__('Flowers', data_dir)
	def num_classes(self):
		return 5
	def num_examples_per_epoch(self, subset):
		if   subset == 'train':      return 3170
		elif subset == 'validation': return 500
		else: raise ValueError('Invalid data subset "%s"' % subset)

class ImagenetData(Dataset):
	def __init__(self, data_dir=None):
		super(ImagenetData, self).__init__('ImageNet', data_dir)
	def num_classes(self):
		return 1000
	def num_examples_per_epoch(self, subset):
		if   subset == 'train':      return 1281167
		elif subset == 'validation': return 50000
		else: raise ValueError('Invalid data subset "%s"' % subset)

class GPUPrefetcherOp(object):
	def __init__(self, parent):
		self.parent = parent
		self.op     = parent._acquire()
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		return self.parent._release(self.op)

class GPUPrefetcher(threading.Thread):
	def __init__(self, sess, device, input_op, dtype, nbuf=2):
		super(GPUPrefetcher, self).__init__()
		self.sess = sess
		self.input_op = input_op
		#with tf.device("/cpu:0"): # TODO: Doesn't work because can't override op-dependent device scopes! Is this a bug?
		self.empty_queue = tf.FIFOQueue(capacity=nbuf, dtypes=[tf.int32], shapes=[])
		self.full_queue  = tf.FIFOQueue(capacity=nbuf, dtypes=[tf.int32], shapes=[])
		self.bufnum = tf.placeholder(tf.int32)
		self.init_op = self.empty_queue.enqueue([self.bufnum])
		with tf.device(device):
			shape = input_op.get_shape()
			# TODO: This is just for a quick POC; it should be removed in favour
			#         of using a dependency on the op(s) that use the prefetch
			#         buffer directly.
			self.output_tmp = tf.Variable(tf.zeros(shape, dtype), trainable=False)
			self.nbuf = nbuf
			self.bufs = [tf.Variable(tf.zeros(shape, dtype), trainable=False)
			             for _ in xrange(nbuf)]
			self.put_ops = [self._put_op(b) for b in xrange(nbuf)]
			self.get_op = self._get_op()
			with tf.control_dependencies([self.get_op]):
				self.output = tf.identity(self.output_tmp)
		self.shutdown = threading.Event()
		
	def _get_buf_op(self, bufnum):
		cases = [(tf.equal(bufnum, b),
		          lambda: self.bufs[b])
		         for b in xrange(self.nbuf)]
		return tf.case(cases,
		               #exclusive=True,
		               default=lambda: self.bufs[0]) # Note: Should never hit
	def _put_op(self, bufnum):
		with tf.device("/cpu:0"):
			dequeue_op = self.empty_queue.dequeue()
		buf = self.bufs[bufnum]
		with tf.control_dependencies([dequeue_op]):
			buf_assign = buf.assign(self.input_op)
		with tf.control_dependencies([buf_assign]):
			with tf.device("/cpu:0"):
				buf_filled = self.full_queue.enqueue([bufnum])
		return buf_filled
	def _get_op(self):
		with tf.device("/cpu:0"):
			bufnum     = self.full_queue.dequeue()[0]
		buf        = self._get_buf_op(bufnum)
		# TODO: Remove use of this tmp (requires some refactoring)
		buf_assign = self.output_tmp.assign(buf)
		with tf.control_dependencies([buf_assign]):
			with tf.device("/cpu:0"):
				buf_cleared = self.empty_queue.enqueue([bufnum])
		return buf_cleared
		
	def run(self):
		for b in xrange(self.nbuf):
			self.sess.run(self.init_op,
			              feed_dict={self.bufnum: b})
		b = 0
		while not self.shutdown.is_set():
			#print "Empty size", self.sess.run(self.empty_queue.size())
			#print "Full size", self.sess.run(self.full_queue.size())
			#print "Putting", b
			self.sess.run(self.put_ops[b])
			b += 1
			b %= self.nbuf
		
	def shutdown(self):
		self.shutdown.set()
	
def test_cnn(model, batch_size, devices,
             dataset=None,
             param_server_device='/gpu:0',
             data_format='NCHW',
             num_batches=10,
             #share_variables=True,
             use_fp16=False,
             enable_mem_growth=False,
             perf_filename=None,
             trace_filename=None):
	
	tf.set_random_seed(1234)
	np.random.seed(4321)
	
	config = tf.ConfigProto()
	config.allow_soft_placement = False
	#config.gpu_options.allocator_type = 'BFC'
	# Allocate as needed rather than all at once
	config.gpu_options.allow_growth = enable_mem_growth
	#config.gpu_options.per_process_gpu_memory_fraction
	config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_fraction
	config.intra_op_parallelism_threads = FLAGS.num_intra_threads
	config.inter_op_parallelism_threads = FLAGS.num_inter_threads
	# TODO: Is this OK to use? Seems to provide a small ~3% speedup on AlexNet
	#config.graph_options.optimizer_options.do_function_inlining = True
	sess = tf.Session(config=config)
	# TODO: Look into these:
	# config.session_inter_op_thread_pool
	# config.use_per_session_threads
	
	nstep_burnin = 10
	perf_results = {}
	perf_results['tf_version']     = tensorflow_version()
	perf_results['model']          = model
	perf_results['mode']           = 'inference' if FLAGS.inference else 'training'
	perf_results['batch_size']     = batch_size
	perf_results['num_batches']    = num_batches
	perf_results['devices']        = devices
	perf_results['dataset']        = str(dataset) if dataset is not None else None
	perf_results['distortions']    = FLAGS.distortions
	perf_results['weak_scaling']   = FLAGS.weak_scaling
	perf_results['num_readers']    = FLAGS.num_readers
	perf_results['num_preproc_threads'] = FLAGS.num_preprocess_threads
	perf_results['num_intra_threads']   = FLAGS.num_intra_threads
	perf_results['num_inter_threads']   = FLAGS.num_inter_threads
	perf_results['memory_fraction']     = FLAGS.memory_fraction
	perf_results['param_server']   = param_server_device
	perf_results['data_format']    = data_format
	perf_results['storage_dtype']  = 'float16' if use_fp16 else 'float32'
	perf_results['compute_dtype']  = 'float32'
	perf_results['mem_growth']     = enable_mem_growth
	perf_results['trace_filename'] = trace_filename
	perf_results['gpu_prefetch']   = FLAGS.gpu_prefetch
	perf_results['learning_rate']  = FLAGS.learning_rate
	perf_results['momentum']       = FLAGS.momentum
	perf_results['weight_decay']   = FLAGS.weight_decay
	perf_results['gradient_clip']  = FLAGS.gradient_clip
	def dump_perf_results():
		if perf_filename is None:
			return
		def madstd(x):
			return 1.4826 * np.median(np.abs(x-np.median(x)))
		print "Dumping perf log to", perf_filename
		perf_results['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")
		times = perf_results['step_train_times'][nstep_burnin:]
		if len(times) > 0:
			perf_results['step_train_time_mean']   =   np.mean(times)
			perf_results['step_train_time_std']    =    np.std(times)
			perf_results['step_train_time_median'] = np.median(times)
			perf_results['step_train_time_madstd'] =    madstd(times)
			perf_results['step_train_time_min']    =    np.min(times)
			perf_results['step_train_time_max']    =    np.max(times)
		#print perf_results
		with open(perf_filename, 'a') as perf_file:
			perf_file.write(json.dumps(perf_results,
			                           #indent=4,
			                           separators=(',', ':'),
			                           sort_keys=True) + '\n')
		
	if model.startswith('vgg') or model == 'googlenet' \
	   or model.startswith('resnet') or model.startswith('test'):
		image_size = 224
	elif model == 'alexnet':
		image_size = 224+3
	elif model == 'overfeat':
		image_size = 231
	elif model.startswith('inception') or model.startswith('xception'):
		image_size = 299
	elif model.startswith('lenet'):
		image_size = 28
	elif model.startswith('cifar10'):
		image_size = 32
	else:
		raise KeyError("Invalid model name: "+model)
	data_type = tf.float16 if use_fp16 else tf.float32
	#input_data_type = data_type
	#input_data_type = tf.float32
	input_data_type = tf.uint8
	input_nchan = 3
	input_shape = [batch_size, image_size, image_size, input_nchan]
	#if share_variables:
	devices = [device_or_param_server(d, param_server_device) for d in devices]
	
	phase_train = tf.placeholder(tf.bool, name='phase_train')
	global_step = tf.get_variable('global_step', [],
	                              initializer=tf.constant_initializer(0),
	                              dtype=tf.int64,
	                              trainable=False)
	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
	                                           global_step,
	                                           FLAGS.lr_decay_steps,
	                                           FLAGS.lr_decay_factor,
	                                           staircase=FLAGS.lr_decay_staircase)
	
	with tf.device("/cpu:0"):
		if dataset is not None:
			#*preproc_train = ImagePreprocessor(image_size, image_size, batch_size, input_data_type, train=True)
			preproc_train = ImagePreprocessor(image_size, image_size, batch_size, input_data_type, train=True)
			images_train, labels_train = preproc_train.minibatch(dataset, subset="train")
			images = images_train
			labels = labels_train
			#nclass = dataset.num_classes()
			# Note: We force all datasets to 1000 to ensure even comparison
			#         This works because we use sparse_softmax_cross_entropy
			nclass = 1000
		else:
			nclass = 1000
			images = tf.truncated_normal(input_shape, dtype=tf.float32,
			                             mean=127, stddev=20,
			                             name="synthetic_images")
			images = tf.cast(images, input_data_type)
			labels = tf.random_uniform([batch_size], minval=1, maxval=nclass, dtype=tf.int32, name="synthetic_labels")
			# Note: This results in a H2D copy, but no computation
			# Note: This avoids recomputation of the random values, but still
			#         results in a H2D copy.
			images = tf.Variable(images, trainable=False)
			labels = tf.Variable(labels, trainable=False)
			#image_dtype = np.float16 if use_fp16 else np.float32
			#synthetic_images = np.random.normal(0, 1e-1, size=input_shape).astype(image_dtype)
			#synthetic_labels = np.ones([batch_size], dtype=np.int32)
			#images = tf.constant(synthetic_images)
			#labels = tf.constant(synthetic_labels)
		labels -= 1 # Change to 0-based (don't use background class like Inception does)
		images_splits = tf.split(0, len(devices), images)
		labels_splits = tf.split(0, len(devices), labels)
	
	print "Generating model"
	device_grads = []
	losses       = []
	prefetchers  = []
	all_predictions = []
	with tf.variable_scope("model"):
		for d, device in enumerate(devices):
			
			host_images = images_splits[d]
			if FLAGS.gpu_prefetch:
				print "Creating GPU prefetcher on GPU", d
				prefetcher = GPUPrefetcher(sess, device, host_images, input_data_type, nbuf=2)
				prefetchers.append(prefetcher)
				images = prefetcher.output
			else:
				images = host_images
				labels = labels_splits[d]
			
			# Note: We want variables on different devices to share the same
			#         variable scope, so we just use a name_scope here.
			with tf.device(device), tf.name_scope('tower_%i' % d) as scope:
				if dataset is None and not FLAGS.include_h2d_in_synthetic:
					# Minor hack to avoid H2D copy when using synthetic data
					images = tf.truncated_normal(images.get_shape(), dtype=tf.float32,
					                             mean=127, stddev=20,
					                             name="synthetic_images")
					images = tf.Variable(images, trainable=False, name='gpu_cached_images')
				
				if input_data_type != data_type:
					images = tf.cast(images, data_type)
				
				# Rescale and shift to [-1,1]
				images = images * (1./127) - 1
				
				network = ConvNetBuilder(images, input_nchan,
				                         phase_train, data_format, data_type)
				if   model == 'vgg11':        inference_vgg11(network)
				elif model == 'vgg13':        inference_vgg13(network)
				elif model == 'vgg16':        inference_vgg16(network)
				elif model == 'vgg19':        inference_vgg19(network)
				elif model == 'lenet':        inference_lenet5(network)
				elif model == 'cifar10':      inference_cifar10(network)
				elif model == 'googlenet':    inference_googlenet(network)
				elif model == 'overfeat':     inference_overfeat(network)
				elif model == 'alexnet':      inference_alexnet(network)
				elif model == 'inception3':   inference_inception_v3(network)
				elif model == 'inception4':   inference_inception_v4(network)
				elif model == 'resnet18':     inference_resnet_v1(network, (2,2,2,2), basic=True)
				elif model == 'resnet34':     inference_resnet_v1(network, (3,4,6,3), basic=True)
				elif model == 'resnet50':     inference_resnet_v1(network, (3,4,6,3))
				elif model == 'resnet101':    inference_resnet_v1(network, (3,4,23,3))
				elif model == 'resnet152':    inference_resnet_v1(network, (3,8,36,3))
				elif model == 'inception-resnet2': inference_inception_resnet_v2(network)
				elif model == 'xception':     inference_xception(network)
				else: raise KeyError("Invalid model name '%s'" % model)
				# Add the final fully-connected class layer
				logits = network.affine(nclass, activation='linear')
				logits = tf.cast(logits, tf.float32)
				loss = loss_function(logits, labels)
				predictions = tf.nn.softmax(logits, name='predictions')
				all_predictions.append(predictions)
				
				weight_decay = FLAGS.weight_decay
				if weight_decay is not None and weight_decay != 0.:
					l2_loss = tf.add_n([tf.nn.l2_loss(v)
					                    for v in tf.trainable_variables()])
					l2_loss = tf.cast(l2_loss, tf.float32)
					loss += (weight_decay * l2_loss) * (1./len(devices))
				
				losses.append(loss)
				params = tf.trainable_variables()
				#device_grads.append( opt.compute_gradients(loss, var_list=params) )
				aggmeth = tf.AggregationMethod.DEFAULT
				#aggmeth = tf.AggregationMethod.ADD_N
				#aggmeth = tf.AggregationMethod.EXPERIMENTAL_TREE
				#aggmeth = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
				grads = tf.gradients(loss, params, aggregation_method=aggmeth)
				gradvars = zip(grads, params)
				device_grads.append(gradvars)
				tf.get_variable_scope().reuse_variables()
	#all_grads = all_average_gradients(device_grads, devices)
	#all_grads = all_average_gradients2(device_grads, devices)
	#all_grads = all_average_gradients3(device_grads, devices)
	all_predictions = tf.concat(0, all_predictions)
	
	with tf.device(param_server_device):
		total_loss = tf.reduce_mean(losses)
		# Note: This cannot be used inside a variable scope with reuse=True
		#         Had to wrap the whole model above in a tf.variable_scope to
		#           avoid this.
		averager = tf.train.ExponentialMovingAverage(0.90, name='avg')
		avg_op = averager.apply([total_loss])
		total_loss_avg = averager.average(total_loss)
		# Note: This must be done _after_ the averager.average() call
		#         because it changes total_loss into a new object.
		with tf.control_dependencies([avg_op]):
			total_loss     = tf.identity(total_loss)
			total_loss_avg = tf.identity(total_loss_avg)
		
		#all_grads = all_average_gradients4(device_grads)
		#avg_grads = all_average_gradients4(device_grads)
		#avg_grads = average_gradients(device_grads)
		avg_grads = average_gradients_inception(device_grads)
		gradient_clip = FLAGS.gradient_clip
		#learning_rate = FLAGS.learning_rate
		momentum      = FLAGS.momentum
		if gradient_clip is not None:
			clipped_grads = [(tf.clip_by_value(grad, -gradient_clip, +gradient_clip), var) for grad,var in avg_grads]
		else:
			clipped_grads = avg_grads
		opt = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=FLAGS.nesterov)
		#opt = tf.train.RMSPropOptimizer(learning_rate)
		#opt = tf.train.AdamOptimizer()
		train_op = opt.apply_gradients(clipped_grads, global_step=global_step)
		# Ensure in-place update ops are executed too (e.g., batch norm)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		if update_ops:
			#updates = tf.group(*update_ops)
			#train_op = control_flow_ops.with_dependencies([updates], train_op)
			train_op = tf.group(train_op, *update_ops)
	
	with tf.device('/cpu:0'):
		for grad, var in avg_grads:
			if grad is not None:
				if tensorflow_version() >= 12:
					tf.summary.histogram(var.op.name + '/gradients', grad)
				else:
					tf.histogram_summary(var.op.name + '/gradients', grad)
		for var in tf.trainable_variables():
			if tensorflow_version() >= 12:
				tf.summary.histogram(var.op.name, var)
			else:
				tf.histogram_summary(var.op.name, var)
	
	with tf.device('/cpu:0'):
		if tensorflow_version() >= 11:
			tf.summary.scalar('total loss raw', total_loss)
			tf.summary.scalar('total loss avg', total_loss_avg)
			tf.summary.scalar('learning_rate', learning_rate)
		else:
			tf.scalar_summary('total loss raw', total_loss)
			tf.scalar_summary('total loss avg', total_loss_avg)
			tf.scalar_summary('learning_rate', learning_rate)
	
	if FLAGS.summaries_dir is not None:
		all_summaries = tf.merge_all_summaries()
		print "Creating SummaryWriter"
		#summaries_dir = os.path.join(FLAGS.summaries_dir,
		#                             "%s-%s" % (FLAGS.model,
		#                                        time.strftime("%Y%m%d-%H%M%S")))
		summaries_dir = FLAGS.summaries_dir
		summary_writer = tf.train.SummaryWriter(summaries_dir, sess.graph)
	
	if tensorflow_version() >= 12:
		init = tf.global_variables_initializer()
	else:
		init = tf.initialize_all_variables()
	
	if FLAGS.graph_file is not None:
		path, filename = os.path.split(FLAGS.graph_file)
		as_text = filename.endswith('txt')
		print "Writing GraphDef as %s to %s" % ('text' if as_text else 'binary', FLAGS.graph_file)
		tf.train.write_graph(sess.graph_def, path, filename, as_text)
	
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	
	print "Initializing all variables"
	# Note: This must be done before starting the queue runners to avoid errors
	sess.run(init)
	
	print "Starting queue runners"
	coordinator = tf.train.Coordinator()
	#coordinator = tf.train.Coordinator((tf.errors.OutOfRangeError,
	#                                    tf.errors.InvalidArgumentError))
	queue_threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
	
	for p in prefetchers:
		p.daemon = True # TODO: Try to avoid needing this
		p.start()
	
	if FLAGS.checkpoint_dir is not None:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		checkpoint_file = os.path.join(FLAGS.checkpoint_dir, "checkpoint")
		if ckpt and ckpt.model_checkpoint_path:
			# Note: ckpt.model_checkpoint_path is the "most-recent model checkpoint"
			saver.restore(sess, ckpt.model_checkpoint_path)
			print "Checkpoint loaded from %s" % ckpt.model_checkpoint_path
		else:
			if not os.path.exists(FLAGS.checkpoint_dir):
				os.mkdir(FLAGS.checkpoint_dir)
			save_path = saver.save(sess, checkpoint_file, global_step=0)
			print "Checkpoint saved to %s" % save_path
	
	last_summary_time = time.time()
	
	print "Step\tImg/sec\tLoss\tEff. accuracy"
	perf_results['step_train_times'] = []
	perf_results['step_losses'] = []
	nstep = num_batches
	oom = False
	step0 = int(sess.run(global_step))
	step = 0
	for step in xrange(step0, nstep):
		if step == FLAGS.nvprof_start:
			cudaProfilerStart()
		if trace_filename is not None and step == 10:
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
		else:
			run_options  = None
			run_metadata = None
		start_time = time.time()
		try:
			if not FLAGS.inference:
				_, lossval = sess.run([train_op, total_loss_avg], feed_dict={phase_train: True},
				                      options=run_options, run_metadata=run_metadata)
			else:
				predictions = sess.run([all_predictions], feed_dict={phase_train: False},
				                       options=run_options, run_metadata=run_metadata)
				lossval = 0.
			train_time = time.time() - start_time
		#except tf.python.framework.errors.ResourceExhaustedError:
		except tf.python.errors.ResourceExhaustedError:
			train_time = -1.
			lossval    = 0.
			oom = True
		except KeyboardInterrupt:
			print "Keyboard interrupt"
			break
		if step == 0 or (step+1) % FLAGS.display_every == 0:
			#print "%i\t%.1f\t%.3f" % (step+1,
			#                          batch_size/train_time,
			#                          np.exp(lossval))
			print "%i\t%.1f\t%.3f\t%.3f %%" % (step+1,
			                                batch_size/train_time,
			                                #np.exp(lossval),
			                                lossval,
			                                (1./np.exp(lossval))*100)
		if trace_filename is not None and step == 10:
			print "Dumping trace to", trace_filename
			trace = timeline.Timeline(step_stats=run_metadata.step_stats)
			with open(trace_filename, 'w') as trace_file:
				trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
		perf_results['step_train_times'].append(train_time)
		perf_results['step_losses'].append(float(lossval))
		#if step == nstep_burnin+10 or step % 100 == 0:
		#	dump_perf_results()
		if (FLAGS.checkpoint_dir is not None and
		    (step+1) % FLAGS.checkpoint_steps == 0):
			save_path = saver.save(sess, checkpoint_file,
			                       global_step=step+1)
			print "Checkpoint saved to %s" % save_path
		#if FLAGS.summaries_dir is not None and ((step+1) % 100 == 0 or (step+1) == nstep):
		if (FLAGS.summaries_dir is not None and
		    (time.time() - last_summary_time >= FLAGS.summaries_interval or
		     step == 0 or
		     (step+1) == nstep)):
			last_summary_time += FLAGS.summaries_interval
			summary = sess.run(all_summaries, feed_dict={phase_train: (not FLAGS.inference)})
			summary_writer.add_summary(summary, step)
			print "Summaries saved to %s" % FLAGS.summaries_dir
		if step+1 == FLAGS.nvprof_stop:
			cudaProfilerStop()
		if oom:
			break
	nstep = step + 1
	if nstep > nstep_burnin:
		times = np.array(perf_results['step_train_times'][nstep_burnin:])
		speeds     = batch_size / times
		speed_mean = np.mean(speeds)
		if nstep - nstep_burnin >= 2:
			speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(len(speeds)))
		else:
			speed_uncertainty = float('nan')
		speed_madstd = 1.4826*np.median(np.abs(speeds - np.median(speeds)))
		speed_jitter = speed_madstd
		print '-'*64
		print "Images/sec: %.1f +/- %.1f (jitter = %.1f)" % (speed_mean, speed_uncertainty, speed_jitter)
		print '-'*64
	else:
		print "No results, did not get past burn-in phase (%i steps)" % nstep_burnin
	dump_perf_results()
	coordinator.request_stop()
	coordinator.join(queue_threads, stop_grace_period_secs=5.)
	if FLAGS.summaries_dir is not None:
		summary_writer.close()
	sess.close()

def device_or_param_server(device, ps):
	return lambda op: ps if op.type == 'Variable' else device

def add_bool_argument(cmdline, shortname, longname=None, default=False, help=None):
	# Based on http://stackoverflow.com/a/31347222
	if longname is None:
		shortname, longname = None, shortname
	elif default == True:
		raise ValueError("""Boolean arguments that are True by default should not have short names.""")
	name = longname[2:]
	feature_parser = cmdline.add_mutually_exclusive_group(required=False)
	if shortname is not None:
		feature_parser.add_argument(shortname, '--'+name, dest=name, action='store_true', help=help, default=default)
	else:
		feature_parser.add_argument(           '--'+name, dest=name, action='store_true', help=help, default=default)
	feature_parser.add_argument('--no'+name, dest=name, action='store_false')
	#print name, default
	#cmdline.set_defaults(name=default) # Doesn't seem to work?
	return cmdline

def edits1(word):
	import string
	"All edits that are one edit away from `word`."
	chars      = string.printable
	splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
	deletes    = [L + R[1:]               for L, R in splits if R]
	transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
	replaces   = [L + c + R[1:]           for L, R in splits if R for c in chars]
	inserts    = [L + c + R               for L, R in splits for c in chars]
	return (deletes + transposes + replaces + inserts)
def edits2(word):
	return [e2 for e1 in edits1(word) for e2 in edits1(e1)]

def main():
	cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	cmdline.add_argument('-m', '--model', default='googlenet',
	                     help="""Name of model to run:
	                     lenet, cifar10, alexnet, overfeat, googlenet,
	                     vgg[11,13,16,19], inception[3,4],
	                     resnet[18,34,50,101,152], inception-resnet2 or
	                     xception.""")
	add_bool_argument(cmdline, '--inference', default=False,
	                  help="""Benchmark inference performance instead of
	                  training.""")
	cmdline.add_argument('-b', '--batch_size', default=64, type=int,
	                     help="""Size of each minibatch""")
	cmdline.add_argument('--num_batches', default=40, type=int,
	                     help="""Number of batches to run.""")
	cmdline.add_argument('-g', '--num_gpus', default=1, type=int,
	                     help="""Number of GPUs to run on.""")
	cmdline.add_argument('--num_devices', default=1, type=int, dest='num_gpus',
	                     help="""Number of devices to run on. EQUIVALENT TO --num_gpus.""")
	add_bool_argument(cmdline, '--weak_scaling',
	                  help="""Interpret batch_size as *per GPU*
	                  rather than total.""")
	cmdline.add_argument('--display_every', default=1, type=int,
	                     help="""How often (in iterations) to print out
	                     running information.""")
	cmdline.add_argument('--shmoo', action='store_true',
	                     help="""Run a big shmoo over many
	                     parameter combinations.""")
	cmdline.add_argument('--data_dir', default=None,
	                     help="""Path to dataset in TFRecord format
	                     (aka Example protobufs). If not specified,
	                     synthetic data will be used.
	                     See also: --include_h2d_in_synthetic.""")
	cmdline.add_argument('--data_name', default=None,
	                     help="""Name of dataset: imagenet or flowers.
	                     If not specified, it is automatically guessed
	                     based on --data_dir.""")
	cmdline.add_argument('--resize_method', default='bilinear',
	                     help="""Method for resizing input images:
	                     crop,nearest,bilinear,trilinear or area.
	                     The 'crop' mode requires source images to be at least
	                     as large as the network input size,
	                     while the other modes support any sizes and apply
	                     random bbox distortions
	                     before resizing (even with --nodistortions).""")
	add_bool_argument(cmdline, '--distortions', default=False,
	                     help="""Enable/disable distortions during
	                     image preprocessing. These include bbox and color
	                     distortions.""")
	cmdline.add_argument('--parameter_server', default='gpu',
	                     help="""Device to use as parameter server:
	                     cpu or gpu.""")
	cmdline.add_argument('--device', default='gpu',
	                     help="""Device to use for computation: cpu or gpu""")
	add_bool_argument(cmdline, '--cpu', default=False,
	                  help="""Shortcut for --device=cpu --parameter_server=cpu
	                  --data_format=NHWC""")
	add_bool_argument(cmdline, '--include_h2d_in_synthetic', default=False,
	                  help="""Include host to device memcopy when using
	                  synthetic data.""")
	cmdline.add_argument('--data_format', default='NCHW',
	                     help="""Data layout to use: NHWC (TF native)
	                     or NCHW (cuDNN native).""")
	add_bool_argument(cmdline, '--use_fp16',
	                  help="""Use fp16 (half) instead of fp32 (float) for
	                  storage (compute is always fp32).""")
	add_bool_argument(cmdline, '--memory_growth',
	                  help="""Enable on-demand memory growth.""")
	cmdline.add_argument('--memory_fraction', default=0., type=float,
	                     help="""Fraction of GPU memory to use.
	                     Set to 0.0 to allocate max amount (default).""")
	cmdline.add_argument('--num_preprocess_threads', default=4, type=int,
	                     help="""Number of preprocessing threads *per GPU*.
	                     Must be a multiple of 4 when distortions are enabled.""")
	cmdline.add_argument('--num_readers', default=1, type=int,
	                     help="""Number of parallel readers during training.
	                     Setting this to 0 is a special case that causes each
	                     preprocessing thread to do its own reading.""")
	cmdline.add_argument('--num_intra_threads', default=0, type=int,
	                     help="""Number of threads to use for intra-op
	                     parallelism. If set to 0, the system will pick
	                     an appropriate number.""")
	cmdline.add_argument('--num_inter_threads', default=0, type=int,
	                     help="""Number of threads to use for inter-op
	                     parallelism. If set to 0, the system will pick
	                     an appropriate number.""")
	cmdline.add_argument('--input_queue_memory_factor', default=16, type=int,
	                     help="""Size of the queue of preprocessed images.
	                     Default is ideal but try smaller values, e.g.
	                     4, 2 or 1, if host memory is constrained.""")
	cmdline.add_argument('--perf_file', default=None,
	                     help="""Write perf log (metadata, training times and
	                     loss values) to this file.""")
	cmdline.add_argument('--trace_file', default=None,
	                     help="""Enable TensorFlow tracing and write trace to
	                     this file.""")
	cmdline.add_argument('--graph_file', default=None,
	                     help="""Write the model's graph definition to this
	                     file. Defaults to binary format unless filename ends
	                     in 'txt'.""")
	cmdline.add_argument('--checkpoint_dir', default=None,
	                     help="""Load/save training checkpoints in this
	                     directory.""")
	cmdline.add_argument('--checkpoint_steps', default=250, type=int,
	                     help="""Interval at which to write checkpoints.""")
	cmdline.add_argument('--summaries_dir', default=None,
	                     help="""Write TensorBoard summary to this
	                     directory.""")
	cmdline.add_argument('--summaries_interval', default=120, type=float,
	                     help="""Time interval (secs) at which to dump
	                     summaries.""")
	cmdline.add_argument('--learning_rate', default=0.003, type=float,
	                     help="""Learning rate for training.""")
	cmdline.add_argument('--momentum', default=0.9, type=float,
	                     help="""Momentum for training.""")
	add_bool_argument(cmdline, '--nesterov', default=True,
	                  help="""Use Nesterov momentum instead of regular.""")
	cmdline.add_argument('--lr_decay_factor', default=0.1, type=float,
	                     help="""Learning rate decay factor""")
	cmdline.add_argument('--lr_decay_steps', default=10000, type=int,
	                     help="""No. steps after which to decay
	                     learning rate.""")
	add_bool_argument(cmdline, '--lr_decay_staircase', default=True,
	                  help="""Whether to decay learning rate in a staircase
	                  fashion.""")
	cmdline.add_argument('--gradient_clip', default=None, type=float,
	                     help="""Gradient clipping magnitude.
	                     Disabled by default.""")
	cmdline.add_argument('--weight_decay', default=1e-5, type=float,
	                     help="""Weight decay factor for training.""")
	cmdline.add_argument('--nvprof_start', default=-1, type=int,
	                     help="""Iteration at which to start CUDA profiling.
	                     A value of -1 means program start.""")
	cmdline.add_argument('--nvprof_stop', default=-1, type=int,
	                     help="""Iteration at which to stop CUDA profiling.
	                     A value of -1 means program end.""")
	add_bool_argument(cmdline, '--gpu_prefetch', default=False,
	                  help="""*EXPERIMENTAL* Enable/disable prefetching over PCIe.""")
	
	global FLAGS
	FLAGS, unknown_args = cmdline.parse_known_args()
	# Check for invalid arguments and look for a correction to suggest
	good_args = ['--'+arg for arg in vars(FLAGS).keys()]
	for bad_arg in unknown_args:
		arg_edits1 = set(edits1(bad_arg))
		matches = [arg for arg in good_args if arg in arg_edits1]
		if not matches and len(bad_arg) <= 30:
			print "Bad argument '%s'" % bad_arg
			print "Searching for suggestions... (hit [Ctrl-C] to cancel)"
			try:
				arg_edits2 = set(edits2(bad_arg))
			except KeyboardInterrupt:
				sys.exit(-1)
			matches = [arg for arg in good_args if arg in arg_edits2]
		msg = "Unknown command line arg: %s" % bad_arg
		if matches:
			msg += "\nDid you mean '%s'?" % matches[0]
		raise ValueError(msg)
	
	if FLAGS.cpu:
		FLAGS.device           = 'cpu'
		FLAGS.parameter_server = 'cpu'
		FLAGS.data_format      = 'NHWC'
	
	model       = FLAGS.model
	batch_size  = FLAGS.batch_size
	devices     = ['/%s:%i'%(FLAGS.device,i) for i in xrange(FLAGS.num_gpus)]
	ps_device   = '/%s:0' % FLAGS.parameter_server
	#share_vars  = FLAGS.share_variables
	mem_growth  = FLAGS.memory_growth
	perf_filename  = FLAGS.perf_file
	trace_filename = FLAGS.trace_file
	
	FLAGS.num_preprocess_threads *= FLAGS.num_gpus
	if FLAGS.weak_scaling:
		batch_size *= FLAGS.num_gpus
	
	dataset = None
	if FLAGS.data_dir is not None:
		if FLAGS.data_name is None:
			if   "imagenet" in FLAGS.data_dir: FLAGS.data_name = "imagenet"
			elif "flowers"  in FLAGS.data_dir: FLAGS.data_name = "flowers"
			else: raise ValueError("Could not identify name of dataset. Please specify with --data_name option.")
		if   FLAGS.data_name == "imagenet": dataset = ImagenetData(FLAGS.data_dir)
		elif FLAGS.data_name == "flowers":  dataset =  FlowersData(FLAGS.data_dir)
		else: raise ValueError("Unknown dataset. Must be one of imagenet or flowers.")
	
	if FLAGS.gpu_prefetch:
		print "*** WARNING: GPU prefetching is highly experimental! ***"
	
	tfversion = tensorflow_version_tuple()
	print "TensorFlow:  %i.%i.%s" % tfversion
	
	data_format = FLAGS.data_format
	num_batches = FLAGS.num_batches
	use_fp16    = FLAGS.use_fp16
	if not FLAGS.shmoo:
		print "Model:      ", model
		print "Mode:       ", 'inference' if FLAGS.inference else 'training'
		print "Batch size: ", batch_size, 'global'
		print "            ", batch_size/len(devices), 'per device'
		print "Devices:    ", devices
		print "Data format:", data_format
		print "Data type:  ", 'fp16' if use_fp16 else 'fp32'
		
		with tf.Graph().as_default(): # Ensure graph is freed
			test_cnn(model, batch_size, devices, dataset, ps_device,
			         data_format, num_batches,# share_variables=share_vars,
			         use_fp16=use_fp16, enable_mem_growth=mem_growth,
			         perf_filename=perf_filename,
			         trace_filename=trace_filename)
	else: # shmoo
		print "Running shmoo"
		for use_fp16 in [False, True]:
			for model in ['alexnet', 'vgg19', 'googlenet', 'overfeat', 'inception3']:
				for ps_device in ['/cpu:0', '/gpu:0']:
					for ngpu in [1, 2, 4, 8]:
						if ngpu > len(devices):
							continue
						shmoo_devices = devices[:ngpu]
						for batch_size in [64, 128, 256, 512]:
							if batch_size > 64 and model in set(['inception3', 'vgg19']):
								# Note: A 12 GB card can fit up to batch_size=112 for inception3
								continue
							for distortions in [False, True]:
								FLAGS.distortions = distortions
								for shmoo_dataset in set([None, dataset]):
									with tf.Graph().as_default(): # Ensure graph is freed
										#try:
											test_cnn(model, batch_size, shmoo_devices, shmoo_dataset, ps_device,
											         data_format, num_batches,
											         use_fp16=use_fp16, enable_mem_growth=mem_growth,
											         perf_filename=perf_filename,
											         trace_filename=None)
										#except tf.python.framework.errors.ResourceExhaustedError:
										#	pass
	
if __name__ == '__main__':
   main()

