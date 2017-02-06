#!/usr/bin/env python

#  Copyright (c) 2016, NVIDIA Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
 This is a mashup and extension of Soumith's convnet benchmark scripts,
   TensorFlow's Inception v3 training scripts, the lm_benchmark.py script,
   and TFSlim's ResNet implementation.
 It is intended for use as a benchmarking tool that provides complete
   control over how TF is used (as opposed to say relying on Keras or
   TFSlim, which may not be GPU-optimal).

**TODO: Avoid H2D copy when using synthetic data
          We want this mode to represent GPU-only perf

---------
Changelog
---------
v0.7
Fixed --distortions not defaulting to True
Added num_intra_threads option with default of 1 (slightly faster than default of 0 => TF decides)
Added printing of final performance statistics

"""

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python import control_flow_ops
from tensorflow.python.client import timeline
import numpy as np
import time
from collections import defaultdict
import os
import json
import argparse
import sys
from ctypes import cdll

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
			init_factor = 2. if activation == 'relu' else 1.
			kernel = tf.get_variable('weights', [kH, kW, nIn, nOut], self.data_type,
			                         tf.random_normal_initializer(stddev=np.sqrt(init_factor/(nIn*kH*kW))))
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
				biased = tf.reshape(tf.nn.bias_add(conv, biases, data_format=self.data_format), conv.get_shape())
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
	def reshape(self, shape, input_layer=None):
		if input_layer is None:
			input_layer = self.top_layer
		self.top_layer = tf.reshape(input_layer, shape)
		self.top_size  = shape[-1] # HACK This may not always work
		return self.top_layer
	def affine(self, nOut, input_layer=None, nIn=None, activation='relu'):
		if input_layer is None:
			input_layer = self.top_layer
		if nIn is None:
			nIn = self.top_size
		name = 'affine' + str(self.counts['affine'])
		self.counts['affine'] += 1
		#with tf.name_scope(name) as scope, tf.variable_scope(name) as varscope:
		with tf.variable_scope(name) as scope:
			init_factor = 2. if activation == 'relu' else 1.
			kernel = tf.get_variable('weights', [nIn, nOut], self.data_type,
			                         tf.random_normal_initializer(stddev=np.sqrt(init_factor/(nIn))))
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
	def resnet_bottleneck_v1(self, depth, depth_bottleneck, stride, input_layer=None, inSize=None):
		if input_layer is None:
			input_layer = self.top_layer
		if inSize is None:
			inSize = self.top_size
		name = 'resnet_v1' + str(self.counts['resnet_v1'])
		self.counts['resnet_v1'] += 1
		#with tf.name_scope(name) as scope:
		with tf.variable_scope(name) as scope:
			if depth == inSize:
				if stride == 1:
					shortcut = input_layer
				else:
					shortcut = self.mpool(1, 1, stride, stride, input_layer=input_layer, nIn=inSize)
			else:
				shortcut = self.conv(depth, 1, 1, stride, stride, activation=None, input_layer=input_layer, nIn=inSize)
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
		#with tf.name_scope(name) as scope:
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
	def residual(self, nout, net, scale=1.0):
		inlayer = self.top_layer
		net(self)
		self.conv(nout, 1, 1, activation=None)
		self.top_layer = tf.nn.relu(inlayer + scale*self.top_layer)
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
		#with tf.name_scope(name) as scope:
		with tf.variable_scope(name) as scope:
			keep_prob_tensor = tf.constant(keep_prob, dtype=self.data_type)
			one_tensor       = tf.constant(1.0,       dtype=self.data_type)
			keep_prob_op = control_flow_ops.cond(self.phase_train,
			                                     lambda: keep_prob_tensor,
			                                     lambda: one_tensor)
			dropout = tf.nn.dropout(input_layer, keep_prob_op)
			self.top_layer = dropout
			return dropout
	def batch_norm(self, input_layer=None, **kwargs):
		if input_layer is None:
			input_layer = self.top_layer
		else:
			self.top_size = None
		name = 'batchnorm' + str(self.counts['batchnorm'])
		self.counts['batchnorm'] += 1
		#with tf.name_scope(name) as scope:
		with tf.variable_scope(name) as scope:
			bn = tf.contrib.layers.batch_norm(input_layer,
			                                  is_training=self.phase_train,
			                                  scope=scope,
			                                  **kwargs)
		self.top_layer = bn
		return bn

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

def inference_vgg(cnn):
	cnn.conv (64, 3, 3)
	cnn.mpool(2, 2)
	cnn.conv (128, 3, 3)
	cnn.mpool(2, 2)
	cnn.conv (256, 3, 3)
	cnn.conv (256, 3, 3)
	cnn.mpool(2, 2)
	cnn.conv (512, 3, 3)
	cnn.conv (512, 3, 3)
	cnn.mpool(2, 2)
	cnn.conv (512, 3, 3)
	cnn.conv (512, 3, 3)
	cnn.mpool(2, 2)
	cnn.reshape([-1, 512 * 7 * 7])
	cnn.affine(4096)
	cnn.affine(4096)
	return cnn

def inference_resnet_v1(cnn, layer_counts):
	cnn.use_batch_norm = True
	cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}
	cnn.conv (64, 7, 7, 2, 2, mode='SAME_RESNET')
	cnn.mpool(3, 3, 2, 2)
	for _ in xrange(layer_counts[0]):
		cnn.resnet_bottleneck_v1(256, 64, 1)
	cnn.resnet_bottleneck_v1(256, 64, 2)
	for _ in xrange(layer_counts[1]):
		cnn.resnet_bottleneck_v1(512, 128, 1)
	cnn.resnet_bottleneck_v1(512, 128, 2)
	for _ in xrange(layer_counts[2]):
		cnn.resnet_bottleneck_v1(1024, 256, 1)
	cnn.resnet_bottleneck_v1(1024, 256, 2)
	for _ in xrange(layer_counts[3]):
		cnn.resnet_bottleneck_v1(2048, 512, 1)
	cnn.spatial_mean()
	return cnn

def inference_googlenet(cnn):
	def inception_v1(cnn, k, l, m, n, p, q):
		cols = [[('conv', k, 1, 1)],
		        [('conv', l, 1, 1), ('conv', m, 3, 3)],
		        [('conv', n, 1, 1), ('conv', p, 5, 5)],
		        [('mpool', 3, 3, 1, 1, 'SAME'), ('conv', q, 1, 1)]]
		return cnn.inception_module('incept_v1', cols)
	#cnn.use_batch_norm = True # HACK TESTING
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
		return cnn.inception_module('incept_resnet_v2_a', cols)
	def inception_resnet_v2_b(cnn):
		cols = [[('conv', 192, 1, 1)],
		        [('conv', 128, 1, 1), ('conv', 160, 1, 7), ('conv', 192, 7, 1)]]
		return cnn.inception_module('incept_resnet_v2_b', cols)
	def inception_resnet_v2_c(cnn):
		cols = [[('conv', 192, 1, 1)],
		        [('conv', 192, 1, 1), ('conv', 224, 1, 3), ('conv', 256, 3, 1)]]
		return cnn.inception_module('incept_resnet_v2_c', cols)
	
	cnn.use_batch_norm = True
	residual_scale = 0.2
	cnn.conv (32, 3, 3, 2, 2, mode='VALID')
	cnn.conv (32, 3, 3, 1, 1, mode='VALID')
	cnn.conv (64, 3, 3)
	inception_v4_sa(cnn)
	inception_v4_sb(cnn)
	inception_v4_sc(cnn)
	for _ in xrange(5):
		cnn.residual( 384, inception_resnet_v2_a, scale=residual_scale)
	inception_v4_ra(cnn, 256, 256, 384, 384)
	for _ in xrange(10):
		# TODO: This was 1154 in the paper, but then the layers don't match up
		#         One Caffe model online appears to use 1088
		#         Facebook's Torch implem uses 1152
		cnn.residual(1152, inception_resnet_v2_b, scale=residual_scale)
	inception_resnet_v2_rb(cnn)
	for _ in xrange(5):
		# TODO: This was 2048 in the paper, but then the layers don't match up
		#         One Caffe model online appears to use 2080
		#         Facebook's Torch implem uses 2048 but modifies the preceding reduction net so that it matches
		#cnn.residual(2144, inception_resnet_v2_c, scale=residual_scale)
		cnn.residual(2048, inception_resnet_v2_c, scale=residual_scale)
	cnn.spatial_mean()
	cnn.dropout(0.8)
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

#cross_entropy = None
def loss_function(logits, labels):
	#global cross_entropy # HACK TESTING
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

def decode_jpeg(image_buffer, scope=None, dtype=tf.float32):
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
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    #image = tf.Print(image, [tf.shape(image)], "Image shape: ")

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=dtype) ## HACK TESTING disabled
    return image

def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  #with tf.op_scope([image, height, width], scope, 'eval_image'):
  #with tf.name_scope(scope, 'eval_image', [image, height, width]):
  with tf.name_scope(scope or 'eval_image'):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    ## HACK TESTING disabled
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
    #image = tf.image.resize_nearest_neighbor(image, [height, width], # HACK TESTING
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    
    #image *= 1./255 # HACK TESTING
    
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
    distorted_image = tf.image.resize_images(distorted_image, height, width,
                                             resize_method)
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
		image = decode_jpeg(image_buffer, dtype=self.dtype)
		if self.train and self.distortions:
			image = distort_image(image, self.height, self.width, bbox, thread_id)
		else:
			image = eval_image(image, self.height, self.width)
		# Rescale to [-1,1] instead of [0, 1)
		image = tf.sub(image, 0.5)
		image = tf.mul(image, 2.0)
		return image

#with tf.device('/gpu:X'):
#	images_queue = tf.FIFOQueue(capacity=2, dtypes=[dtype], name='images_queue')
#	enqueue_op = images_queue.enqueue([image])
#	tf.train.queue_runner.add_queue_runner(
#	  tf.train.queue_runner.QueueRunner(images_queue, [enqueue_op]))

	def minibatch(self, dataset, subset):
		with tf.name_scope('batch_processing'):
			data_files = dataset.data_files(subset)
			shuffle  = self.train
			capacity = 16 if self.train else 1
			print data_files
			filename_queue = tf.train.string_input_producer(data_files,
			                                                shuffle=shuffle,
			                                                capacity=capacity)
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
					dtypes=[tf.string])
			else:
				examples_queue = tf.FIFOQueue(
					capacity=examples_per_shard + 3 * self.batch_size,
					dtypes=[tf.string])
			if self.num_readers == 0: # Special case to use one reader per preproc thread
				_, example_serialized = dataset.reader().read(filename_queue)
				# HACK TESTING
				#_, ex = dataset.reader().read(filename_queue)
				#print type(ex)
				#example_serialized = tf.Variable(ex, trainable=False)
			else:
				enqueue_ops = []
				for _ in xrange(self.num_readers):
					_2, value = dataset.reader().read(filename_queue)
					enqueue_ops.append(examples_queue.enqueue([value]))
				tf.train.queue_runner.add_queue_runner(
				  tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
				example_serialized = examples_queue.dequeue()
			images_and_labels = []
			for thread_id in xrange(self.num_preprocess_threads):
				print '*'*100
				#example_serialized_hack = tf.Variable(example_serialized, trainable=False) # HACK TESTING
				# Parse a serialized Example proto to extract the image and metadata.
				image_buffer, label_index, bbox, _ = parse_example_proto(example_serialized)
				#image_buffer, label_index, bbox, _ = parse_example_proto(example_serialized_hack)
				image = self.preprocess(image_buffer, bbox, thread_id)
				images_and_labels.append([image, label_index])
			images, label_index_batch = tf.train.batch_join(
				images_and_labels,
				batch_size=self.batch_size,
				capacity=2 * self.num_preprocess_threads * self.batch_size)
			images = tf.cast(images, self.dtype)
			depth = 3
			images = tf.reshape(images, shape=[self.batch_size, self.height, self.width, depth])
			label_index_batch = tf.reshape(label_index_batch, [self.batch_size])
			# Display the training images in the visualizer.
			#tf.image_summary('images', images)
			
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
		tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % subset)
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
	nstep_burnin = 10
	perf_results = {}
	perf_results['model']          = model
	perf_results['batch_size']     = batch_size
	perf_results['num_batches']    = num_batches
	perf_results['devices']        = devices
	perf_results['dataset']        = str(dataset) if dataset is not None else None
	perf_results['distortions']    = FLAGS.distortions
	perf_results['weak_scaling']   = FLAGS.weak_scaling
	perf_results['num_readers']    = FLAGS.num_readers
	perf_results['num_preproc_threads'] = FLAGS.num_preprocess_threads
	perf_results['num_intra_threads']   = FLAGS.num_intra_threads
	perf_results['memory_fraction']     = FLAGS.memory_fraction
	perf_results['param_server']   = param_server_device
	perf_results['data_format']    = data_format
	perf_results['storage_dtype']  = 'float16' if use_fp16 else 'float32'
	perf_results['compute_dtype']  = 'float32'
	perf_results['mem_growth']     = enable_mem_growth
	perf_results['trace_filename'] = trace_filename
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
		
	if model == 'vgg' or model == 'googlenet' or model.startswith('resnet'):
		image_size = 224
	elif model == 'alexnet':
		image_size = 224+3
	elif model == 'overfeat':
		image_size = 231
	elif model.startswith('inception'):
		image_size = 299
	else:
		raise KeyError("Invalid model name: "+model)
	data_type = tf.float16 if use_fp16 else tf.float32
	#input_data_type = data_type
	input_data_type = tf.float32
	input_nchan = 3
	input_shape = [batch_size, image_size, image_size, input_nchan]
	#if share_variables:
	devices = [device_or_param_server(d, param_server_device) for d in devices]
	
	tf.set_random_seed(1234)
	np.random.seed(4321)
	phase_train = tf.placeholder(tf.bool, name='phase_train')
	
	with tf.device("/cpu:0"):
		if dataset is not None:
			preproc_train = ImagePreprocessor(image_size, image_size, batch_size, input_data_type, train=True)
			images_train, labels_train = preproc_train.minibatch(dataset, subset="train")
			# TODO: Can't do this because it creates another set of preload threads
			#preproc_valid = ImagePreprocessor(image_size, image_size, batch_size, input_data_type, train=False)
			#images_valid, labels_valid = preproc_valid.minibatch(dataset, subset="validation")
			#images = control_flow_ops.cond(phase_train, lambda: images_train, lambda: images_valid)
			#labels = control_flow_ops.cond(phase_train, lambda: labels_train, lambda: labels_valid)
			images = images_train
			labels = labels_train
			nclass = dataset.num_classes()
		else:
			nclass = 1000
			images = tf.truncated_normal(input_shape, dtype=input_data_type, stddev=1e-1, name="synthetic_images")
			#labels = tf.ones([batch_size], dtype=tf.int32, name="synthetic_labels")
			labels = tf.random_uniform([batch_size], minval=1, maxval=nclass, dtype=tf.int32, name="synthetic_labels")
			# Note: This results in a H2D copy, but no computation
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
	## Note: Label 0 is reserved for an (unused) background class
	#nclass += 1
	
		# HACK TESTING
		"""
		image_queues = []
		for d, device in enumerate(devices):
			image_queues.append(tf.FIFOQueue(capacity=4,
			                                 dtypes=[input_data_type],
			                                 shapes=[images_splits[d].get_shape()]))
			enqueue_op = image_queues[d].enqueue([images_splits[d]])
			tf.train.queue_runner.add_queue_runner(
				tf.train.queue_runner.QueueRunner(image_queues[d], [enqueue_op]))
		"""
	
	print "Generating model"
	device_grads = []
	losses       = []
	for d, device in enumerate(devices):
		# Note: We want variables on different devices to share the same
		#         variable scope, so we just use a name_scope here.
		#with tf.name_scope('tower_%i' % d) as scope:
		with tf.device(device), tf.name_scope('tower_%i' % d) as scope:
			#reuse = (d>0 and share_variables) or None
			#reuse = d>0 or None
			#*with tf.device(device):#, tf.variable_scope(tf.get_variable_scope(),
			                       #                   reuse=reuse):
				images = images_splits[d]
				# HACK TESTING
				"""
			grads = []
			nsubstep = 8
			nbuf = nsubstep#4
			image_bufs = []
			for buf in xrange(nbuf):
				image_bufs.append(tf.Variable(tf.zeros(input_shape, input_data_type), trainable=False))
			for substep in xrange(nsubstep):
				image_buf = image_bufs[substep % len(image_bufs)]
				with tf.device('/cpu:0'):
					# TODO: This works for synthetic data, but not for real
					#         data, probably because it adds more CPU overhead.
					host_images = image_queues[d].dequeue()
					#host_images = images_splits[d] # Does not work as intended (i.e., reuses instead of re-dequeueing)
				h2d = image_buf.assign(host_images)
				with tf.control_dependencies(grads + [h2d]):
					images = tf.identity(image_buf)
				"""
				
				"""
			h2d = image_buf[0].assign(image_queues[d].dequeue())
			imagesets = []
			for substep in xrange(1, nsubstep):
				with tf.control_dependencies([h2d]):
					images = tf.identity(image_buf)
				h2d = image_buf[substep].assign(image_queues[d].dequeue())
				
				image_buf = image_bufs[substep]
				#with tf.device('/cpu:0'):
				#	host_images = image_queues[d].dequeue()
				h2d = image_buf.assign(host_images)
				with tf.control_dependencies([h2d]):
					images = tf.identity(image_buf)
				"""
				"""
				# Sadly this doesn't work, because queues are CPU-only
				images_queue = tf.FIFOQueue(capacity=2,
				                            dtypes=[input_data_type],
				                            shapes=[input_shape],
				                            name='images_queue')
				with tf.device('/cpu:0'):
					enqueue_op = images_queue.enqueue([images_splits[d]])
				tf.train.queue_runner.add_queue_runner(
					tf.train.queue_runner.QueueRunner(images_queue, [enqueue_op]))
				images = images_queue.dequeue()
				"""
				
				if data_format == 'NCHW':
					images = tf.transpose(images, [0,3,1,2])
				if input_data_type != data_type:
					images = tf.cast(images, data_type)
				network = ConvNetBuilder(images, input_nchan,
				                         phase_train, data_format, data_type)
				if   model == 'vgg':        inference_vgg(network)
				elif model == 'googlenet':  inference_googlenet(network)
				elif model == 'overfeat':   inference_overfeat(network)
				elif model == 'alexnet':    inference_alexnet(network)
				elif model == 'inception3': inference_inception_v3(network)
				elif model == 'inception4': inference_inception_v4(network)
				elif model == 'resnet50':   inference_resnet_v1(network, (2,3,5,3))
				elif model == 'resnet101':  inference_resnet_v1(network, (2,3,22,3))
				elif model == 'resnet152':  inference_resnet_v1(network, (2,7,35,3))
				elif model == 'inception-resnet2': inference_inception_resnet_v2(network)
				else: raise KeyError("Invalid model name '%s'" % model)
				# Add the final fully-connected class layer
				logits = network.affine(nclass, activation='linear')
				# Note: This loss function uses Ops that have no current GPU implem
				#with tf.device("/cpu:0"):
				#	loss = loss_function_old(network.top_layer, labels_splits[d])
				loss = loss_function(logits, labels_splits[d])
				## Ensure in-place update ops are executed too (e.g., batch norm)
				#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				#if update_ops:
				#	updates = tf.group(*update_ops)
				#	loss = control_flow_ops.with_dependencies([updates], loss)
				
				l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
				weight_decay = FLAGS.weight_decay
				if weight_decay is not None and weight_decay != 0.:
					loss += weight_decay * l2_loss
				
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
	
	with tf.device(param_server_device):
		total_loss = tf.reduce_mean(losses)
		#all_grads = all_average_gradients4(device_grads)
		#avg_grads = all_average_gradients4(device_grads)
		#avg_grads = average_gradients(device_grads)
		avg_grads = average_gradients_inception(device_grads)
		gradient_clip = FLAGS.gradient_clip
		learning_rate = FLAGS.learning_rate
		momentum      = FLAGS.momentum
		if gradient_clip is not None:
			clipped_grads = [(tf.clip_by_value(grad, -gradient_clip, +gradient_clip), var) for grad,var in avg_grads]
		else:
			clipped_grads = avg_grads
		opt = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
		#opt = tf.train.RMSPropOptimizer(learning_rate)
		#opt = tf.train.AdamOptimizer()
		train_op = opt.apply_gradients(clipped_grads)
		# Ensure in-place update ops are executed too (e.g., batch norm)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		#print "****", update_ops
		if update_ops:
			#updates = tf.group(*update_ops)
			#train_op = control_flow_ops.with_dependencies([updates], train_op)
			train_op = tf.group(train_op, *update_ops)
	"""
	#device_train_ops = []
	for d, device in enumerate(devices):
		with tf.device(device):
			with tf.name_scope('tower_%i' % d) as scope:
				#grads = all_grads[d]
				grads = avg_grads
				grad_clipping = 1.
				for grad,var in grads:
					print grad, var
				clipped_grads = [(tf.clip_by_value(grad, -grad_clipping, +grad_clipping), var) for grad,var in grads]
				
				#device_train_ops.append( opt.apply_gradients(clipped_grads) )
	#all_train_ops = tf.group(*device_train_ops)
	"""
	init = tf.initialize_all_variables()
	
	config = tf.ConfigProto()
	#config.gpu_options.allocator_type = 'BFC'
	# Allocate as needed rather than all at once
	config.gpu_options.allow_growth = enable_mem_growth
	#config.gpu_options.per_process_gpu_memory_fraction
	config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_fraction
	config.intra_op_parallelism_threads = FLAGS.num_intra_threads
	#config.inter_op_parallelism_threads = 1 # HACK TESTING
	# TODO: Is this OK to use? Seems to provide a small ~3% speedup on AlexNet
	#config.graph_options.optimizer_options.do_function_inlining = True
	sess = tf.Session(config=config)
	# TODO: Look into these:
	# config.session_inter_op_thread_pool
	# config.use_per_session_threads
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
	queue_threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
	
	if FLAGS.checkpoint_file is not None:
		save_path = saver.save(sess, FLAGS.checkpoint_file, global_step=0)
		print "Checkpoint saved to %s" % save_path
	
	#print sess.run(images, feed_dict={phase_train: True})
	#print sess.run(network.top_layer, feed_dict={phase_train: True})
	#print sess.run(labels, feed_dict={phase_train: True})
	#print sess.run(cross_entropy, feed_dict={phase_train: True})
	#print sess.run(loss, feed_dict={phase_train: True})
	
	#all_grads = [grad for gradvars in device_grads for grad,var in gradvars]
	
	#run_metadata = tf.RunMetadata()
	#run_options  = None
	#if trace_filename is not None:
	#	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	
	#print "Step\tInfer\tTrain\texp(loss)"
	print "Step\tImg/sec\texp(loss)"
	perf_results['step_train_times'] = []
	perf_results['step_losses'] = []
	nstep = num_batches
	oom = False
	for step in xrange(nstep):
		if step == FLAGS.nvprof_start:
			cudaProfilerStart()
		if trace_filename is not None and step == 10:
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
		else:
			run_options  = None
			run_metadata = None
		
		start_time = time.time()
		# HACK TODO: Decide on how to allow inference mode
		## TODO: Should be something other than network.top_layer?
		#sess.run(network.top_layer, feed_dict={phase_train: False})
		infer_time = float('nan')#time.time() - start_time
		start_time = time.time()
		try:
			#sess.run(all_train_ops, feed_dict={phase_train: True})
			#sess.run(all_grads, feed_dict={phase_train: True})
			#_, lossval = sess.run([all_train_ops, total_loss], feed_dict={phase_train: True})
			_, lossval = sess.run([train_op, total_loss], feed_dict={phase_train: True},
			                      options=run_options, run_metadata=run_metadata)
			train_time = time.time() - start_time
		except tf.python.framework.errors.ResourceExhaustedError:
			train_time = -1.
			lossval    = -1.
			oom = True
		#print "Unlearned classes: %.7f" % np.exp(lossval)
		#print "%i\t%.3f\t%.3f\t%.3f" % (step+1, infer_time, train_time, np.exp(lossval))
		#print "%i\t%.1f\t%.1f\t%.3f" % (step+1,
		print "%i\t%.1f\t%.3f" % (step+1,
		                          #batch_size/infer_time,
		                          batch_size/train_time,
		                          np.exp(lossval))
		if trace_filename is not None and step == 10:
			print "Dumping trace to", trace_filename
			trace = timeline.Timeline(step_stats=run_metadata.step_stats)
			with open(trace_filename, 'w') as trace_file:
				trace_file.write(trace.generate_chrome_trace_format())
		perf_results['step_train_times'].append(train_time)
		perf_results['step_losses'].append(float(lossval))
		#if step == nstep_burnin+10 or step % 100 == 0:
		#	dump_perf_results()
		if FLAGS.checkpoint_file is not None and step+1 % 250 == 0:
			save_path = saver.save(sess, FLAGS.checkpoint_file, global_step=step+1)
			print "Checkpoint saved to %s" % save_path
		if step+1 == FLAGS.nvprof_stop:
			cudaProfilerStop()
		if oom:
			break
	times = np.array(perf_results['step_train_times'][nstep_burnin:])
	speeds     = batch_size / times
	speed_mean = np.mean(speeds)
	speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(len(speeds)))
	speed_madstd = 1.4826*np.median(np.abs(speeds - np.median(speeds)))
	speed_jitter = speed_madstd
	print '-'*64
	print "Images/sec: %.1f +/- %.1f (jitter = %.1f)" % (speed_mean, speed_uncertainty, speed_jitter)
	print '-'*64
	dump_perf_results()
	coordinator.request_stop()
	coordinator.join(queue_threads, stop_grace_period_secs=5.)
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
	cmdline = argparse.ArgumentParser()
	cmdline.add_argument('-m', '--model', default='googlenet',
	                     help="""Name of model to run:
	                     googlenet, vgg, overfeat, alexnet, inception[3,4],
	                     resnet[50,101,152] or inception-resnet2.""")
	cmdline.add_argument('-b', '--batch_size', default=64, type=int,
	                     help="""Size of each minibatch""")
	cmdline.add_argument('--num_batches', default=40, type=int,
	                     help="""Number of batches to run.""")
	cmdline.add_argument('-g', '--num_gpus', default=1, type=int,
	                     help="""Number of GPUs to run on.""")
	cmdline.add_argument('--data_format', default='NCHW',
	                     help="""Data layout to use: NHWC (TF native)
	                     or NCHW (cuDNN native).""")
	add_bool_argument(cmdline, '--use_fp16',
	                  help="""Use fp16 (half) instead of fp32 (float) for
	                  storage (compute is always fp32).""")
	cmdline.add_argument('--parameter_server', default='gpu',
	                     help="""Device to use as parameter server:
	                     cpu or gpu.""")
	add_bool_argument(cmdline, '--memory_growth',
	                  help="""Enable on-demand memory growth.""")
	cmdline.add_argument('--data_dir', default=None,
	                     help="""Path to dataset in TFRecord format
	                     (aka Example protobufs). If not specified,
	                     synthetic data will be used.""")
	cmdline.add_argument('--data_name', default=None,
	                     help="""Name of dataset: imagenet or flowers.
	                     If not specified, it is automatically guessed
	                     based on --data_dir.""")
	cmdline.add_argument('--num_preprocess_threads', default=4, type=int,
	                     help="""Number of preprocessing threads *per GPU*.
	                     Must be a multiple of 4.""")
	cmdline.add_argument('--num_readers', default=4, type=int,
	                     help="""Number of parallel readers during training.
	                     Setting this to 0 is a special case that causes each
	                     preprocessing thread to do its own reading.""")
	cmdline.add_argument('--num_intra_threads', default=1, type=int,
	                     help="""Number of threads to use for intra-op
	                     parallelism. If set to 0, the system will pick
	                     an appropriate number.""")
	cmdline.add_argument('--input_queue_memory_factor', default=16, type=int,
	                     help="""Size of the queue of preprocessed images.
	                     Default is ideal but try smaller values, e.g.
	                     4, 2 or 1, if host memory is constrained.""")
	add_bool_argument(cmdline, '--distortions', default=True,
	                     help="""Enable/disable distortions during
	                     image preprocessing.""")
	add_bool_argument(cmdline, '--weak_scaling',
	                  help="""Interpret batch_size as *per GPU*
	                  rather than total.""")
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
	cmdline.add_argument('--checkpoint_file', default=None,
	                     help="""Write training checkpoints with this file
	                     name (note: also writes some other files to same
	                     path).""")
	cmdline.add_argument('--learning_rate', default=0.002, type=float,
	                     help="""Learning rate for training.""")
	cmdline.add_argument('--momentum', default=0.9, type=float,
	                     help="""Momentum for training.""")
	cmdline.add_argument('--gradient_clip', default=None, type=float,
	                     help="""Gradient clipping magnitude.
	                     Disabled by default.""")
	cmdline.add_argument('--weight_decay', default=1e-4, type=float,
	                     help="""Weight decay factor for training.""")
	cmdline.add_argument('--shmoo', action='store_true',
	                     help="""Run a big shmoo over many
	                     parameter combinations.""")
	cmdline.add_argument('--nvprof_start', default=-1, type=int,
	                     help="""Iteration at which to start CUDA profiling.
	                     A value of -1 means program start.""")
	cmdline.add_argument('--nvprof_stop', default=-1, type=int,
	                     help="""Iteration at which to stop CUDA profiling.
	                     A value of -1 means program end.""")
	cmdline.add_argument('--memory_fraction', default=0., type=float,
	                     help="""Fraction of GPU memory to use.
	                     Set to 0.0 to allocate max amount (default).""")
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
	
	model       = FLAGS.model
	batch_size  = FLAGS.batch_size
	devices     = ['/gpu:%i'%i for i in xrange(FLAGS.num_gpus)]
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
	
	data_format = FLAGS.data_format
	num_batches = FLAGS.num_batches
	use_fp16    = FLAGS.use_fp16
	if not FLAGS.shmoo:
		print "Model:      ", model
		print "Batch size: ", batch_size, 'global'
		print "            ", batch_size/len(devices), 'per device'
		print "Devices:    ", devices
		print "Data format:", data_format
		print "Data type:  ", 'fp16' if use_fp16 else 'fp32'
		
		#test_cnn(model, batch_size/len(devices), devices, ps_device,
		with tf.Graph().as_default(): # Ensure graph is freed
			test_cnn(model, batch_size, devices, dataset, ps_device,
			         data_format, num_batches,# share_variables=share_vars,
			         use_fp16=use_fp16, enable_mem_growth=mem_growth,
			         perf_filename=perf_filename,
			         trace_filename=trace_filename)
	else: # shmoo
		print "Running shmoo"
		for use_fp16 in [False, True]:
			for model in ['alexnet', 'vgg', 'googlenet', 'overfeat', 'inception3']:
				for ps_device in ['/cpu:0', '/gpu:0']:
					for ngpu in [1, 2, 4, 8]:
						if ngpu > len(devices):
							continue
						shmoo_devices = devices[:ngpu]
						for batch_size in [64, 128, 256, 512]:
							if model == 'inception3' and batch_size > 64:
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

