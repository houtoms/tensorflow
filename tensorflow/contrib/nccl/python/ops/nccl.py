"""NCCL op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

_nccl_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'tf_nccl.so'))
_nccl_all_reduce = _nccl_module.all_reduce
_nccl_bcast = _nccl_module.bcast

_dependency_tracking = {}

def all_reduce(to_reduce, my_rank, all_ranks):
    if my_rank in _dependency_tracking:
        with tf.control_dependencies([_dependency_tracking[my_rank]]):
            ret = _nccl_all_reduce(to_reduce, my_rank, all_ranks)
    else:
        ret = _nccl_all_reduce(to_reduce, my_rank, all_ranks)
    _dependency_tracking[my_rank] = ret
    return ret

def bcast(to_reduce, from_rank, my_rank, all_ranks):
    if my_rank in _dependency_tracking:
        with tf.control_dependencies([_dependency_tracking[my_rank]]):
            ret = _nccl_bcast(to_reduce, from_rank, my_rank, all_ranks)
    else:
        ret = _nccl_bcast(to_reduce, from_rank, my_rank, all_ranks)
    _dependency_tracking[my_rank] = ret
    return ret

def clean():
    global _dependency_tracking
    _dependency_tracking = {}
