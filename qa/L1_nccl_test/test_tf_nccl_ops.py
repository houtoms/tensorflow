#!/usr/bin/env python

"""
Notes on NCCL ops
-----------------
All collective ops must be executed together to avoid deadlock.
"""
from __future__ import print_function
from builtins import range

import tensorflow as tf
from tensorflow.python.ops import nccl_ops

import numpy as np
import time

def test_allreduce(sess, devices):
    n = 4096
    data = np.random.randint(-128, 128, size=(n,n)).astype(np.float32)
    inputs = []
    for device_name in devices:
        with tf.device(device_name) as device:
            inputs.append(tf.constant(data))
    
    sums = nccl_ops.all_sum(inputs)
    sum_ops = [s.op for s in sums]
    
    ngpu = len(devices)
    nwarmup = 10
    nrep    = 100
    print("Running allreduce warmup")
    for _ in range(nwarmup):
        sess.run(sum_ops)
    print("Running allreduce actual")
    start_time = time.time()
    for _ in range(nrep):
        sess.run(sum_ops)
    elapsed_secs = time.time() - start_time
    nbyte = data.size * data.dtype.itemsize
    print("Allreduce aggregate BW: {} GB/s".format(nrep*ngpu*nbyte/elapsed_secs/1e9))
    
    results = sess.run(sums)
    return all([np.all(result == ngpu*data) for result in results])

def test_broadcast(sess, devices):
    n = 4096
    data = np.random.randint(-128, 128, size=(n,n)).astype(np.float32)
    with tf.device(devices[0]):
        data0 = tf.constant(data)
    bcast_data = nccl_ops.broadcast(data0)

    received_tensors = []
    for device_name in devices:
        with tf.device(device_name):
            received_tensors.append(tf.identity(bcast_data))
    recv_ops = [r.op for r in received_tensors]
    
    ngpu = len(devices)
    nwarmup = 10
    nrep    = 100
    print("Running broadcast warmup")
    for _ in range(nwarmup):
        sess.run(recv_ops)
    print("Running broadcast actual")
    start_time = time.time()
    for _ in range(nrep):
        sess.run(recv_ops)
    elapsed_secs = time.time() - start_time
    nbyte = data.size * data.dtype.itemsize
    print("Broadcast aggregate BW: {} GB/s".format(nrep*ngpu*nbyte/elapsed_secs/1e9))
    
    results = sess.run(received_tensors)
    return all([np.all(result == data) for result in results]) and len(results) == len(devices)

def main():
    import sys
    ngpu = 2
    if len(sys.argv) > 1:
        ngpu = int(sys.argv[1])
    print("Running on {} GPUs".format(ngpu))
    devices = ['/gpu:%i' % i for i in range(ngpu)]
    sess = tf.Session()
    allreduce_success = test_allreduce(sess, devices)
    broadcast_success = test_broadcast(sess, devices)
    print("Allreduce", "PASSED" if allreduce_success else "FAILED")
    print("Broadcast", "PASSED" if broadcast_success else "FAILED")
    sess.close()
    
    if not allreduce_success or not broadcast_success:
        sys.exit(-1)

if __name__ == '__main__':
    main()

