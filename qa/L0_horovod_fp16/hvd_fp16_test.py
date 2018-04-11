import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
import sys

hvd.init()

rank = hvd.rank()
num_ranks = hvd.size()

if rank == 0:
    print("Running {} ranks\n".format(num_ranks))

config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

expected_val = max(num_ranks-1, 1) * num_ranks
num_passed = 0
num_failed = 0

sess = tf.Session(config=config)

for device in ["/cpu:0", "/gpu:0"]:
    for precision in [tf.float32, tf.float16]:
        if rank == 0:
            print("Running {} test on {}".format(precision.name, device))
        tag = "_{}_{}".format(precision.name, device[1:4])
        with tf.device("/gpu:0"):
            v = tf.placeholder(precision)
            x = tf.fill([5, 4], v)
            b = hvd.broadcast(x, num_ranks-1, name="bcast"+tag)
            s = hvd._allreduce(b, name="reduce"+tag)
            g = hvd.allgather(s, name="gather"+tag)
        
        result = sess.run(g, feed_dict={v : rank if num_ranks > 1 else 1})
        expected = np.full((5*num_ranks, 4), expected_val,
                           dtype=precision.as_numpy_dtype())

        # We depend on exact arithmetic for small integers
        if np.array_equal(result, expected):
            print("PASS on rank {}".format(rank))
            num_passed += 1
        else:
            print("FAIL on rank {}".format(rank))
            print("Expected:\n{}\n BUT GOT\n{}".format(expected, result))
            num_failed += 1

num_tot = num_passed + num_failed
print("{} of {} tests passed on rank {}".format(num_passed, num_tot, rank))
sys.exit(num_failed)
