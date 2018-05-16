import tensorflow as tf
import numpy as np


inputs = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]


tf_mean = tf.constant([121, 115, 100], dtype=tf.float32)
tf_std  = tf.constant([70, 68, 71], dtype=tf.float32)
tf_in = tf.placeholder(tf.float32)
tf_out = (tf_in - tf_mean) * (1. / tf_std)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("TF RESULT")
print(sess.run(tf_out, feed_dict={tf_in: inputs}))


np_mean = np.array([121, 115, 100], dtype=np.float32)
np_std  = np.array([70, 68, 71], dtype=np.float32)
print("NP RESULT")
print((inputs-np_mean)*(1. / np_std))
