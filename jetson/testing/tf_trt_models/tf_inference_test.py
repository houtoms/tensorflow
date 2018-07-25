import pickle
import time
import argparse
from PIL import Image
import sys
import os
import urllib
import tensorflow as tf
#import tensorflow.contrib.tensorrt as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tf_trt_models.classification import download_classification_checkpoint, build_classification_graph


parser = argparse.ArgumentParser(description='choose model')
parser.add_argument('--model', dest='model', metavar='M', default='inception_v4')
parser.add_argument('--num_classes', dest='num_classes',type=int, metavar='N', default=1001)
args = parser.parse_args()





MODEL = args.model
print(MODEL)
CHECKPOINT_PATH = MODEL + '.ckpt'
NUM_CLASSES = args.num_classes
print(NUM_CLASSES)
LABELS_PATH = './examples/classification/data/imagenet_labels_%d.txt' % NUM_CLASSES
IMAGE_PATH = './examples/classification/data/dog-yawning.jpg'



checkpoint_path = download_classification_checkpoint(MODEL, './data')


frozen_graph, input_names, output_names = build_classification_graph(model=MODEL, checkpoint=checkpoint_path, num_classes=NUM_CLASSES)




tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(frozen_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_output = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')






image = Image.open(IMAGE_PATH)

plt.imshow(image)

width = int(tf_input.shape.as_list()[1])
height = int(tf_input.shape.as_list()[2])

image = np.array(image.resize((width, height)))




for i in range(5):
    start = time.time()
    output = tf_sess.run(tf_output, feed_dict={
        tf_input: image[None, ...]
    })
    print(i, " :    ", time.time()-start)

scores = output[0]



with open(LABELS_PATH, 'r') as f:
    labels = f.readlines()

sc = scores.argmax()

#[::-1][0:5]

pickle.dump(sc, open('cur_res' + MODEL + '.pkl', "w+"))

print(scores[i], labels[i])



