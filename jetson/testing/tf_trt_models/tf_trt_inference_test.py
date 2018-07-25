import image_processing
import time
import pickle
import argparse
import sys
import os
import urllib
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
from tf_trt_models.classification import download_classification_checkpoint, build_classification_graph

parser = argparse.ArgumentParser(description='choose model')
parser.add_argument('--model', dest='model', metavar='M', default='inception_v4')
parser.add_argument('--num_classes', dest='num_classes',type=int, metavar='N', default=1001)
parser.add_argument('--use_trt', dest='use_trt', type=int, metavar='T', default=0)
parser.add_argument('--detection', dest='det_mode', type=int, metavar='D', default=0)
parser.add_argument('--tolerance', dest='tolerance', type=float, default=0.0001)
args = parser.parse_args()

MODEL = args.model
print(MODEL)
CHECKPOINT_PATH = MODEL + '.ckpt'
NUM_CLASSES = args.num_classes
print(NUM_CLASSES)
USE_TRT = args.use_trt
DET_MODE = args.det_mode
TOLERANCE = args.tolerance


checkpoint_path = download_classification_checkpoint(MODEL, './data')

frozen_graph, input_names, output_names = build_classification_graph(model=MODEL, checkpoint=checkpoint_path, num_classes=NUM_CLASSES)

if USE_TRT == 1:
    frozen_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=32,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(frozen_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_output = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')

height = tf_input.shape.as_list()[1] 
width = tf_input.shape.as_list()[2]
preproc_func = lambda record: image_processing._parse_and_preprocess_image_record(record, 0, height, width)


print(MODEL)
for batch_size in [1]:
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(preproc_func)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()


    val_filenames = [os.getcwd() + '/dataset/validation-00001-of-00128']

    output = tf_sess.run(iterator.initializer, feed_dict={filenames: val_filenames})

    correct_numb = 0
    
    start = time.time()
    for i in range(int((1023 + batch_size)/batch_size)):
        
        n = tf_sess.run(next_element)
        x = n[0]
        value = tf_sess.run(tf_output, feed_dict={tf_input: x})
        
        for j in range(batch_size):
            if (np.argmax(value[j]) - (NUM_CLASSES-1000)) == n[1][j]:
                correct_numb += 1
    t = time.time() - start

    if DET_MODE == 0:
        print("for batch_size = " + str( batch_size) + " accuracy = " + str(float(correct_numb)/1024.0))
        print("time = " + str(t))
        
        if MODEL == 'inception_v1':
            if abs(result - 0.7197265625) > TOLERANCE:
                print("FAILED!")
            else:
                print("PASSED!")

        if MODEL == 'inception_v2':
            if abs(result - 0.7392578125) > TOLERANCE:
                print("FAILED!")
            else:
                print("PASSED!")

        if MODEL == 'inception_v3':
            if abs(result - 0.787109375) > TOLERANCE:
                print("FAILED!")
            else:
                print("PASSED!")

        if MODEL == 'inception_v4':
            if abs(result - 0.794921875) > TOLERANCE:
                print("FAILED!")
            else:
                print("PASSED!")

        if MODEL == 'resnet_v2_50':
            if abs(result - 0.775390625) > TOLERANCE:
                print("FAILED!")
            else:
                print("PASSED!")


        if MODEL == 'resnet_v2_101':
            if abs(result - 0.798828125) > TOLERANCE:
                print("FAILED!")
            else:
                print("PASSED!")



pickle.dump(sc, open('cur_res' + MODEL + '.pkl', "w+"))



