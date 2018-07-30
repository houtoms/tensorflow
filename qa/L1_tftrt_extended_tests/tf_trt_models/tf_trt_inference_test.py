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
from tf_trt_models.detection import download_detection_model, build_detection_graph


parser = argparse.ArgumentParser(description='choose model')
parser.add_argument('--model', dest='model', metavar='M', default='inception_v4')
parser.add_argument('--num_classes', dest='num_classes',type=int, metavar='N', default=1001)
parser.add_argument('--use_trt', dest='use_trt', type=int, metavar='T', default=0)
parser.add_argument('--detection', dest='det_mode', type=int, metavar='D', default=0)
parser.add_argument('--tolerance', dest='tolerance', type=float, default=0.0001)
args = parser.parse_args()

MODEL = args.model
CHECKPOINT_PATH = MODEL + '.ckpt'
CONFIG_PATH = MODEL + '.config'
NUM_CLASSES = args.num_classes
USE_TRT = args.use_trt
DET_MODE = args.det_mode
TOLERANCE = args.tolerance


if DET_MODE == 0:
	checkpoint_path = download_classification_checkpoint(MODEL, './data')
	frozen_graph, input_names, output_names = build_classification_graph(model=MODEL, checkpoint=checkpoint_path, num_classes=NUM_CLASSES)
else:
	config_path, checkpoint_path = download_detection_model(MODEL, './data')
	frozen_graph, input_names, output_names = build_detection_graph(config=config_path, checkpoint=checkpoint_path)


if USE_TRT == 1:
    frozen_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=32,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP32',
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

if DET_MODE == 1:
    preproc_func = lambda record: image_processing._parse_and_preprocess_image_record(record, 0, 300, 300)
else:
    preproc_func = lambda record: image_processing._parse_and_preprocess_image_record(record, 0, height, width)


print(MODEL)

batches = [1, 32]
if MODEL == 'resnet_v2_101':
    batches = [1, 16]    


for batch_size in batches:
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
    
    corr = []
    out = []

    start = time.time()
    for i in range(int((1023 + batch_size)/batch_size)):
        
        n = tf_sess.run(next_element)
        x = n[0]
        value = tf_sess.run(tf_output, feed_dict={tf_input: x})
		

        for j in range(batch_size):
            corr.append(n[1][j])
            out.append(value[j])
            if (np.argmax(value[j]) - (NUM_CLASSES-1000)) == n[1][j]:
                correct_numb += 1
    t = time.time() - start
    
    print("MODEL")
    if DET_MODE == 0:
        print("for batch_size = " + str( batch_size) + " accuracy = " + str(float(correct_numb)/1024.0))
        print("time = " + str(t))
        result = float(correct_numb)/1024.0
        if MODEL == 'inception_v1':
            if abs(result - 0.72265625) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'inception_v2':
            if abs(result - 0.7392578125) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'inception_v3':
            if abs(result - 0.7841796875) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'inception_v4':
            if abs(result - 0.794921875) > TOLERANCE:
				print("FAIL")
                #exit(1)
            else:
                print("PASS")

        if MODEL == 'resnet_v2_50':
            if abs(result - 0.783203125) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'resnet_v2_101':
            if abs(result - 0.7958984375) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'resnet_v2_152':
            if abs(result - 0.794921875) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'resnet_v1_50':
            if abs(result - 0.7451171875) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'resnet_v1_101':
            if abs(result - 0.76953125) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'resnet_v1_152':
            if abs(result - 0.794921875) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")
        if MODEL == 'mobilenet_v1_0p25_128':
            if abs(result - 0.42578125) > TOLERANCE:
                print("FAIL")
                exit(1)
            else:
                print("PASS")

        if MODEL == 'mobilenet_v1_0p5_160':
            if abs(result - 0.580078125) > TOLERANCE:
                print("FAIL")
            else:
                print("PASS")

    else:
        out = np.stack(out, axis=0)
        out = out.flatten()
        with open('dataset/cpures.pickle', 'rb') as handle:
            corr = pickle.load(handle)
        r = corr - out
        r = r*r
        if sqrt(np.sum(r)/len(r)) > TOLERANCE:
            print("FAIL")
            exit(1)

        else:
            print("PASS")
	
