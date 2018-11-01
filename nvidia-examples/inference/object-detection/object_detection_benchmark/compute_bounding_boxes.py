import argparse
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json
import os
import pdb
from .models import INPUT_NAME, NUM_DETECTIONS_NAME, BOXES_NAME, CLASSES_NAME, SCORES_NAME
import threading
import time

try:
    from Queue import Queue
except:
    from queue import Queue

"""Computes bounding boxes over a set of COCO images."""

def compute_bounding_boxes(
    frozen_graph_path, 
    annotation_file,
    images_dir,
    batch_size=1,
    image_ids=None,
    max_queue_size=20,
    image_shape=None):
    """Computes bounding boxes and returns detections and performance metrics"""

    coco = COCO(annotation_file=annotation_file)
    if image_ids is None:
        image_ids = coco.getImgIds()

    # load frozen graph from file
    with open(frozen_graph_path, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True

    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:

            # import frozen graph and get relevant tensors
            tf.import_graph_def(frozen_graph, name='')
            tf_input = tf_graph.get_tensor_by_name(INPUT_NAME + ':0')
            tf_boxes = tf_graph.get_tensor_by_name(BOXES_NAME + ':0')
            tf_classes = tf_graph.get_tensor_by_name(CLASSES_NAME + ':0')
            tf_scores = tf_graph.get_tensor_by_name(SCORES_NAME + ':0')
            tf_num_detections = tf_graph.get_tensor_by_name(NUM_DETECTIONS_NAME + ':0')

            detections = []

            load_queue = Queue()
            process_queue = Queue(max_queue_size)

            # enqueue start image index of each batch for processing
            for image_index in range(0, len(image_ids), batch_size):
                batch_image_ids = image_ids[image_index:image_index + batch_size]
                load_queue.put(batch_image_ids)

            def load_batches():
                """Load images from file, add to batch, and queue for processing"""
                while not load_queue.empty():
                    batch_image_ids = load_queue.get()
                    batch_images = []

                    for image_id in batch_image_ids:
                        image_path = os.path.join(images_dir, coco.imgs[image_id]['file_name'])
                        image = Image.open(image_path).convert('RGB')

                        if image_shape is not None:
                            # image shape is provided as (height, width) so reversed indexing is used
                            # to match PIL api
                            image = image.resize(image_shape[::-1]) 

                        image = np.array(image)
                        batch_images.append(image)
                    process_queue.put((batch_image_ids, batch_images))

            def process_batches(performance):
                total_image_count = len(image_ids)
                print_interval = 100
                avg_runtime = 0.0
                image_count = 0
                latencys = []
                throughputs = []
                latencys_int = []
                throughputs_int = []
                """Run neural network to process image batches from queue."""
                while (not load_queue.empty()) or (not process_queue.empty()):

                    if process_queue.empty():
                        continue

                    batch_image_ids, batch_images = process_queue.get()

                    t0 = time.time()
                    boxes, classes, scores, num_detections = tf_sess.run(
                        [ tf_boxes, tf_classes, tf_scores, tf_num_detections ],
                        feed_dict={
                            tf_input: batch_images
                    })
                    t1 = time.time()
                    latency = 1000.0 * float(t1 - t0)
                    throughput = float(len(batch_images)) / float(t1 - t0)
                    latencys.append(latency)
                    throughputs.append(throughput)
                    latencys_int.append(latency)
                    throughputs_int.append(throughput)
                    image_count += len(batch_images)

                    if (image_count % print_interval) < batch_size:
                        print('    image %d/%d, latency(ms)=%f, throughput(images/sec)=%f' % (image_count, total_image_count, 
                            np.mean(latencys_int), np.mean(throughputs_int)))
                        latencys_int = []
                        throughputs_int = []

                    for i, image_id in enumerate(batch_image_ids):


                        image_width = coco.imgs[image_id]['width']
                        image_height = coco.imgs[image_id]['height']

                        for j in range(int(num_detections[i])):

                            bbox = boxes[i][j]
                            bbox_coco_fmt = [
                                bbox[1] * image_width, # x0
                                bbox[0] * image_height, # y0
                                (bbox[3] - bbox[1]) * image_width, # width
                                (bbox[2] - bbox[0]) * image_height # height
                            ]

                            detection = {
                                'image_id': image_id,
                                'category_id': int(classes[i][j]),
                                'bbox': bbox_coco_fmt,
                                'score': float(scores[i][j])
                            }

                            detections.append(detection)

                performance['throughput_mean'] = np.mean(throughputs)
                performance['throughput_median'] = np.median(throughputs)
                performance['latency_mean'] = np.mean(latencys)
                performance['latency_median'] = np.median(latencys)
                performance['latency_99th'] = np.percentile(latencys, q=99, interpolation='lower')

            performance = {} # will be filled in by process_thread

            load_thread = threading.Thread(target=load_batches)
            process_thread = threading.Thread(target=process_batches, args=(performance,))
            load_thread.start()
            process_thread.start()

            load_thread.join()
            process_thread.join()

    return detections, performance 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('frozen_graph_path')
    parser.add_argument('annotation_file')
    parser.add_argument('images_dir')
    parser.add_argument('detections_file_path')
    parser.add_argument('--image_ids_path', default='image_ids.json')
    parser.add_argument('--batch_size', default=1)
    args = parser.parse_args()

    with open(args.image_ids_path, 'r') as f:
        image_ids = json.load(f)

    detections, performance = compute_bounding_boxes(
        args.frozen_graph_path,
        args.annotation_file,
        args.images_dir,
        args.detections_file_path,
        batch_size=args.batch_size,
        image_ids=image_ids
    )

    with open(args.detections_file_path, 'w') as f:
        json.dump(detections, f)
