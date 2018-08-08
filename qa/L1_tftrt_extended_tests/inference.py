import argparse
import os
import tensorflow as tf
import time
import numpy as np
import sys
from graph import get_frozen_graph, get_preprocess_fn

class LoggerHook(tf.train.SessionRunHook):
  """Logs runtime of each iteration"""
  def __init__(self):
      self.iter_times = []

  def begin(self):
    self.start_time = time.time()

  def after_run(self, run_context, run_values):
    current_time = time.time()
    duration = current_time - self.start_time
    self.start_time = current_time
    self.iter_times.append(duration)

def run(model, use_trt, data_dir, batch_size, num_iterations):
    """Evaluates a model
    
    This function evaluates a given model on the ImageNet validation set.
    The model definition is pulled from the NETS dict in graph.py and used to create a frozen
    GraphDef ready to be used for inference. tf.estimator.Estimator is used to evaluate the
    accuracy of the model and a few other metrics. The results are printed to stdout.

    model: strings, the model name (see NETS table in graph.py)
    use_trt: bool, if true, use TensorRT
    data_dir: string, directory containing ImageNet validation TFRecord files
    batch_size: int, batch size for TensorRT optimizations
    num_iterations: int, number of iterations(batches) to run for
    """
    # Retrive the graph
    frozen_graph = get_frozen_graph(model, use_trt, batch_size)
    # Define model function for tf.estimator.Estimator
    def model_fn(features, labels, mode):
        logits_out, classes_out = tf.import_graph_def(frozen_graph,
            input_map={'input': features},
            return_elements=['logits:0', 'classes:0'],
            name='')
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=classes_out, name='acc_op')
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    # Create the dataset
    preprocess_fn = get_preprocess_fn(model)
    validation_files = tf.gfile.Glob(os.path.join(data_dir, 'validation*'))
    # Define the dataset input function for tf.estimator.Estimator
    def eval_input_fn():
        dataset = tf.data.TFRecordDataset(validation_files)
        dataset = dataset.map(preprocess_fn)
        dataset = dataset.repeat(count=1)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    # Evaluate model
    logger = LoggerHook()
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    results = estimator.evaluate(eval_input_fn, steps=num_iterations, hooks=[logger])
    
    # Gather additional results
    print("model: " + model)
    print("total_time: %.1f" % sum(logger.iter_times))
    print("images_per_sec: %d" % int(round(len(logger.iter_times) * batch_size / sum(logger.iter_times))))
    print("99th_percentile: %.1f" % (np.sort(logger.iter_times)[int(0.99 * len(logger.iter_times)) - 1] * 1000))
    print("num_nodes: %d" % len(frozen_graph.node))
    print("accuracy: %.4f" % results['accuracy'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose model')
    parser.add_argument('--model', type=str, default='inception_v4', help='Which model to use. See NETS table in graph.py for available networks.')
    parser.add_argument('--data_dir', type=str, default='/data/imagenet/train-val-tfrecord-480', help='Directory containing validation set TFRecord files.')
    parser.add_argument('--use_trt', action='store_true', help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--mode', type=str, choices=['classification', 'detection'], default='classification', help='Whether the model will be used for classification or object detection.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images per batch.')
    parser.add_argument('--num_iterations', type=int, default=None, help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    args = parser.parse_args()

    if args.mode == 'detection':
        raise NotImplementedError('This script currently only supports classification.')

    run(model=args.model, use_trt=args.use_trt, data_dir=args.data_dir, batch_size=args.batch_size, num_iterations=args.num_iterations)
