import argparse
import subprocess
import os
import sys
from .models import MODELS, PIPELINE_CONFIG_NAME, \
    FROZEN_GRAPH_NAME, ssd_config_override, faster_rcnn_config_override, \
    CHECKPOINT_PREFIX
from .download_model import download_model
from .optimize_graph import optimize_graph
from .compute_bounding_boxes import compute_bounding_boxes
from .compute_stats import compute_stats
import json

MODELS_SUBDIR = 'models'
REFERENCE_MAP_nms30 = {
    'ssd_mobilenet_v1_coco': 0.23379014555646466,
    'ssd_mobilenet_v2_coco': 0.25172600340734186,
    'ssd_inception_v2_coco': 0.28200308679299046,
    'ssd_resnet_50_fpn_coco': 0.38640100249113785,
    'faster_rcnn_resnet50_coco': 0.32181505601180527,
    'mask_rcnn_resnet50_atrous_coco': 0.31990751371814236
}

def test_config_str(
    model_name,
    use_trt,
    precision_mode,
    batch_size,
    minimum_segment_size,
    nms_score_threshold,
    force_nms_cpu,
    remove_assert):
    """Generates a unique configuration string given configuration parameters"""

    use_trt_substr = {False: 'tf', True: 'trt'}

    config_str = model_name
    config_str += '_' + use_trt_substr[use_trt]
    config_str += '_' + precision_mode
    config_str += '_bs%d' % batch_size
    config_str += '_mss%d' % minimum_segment_size
    config_str += '_nms%d' % int(nms_score_threshold * 100)
    config_str += '_pp%d%d' % (force_nms_cpu, remove_assert)

    return config_str


def test(
    model_name,
    coco_dir='coco',
    coco_year=2017,
    static_data_dir='static_data',
    data_dir='data',
    use_trt=False,
    precision_mode='FP32',
    batch_size=1,
    minimum_segment_size=2,
    force_nms_cpu=False,
    remove_assert=False,
    nms_score_threshold=0.3,
    image_ids_path=None,
    reference_map=None,
    map_error_threshold=0.001):

    config_str = test_config_str(
        model_name,
        use_trt,
        precision_mode,
        batch_size,
        minimum_segment_size,
        nms_score_threshold,
        force_nms_cpu,
        remove_assert
    )

    input_models_dir = os.path.join(static_data_dir, MODELS_SUBDIR)
    models_dir = os.path.join(data_dir, MODELS_SUBDIR)

    model_dir = os.path.join(models_dir, config_str)
    frozen_graph_path = os.path.join(model_dir, FROZEN_GRAPH_NAME)
    results_file_path = os.path.join(model_dir, 'results.json')
    stats_file_path = os.path.join(model_dir, 'stats.json')
    input_model_dir = os.path.join(input_models_dir, MODELS[model_name].extract_dir)
    input_pipeline_config_path = os.path.join(input_model_dir, PIPELINE_CONFIG_NAME)
    input_checkpoint_prefix = os.path.join(input_model_dir, CHECKPOINT_PREFIX)

    coco_annotation_path = os.path.join(coco_dir, 'annotations', 'instances_val%d.json' % coco_year)
    coco_image_dir = os.path.join(coco_dir, 'val%d' % coco_year)

    # download source model (will use cached if available)
    download_model(model_name, output_dir=input_models_dir)

    if 'ssd' in model_name:
        config_override = ssd_config_override(nms_score_threshold)
    else:
        config_override = faster_rcnn_config_override(nms_score_threshold)

    # re-export model to resolve version dependencies
    if os.path.isdir(model_dir):
        print('Existing configuration found, removing it.')
        subprocess.call(['rm', '-rf', model_dir])

    subprocess.call(['python', '-m', 'object_detection.export_inference_graph',
        '--input_type', 'image_tensor',
        '--input_shape', '%d,-1,-1,3' % batch_size,
        '--pipeline_config_path', input_pipeline_config_path,
        '--output_directory', model_dir,
        '--trained_checkpoint_prefix', input_checkpoint_prefix,
        '--config_override', config_override])

    # optimize model 
    optimize_graph(
        frozen_graph_path, 
        frozen_graph_path, # output same path tooverwrite frozen graph
        force_nms_cpu=force_nms_cpu,
        remove_assert=remove_assert,
        use_trt=use_trt,
        precision_mode=precision_mode,
        minimum_segment_size=minimum_segment_size,
        batch_size=batch_size
    )

    # compute bounding boxes
    if image_ids_path is not None:
        with open(image_ids_path, 'r') as f:
            image_ids = json.load(f)
    else:
        image_ids = None

    results, avg_runtime = compute_bounding_boxes(
        frozen_graph_path,
        coco_annotation_path,
        coco_image_dir,
        batch_size=batch_size,
        image_ids=image_ids
    )

    with open(results_file_path, 'w') as f:
        json.dump(results, f)

    # compute statistics
    stats = compute_stats(coco_annotation_path, results_file_path, image_ids=image_ids)
    with open(stats_file_path, 'w') as f:
        json.dump(stats, f)

    return abs(stats['map'] - reference_map) <= map_error_threshold


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--coco_dir', default='coco')
    parser.add_argument('--coco_year', default=2017)
    parser.add_argument('--static_data_dir', default='static_data')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--use_trt', action='store_true')
    parser.add_argument('--precision_mode', default='FP32')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--minimum_segment_size', default=2)
    parser.add_argument('--force_nms_cpu', action='store_true')
    parser.add_argument('--remove_assert', action='store_true')
    parser.add_argument('--nms_score_threshold', default=0.3)
    parser.add_argument('--image_ids_path', default=None)
    parser.add_argument('--reference_map_path', default=None)
    parser.add_argument('--reference_map', default=None)
    parser.add_argument('--map_error_threshold', default=0.001)
    args = parser.parse_args()

    # get reference MAP from 
    # (a) Json file containing map of model names to MAPs (set using --reference_map_path)
    # (b) reference map set using --reference_map
    # (c) default reference map for model
    if args.reference_map_path is not None:
        with open(args.reference_map_path, 'r') as f:
            reference_map_dict = json.load(f)
            reference_map = reference_map_dict[args.model_name]
    elif args.reference_map is not None:
        reference_map = float(args.reference_map)
    else:
        reference_map = REFERENCE_MAP_nms30[args.model_name] # default to reference MAP at 0.3 nms thresh


    success = test(
        args.model_name,
        coco_dir=args.coco_dir,
        coco_year=int(args.coco_year),
        static_data_dir=args.static_data_dir,
        data_dir=args.data_dir,
        use_trt=args.use_trt,
        precision_mode=args.precision_mode,
        batch_size=int(args.batch_size),
        minimum_segment_size=int(args.minimum_segment_size),
        force_nms_cpu=args.force_nms_cpu,
        remove_assert=args.remove_assert,
        nms_score_threshold=float(args.nms_score_threshold),
        image_ids_path=args.image_ids_path,
        reference_map=reference_map,
        map_error_threshold=float(args.map_error_threshold)
    )

    if not success:
        sys.exit(1)
