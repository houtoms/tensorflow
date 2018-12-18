import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_accuracy(annotation_path, detections_file_path, image_ids=None):

    cocoGt = COCO(annotation_file=annotation_path)
    cocoDt = cocoGt.loadRes(detections_file_path)
    eval = COCOeval(cocoGt, cocoDt, 'bbox')

    if image_ids is not None:
        eval.params.imgIds = image_ids

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return {'raw_stats': list(eval.stats), 'map': eval.stats[0]}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_file')
    parser.add_argument('detections_file_path')
    parser.add_argument('--accuracy_file', default=None)
    parser.add_argument('--image_ids_path', default='image_ids.json')
    args = parser.parse_args()

    with open(args.image_ids_path, 'r') as f:
        image_ids = json.load(f)

    accuracy = compute_accuracy(args.annotation_file, args.detections_file_path, image_ids=image_ids)

    if args.summary_file is not None:
        with open(args.summary_file, 'w') as f:
            json.dump(accuracy)
