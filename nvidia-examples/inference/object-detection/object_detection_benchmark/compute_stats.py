import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_stats(annotation_path, results_path, image_ids=None):

    cocoGt = COCO(annotation_file=annotation_path)
    cocoDt = cocoGt.loadRes(results_path)
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
    parser.add_argument('results_file')
    parser.add_argument('--summary_file', default=None)
    parser.add_argument('--image_ids_path', default='image_ids.json')
    args = parser.parse_args()

    with open(args.image_ids_path, 'r') as f:
        image_ids = json.load(f)

    stats = compute_stats(args.annotation_file, args.results_file, image_ids=image_ids)

    if args.summary_file is not None:
        with open(args.summary_file, 'w') as f:
            json.dump(stats)
