import argparse
import json
import os


def get_map_pair(model_dir, reference_map_path):

    model_build_name = os.path.basename(model_dir)

    with open(reference_map_path, 'r') as f:
        reference_map = json.load(f)
        gt_map = None
        for ref_model_name, ref_map in reference_map.items():
            # ref_model_name should be substring of model_build_name
            # and only one ref_model_name should be substring od model_build_name
            if ref_model_name in model_build_name:
                gt_map = ref_map
                break
        if gt_map is None:
            raise RuntimeError('Match for %s not found in reference map file' % model_build_name)

    summary_file = os.path.join(model_dir, 'summary.json')

    with open(summary_file, 'r') as f:
        summary = json.load(f)
        map = summary['stats'][0]

    return gt_map, map


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('reference_map_path')
    parser.add_argument('--map_threshold', default=0.001)
    args = parser.parse_args()
    print(args)

    thresh = float(args.map_threshold)

    gt_map, map = get_map_pair(args.model_dir, args.reference_map_path)
    print('Reference MAP: %s' % gt_map)
    print('%s MAP: %s' % (args.model_dir, map))
    print('Difference: %f' % abs(gt_map - map))
    if abs(gt_map - map) > thresh:
        raise ValueError('MAP does not match reference value within %s' % thresh)
