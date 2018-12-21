import argparse
import ast
import sys
import re


def parse_file(filename):
    with(open(filename)) as f:
        f_data = f.read()
    results = {}

    def regex_match(regex):
        match=re.match(regex, f_data, re.DOTALL)
        if match is not None:
            results[match.group(1)] = match.group(2)

    regex_match('.*(num_nodes\(native_tf\)): (\d*)')
    regex_match('.*(num_nodes\(tftrt_total\)): (\d*)')
    regex_match('.*(num_nodes\(trt_only\)): (\d*)')


    assert len(results) == 3, '{}'.format(results)
    return results

def check_nodes(res, filename, model, tol):
    dest_res_tftrt_total = {
        'mobilenet_v1': 12,
        'mobilenet_v2': 14,
        'nasnet_large': 50,
        'nasnet_mobile': 44,
        'resnet_50_v1': 15,
        'resnet_50_v2': 15,
        'vgg16': 10,
        'vgg19': 10,
        'inception_v3': 12,
        'inception_v4': 21
    }

    dest_res_trt_only = {
        'mobilenet_v1': 1,
        'mobilenet_v2': 1,
        'nasnet_large': 4,
        'nasnet_mobile': 4,
        'resnet_50_v1': 2,
        'resnet_50_v2': 2,
        'vgg16': 1,
        'vgg19': 1,
        'inception_v3': 1,
        'inception_v4': 2
    }

    if dest_res_trt_only[model] == int(res['num_nodes(trt_only)']):
        print("trt nodes only")
        print("PASS")
    else:
        print("FAIL")
        print("number of trt nodes {} vs. {}".format(res['num_nodes(trt_only)'], dest_res_trt_only[model]))
        sys.exit(1)

    
    if abs(int(res['num_nodes(tftrt_total)']) - int(dest_res_tftrt_total[model])) < tol*dest_res_tftrt_total[model]:
        print("tftrt nodes total")
        print("PASS")
    else:
        print("FAIL")
        print("total number of nodes {} vs. {}".format(res['num_nodes(tftrt_total)'], dest_res_tftrt_total[model]))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--tolerance', default=0.1)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--precision', default='tf_fp32')
    parser.add_argument('--input_path')
    parser.add_argument('--dynamic_op', default=False)

    args = parser.parse_args()

    fn = 'output_' + args.precision + '_bs' + str(args.batch_size) + '_' + args.model + '_dynamic_op=' + str(args.dynamic_op)

    print()
    print('checking nodes...')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    res = parse_file((args.input_path + '/' + fn))
    check_nodes(res, fn, args.model, args.tolerance)

if __name__ == "__main__":
    main()



